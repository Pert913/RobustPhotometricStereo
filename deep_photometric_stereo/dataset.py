"""
Dataset Loaders for Single-Image Photometric Stereo.
Supports both DiLiGenT (Real) and Synthetic (PRPS) datasets using RGB-Multiplexing simulation.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import glob
import random
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


def _canonical_led_directions(tilt_deg: float = 30.0) -> np.ndarray:
    """
    24 unit-vectors for the canonical LED ring. LED 0 sits at the bottom of the
    ring and indices advance counter-clockwise as viewed from the camera. LEDs
    0-7 form the R arc, 8-15 G, 16-23 B (3 contiguous 8-LED arcs).
    Frame: +x right, +y up, +z from object toward camera. The +z component is
    cos(tilt) so all LEDs sit on the camera side of the object.
    """
    theta = np.deg2rad(float(tilt_deg))
    phi = np.deg2rad(-90.0 + np.arange(24) * (360.0 / 24.0)).astype(np.float32)
    return np.stack([
        (np.sin(theta) * np.cos(phi)).astype(np.float32),
        (np.sin(theta) * np.sin(phi)).astype(np.float32),
        np.full(24, np.cos(theta), dtype=np.float32),
    ], axis=1)


def _match_canonical(available_dirs: np.ndarray, canonical_dirs: np.ndarray) -> np.ndarray:
    """
    For each canonical direction, return the index of the available direction
    with the largest cosine similarity. Returns (K,) int array of indices into
    available_dirs. Indices may repeat when len(available_dirs) < K.
    """
    a = np.asarray(available_dirs, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    c = np.asarray(canonical_dirs, dtype=np.float32)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    sims = c @ a.T  # (K, N)
    return np.argmax(sims, axis=1).astype(np.int32)


def _load_prps_light_dirs_cam(cam_folder: str) -> np.ndarray:
    """
    For a PRPS cam folder '.../<material>/cam_NNNNN', return (N, 3) unit vectors
    pointing from the object (assumed at world origin) to each point light, in
    camera-frame coordinates.

    Reads <material>/point_lights.config ('name x y z intensity') and
    <material>/cams.config ('name 3-floats 16-floats(4x4 cam-to-world)'). Falls
    back to normalized world positions if either config is missing or the cam
    row can't be parsed.

    Returns None if point_lights.config is missing entirely.
    """
    material_dir = os.path.dirname(cam_folder.rstrip('/\\'))
    cam_name = os.path.basename(cam_folder.rstrip('/\\'))

    pl_cfg = os.path.join(material_dir, "point_lights.config")
    if not os.path.exists(pl_cfg):
        return None
    positions = []
    with open(pl_cfg, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].startswith("point_light_"):
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not positions:
        return None
    L_world = np.array(positions, dtype=np.float32)  # (N, 3)

    cam_R_c2w = None
    cam_pos = None
    cam_cfg = os.path.join(material_dir, "cams.config")
    if os.path.exists(cam_cfg):
        with open(cam_cfg, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != cam_name:
                    continue
                if len(parts) >= 1 + 3 + 16:
                    mat = np.array([float(x) for x in parts[4:4 + 16]], dtype=np.float32).reshape(4, 4)
                    cam_R_c2w = mat[:3, :3]
                    cam_pos = mat[:3, 3]
                break

    if cam_R_c2w is None:
        # World-frame fallback: just normalize positions.
        norms = np.linalg.norm(L_world, axis=1, keepdims=True)
        return L_world / (norms + 1e-8)

    # Direction from object (world origin) to light in cam frame.
    # vec_world = L_world - 0; vec_cam = R_c2w.T @ vec_world.
    vec_cam = L_world @ cam_R_c2w  # equivalent to (R.T @ L.T).T
    norms = np.linalg.norm(vec_cam, axis=1, keepdims=True)
    dirs = vec_cam / (norms + 1e-8)
    # Sanity-flip z if the renderer's convention places "toward camera" at -z.
    if dirs[:, 2].mean() < 0:
        dirs[:, 2] *= -1.0
    return dirs


def _ring_mix_channels(point_imgs, env_rgb, ring_indices, alpha, b_weights, dark_noise_std: float = 0.0):
    """
    Compose the 3-channel ring-mixed image:
        I_c_train     = alpha * I_c_env + (1 - alpha) * sum_{i=0..7} b_{c,i} * I_point[ring_indices[c*8+i]]
        I_c_lights_off = alpha * I_c_env + N(0, dark_noise_std^2)   (per-pixel iid)
        I_c_sub       = I_c_train - I_c_lights_off
    The noise simulates the read/shot noise present in a real dark-frame capture;
    in noiseless arithmetic I_sub reduces to (1-alpha) * sum_c b_{c,i} * I_point.
    Returns I_sub (3, H, W) float32. env_rgb may be None (treated as zeros).
    """
    n_per_channel = 8
    H, W = point_imgs.shape[1], point_imgs.shape[2]
    I_point = np.zeros((3, H, W), dtype=np.float32)
    for c in range(3):
        for i in range(n_per_channel):
            idx = int(ring_indices[c * n_per_channel + i])
            I_point[c] += float(b_weights[c, i]) * point_imgs[idx]
    if env_rgb is None:
        I_env = np.zeros((3, H, W), dtype=np.float32)
    else:
        I_env = env_rgb.astype(np.float32, copy=False)
    I_train = alpha * I_env + (1.0 - alpha) * I_point
    I_lights_off = alpha * I_env
    if dark_noise_std > 0.0:
        I_lights_off = I_lights_off + np.random.normal(
            0.0, dark_noise_std, size=I_lights_off.shape
        ).astype(np.float32)
    return I_train - I_lights_off


def _max_spread_indices(dirs, k=3):
    """
    Given an (N, 3) array of unit light-direction vectors, return k indices
    whose directions maximise angular spread.
    Criterion: maximise |det([l_i, l_j, l_k])| over all C(N, k) triples,
    i.e. the volume of the parallelepiped — the tightest single proxy for spread.
    Falls back to evenly-spaced indices when N < k.
    """
    from itertools import combinations
    dirs = np.asarray(dirs, dtype=np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-8)
    n = len(dirs)
    if n < k:
        return list(range(n))
    idx_all = np.array(list(combinations(range(n), k)), dtype=np.int32)
    triplets = dirs[idx_all]                      # (C(N,3), 3, 3)
    dets = np.abs(np.linalg.det(triplets))        # (C(N,3),)
    return sorted(idx_all[int(np.argmax(dets))].tolist())


class DiLiGentDataset(Dataset):
    """
    Training Dataset Loader for DiLiGenT.
    Simulates a 3-light color multiplexing system by randomly selecting 
    3 grayscale images and stacking them into the RGB channels.
    """
    def __init__(
        self,
        data_root: str,
        objects: list,
        patch_size: int = 128,
        patches_per_epoch: int = 2000,
        augment: bool = True,
        use_ring_mixing: bool = True,
        ring_tilt_deg: float = 30.0,
        alpha_min: float = 0.4,
        alpha_max: float = 0.7,
        dark_noise_std: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1.0,
    ):
        self.data_root = data_root
        self.objects = objects
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment
        self.K_train = 3 # Take 3 images to simulate 3 light clusters (R, G, B)
        self.use_ring_mixing = use_ring_mixing
        self.ring_tilt_deg = ring_tilt_deg
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.dark_noise_std = float(dark_noise_std)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self._canonical = _canonical_led_directions(ring_tilt_deg) if use_ring_mixing else None

        self.object_info = []
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            info = self._scan_object(obj_dir, obj_name)
            if info is not None:
                self.object_info.append(info)

        if not self.object_info:
            raise RuntimeError(f"No valid objects found in {data_root} for {objects}")

        mode = "RingMix-24" if use_ring_mixing else "RGB-Multiplexing"
        print(f"DiLiGentDataset: {len(self.object_info)} objects, {mode}, patch={patch_size}, patches/epoch={patches_per_epoch}")

    def _scan_object(self, obj_dir: str, obj_name: str):
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path): 
            return None

        mask_path = os.path.join(obj_dir, "mask.npy")
        if not os.path.exists(mask_path):
            mask_path = None
            for mask_name in ["mask.png", "inboundary.png"]:
                mp = os.path.join(obj_dir, mask_name)
                if os.path.exists(mp):
                    mask_path = mp
                    break

        image_paths = self._scan_images(obj_dir)
        if not image_paths or len(image_paths) < 3:
            return None # Requires at least 3 images to simulate RGB channels

        light_dirs = None
        ring_indices = None
        dirs_path = os.path.join(obj_dir, "light_directions.txt")
        if os.path.exists(dirs_path):
            try:
                ld = np.loadtxt(dirs_path, dtype=np.float32)
                if ld.ndim == 2 and ld.shape[1] == 3 and ld.shape[0] == len(image_paths):
                    norms = np.linalg.norm(ld, axis=1, keepdims=True)
                    light_dirs = (ld / (norms + 1e-8)).astype(np.float32)
                    if self._canonical is not None:
                        ring_indices = _match_canonical(light_dirs, self._canonical)
            except Exception:
                light_dirs = None
                ring_indices = None

        return {
            "name": obj_name,
            "dir": obj_dir,
            "normal_path": normal_path,
            "mask_path": mask_path,
            "image_paths": image_paths,
            "light_dirs": light_dirs,
            "ring_indices": ring_indices,
        }

    @staticmethod
    def _scan_images(obj_dir):
        non_image_names = {"Normal_gt.png", "mask.png", "inboundary.png", "onboundary.png", "gt_normal.tif"}
        def _is_numbered_image(path):
            name = os.path.basename(path)
            return name not in non_image_names and name[0].isdigit()

        direct_pngs = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
        image_pngs = [p for p in direct_pngs if _is_numbered_image(p)]
        if image_pngs: return image_pngs

        png_dirs = glob.glob(os.path.join(obj_dir, "*PNG"))
        if png_dirs:
            pngs = sorted(glob.glob(os.path.join(png_dirs[0], "*.png")))
            if pngs: return pngs

        for subname in ["images_specular", "images_metallic"]:
            sub = os.path.join(obj_dir, subname)
            if os.path.isdir(sub):
                tifs = sorted(glob.glob(os.path.join(sub, "*.tif")))
                if tifs: return tifs

        for subname in sorted(os.listdir(obj_dir)):
            sub = os.path.join(obj_dir, subname)
            if not os.path.isdir(sub): continue
            npy_paths = sorted(glob.glob(os.path.join(sub, "*.npy")))
            if npy_paths: return npy_paths
            tif_paths = sorted(glob.glob(os.path.join(sub, "*.tif")))
            if tif_paths: return tif_paths
        return None

    @staticmethod
    def _load_single_image_as_gray(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            img = np.load(path).astype(np.float32)
            if img.ndim == 3: img = img.mean(axis=-1)
            if img.max() > 1.5: img = img / 255.0
        elif ext in [".tif", ".tiff"]:
            try:
                import tifffile
                img = tifffile.imread(path).astype(np.float32)
            except ImportError:
                img = np.array(Image.open(path)).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                else: img = img[:, :, 0]
            if img.max() > 1.5:
                img = img / img.max() if img.max() > 0 else img
        else:
            img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        return img

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        obj_info = random.choice(self.object_info)
        normal_gt = np.load(obj_info["normal_path"]).astype(np.float32)
        
        if obj_info["mask_path"]:
            ext = os.path.splitext(obj_info["mask_path"])[1].lower()
            if ext == ".npy":
                mask = np.load(obj_info["mask_path"]).astype(np.float32)
            else:
                mask = np.array(Image.open(obj_info["mask_path"]).convert("L")).astype(np.float32)
                mask = (mask > 128).astype(np.float32)
        else:
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        if mask.ndim != 2: mask = mask.reshape(normal_gt.shape[:2])
        if mask.dtype != np.float32:
            mask = (mask > 0.5).astype(np.float32) if mask.max() <= 1.0 else (mask > 128).astype(np.float32)

        H, W = normal_gt.shape[:2]
        ps = self.patch_size

        valid_coords = np.argwhere(mask > 0.5)
        if len(valid_coords) > 0:
            rand_idx = random.randint(0, len(valid_coords) - 1)
            center_y, center_x = valid_coords[rand_idx]
            y = max(0, min(center_y - (ps // 2), H - ps))
            x = max(0, min(center_x - (ps // 2), W - ps))
        else:
            y = random.randint(0, max(0, H - ps))
            x = random.randint(0, max(0, W - ps))
            
        mask_patch = mask[y:y+ps, x:x+ps]
        normal_patch = normal_gt[y:y+ps, x:x+ps, :]

        image_paths = obj_info["image_paths"]
        total_imgs = len(image_paths)

        if self.use_ring_mixing:
            ring_indices = obj_info.get("ring_indices")
            if ring_indices is None:
                # No light_directions.txt or wrong shape — fall back to a random
                # 24-index draw (with replacement when fewer than 24 lights).
                if total_imgs >= 24:
                    ring_indices = np.array(random.sample(range(total_imgs), 24), dtype=np.int32)
                else:
                    ring_indices = np.array(
                        [random.randrange(total_imgs) for _ in range(24)], dtype=np.int32
                    )

            unique_idx = sorted(set(int(i) for i in ring_indices))
            local_pos = {gi: li for li, gi in enumerate(unique_idx)}
            point_imgs = np.stack([
                self._load_single_image_as_gray(image_paths[gi])[y:y+ps, x:x+ps]
                for gi in unique_idx
            ], axis=0).astype(np.float32)
            local_ring = np.array([local_pos[int(gi)] for gi in ring_indices], dtype=np.int32)

            b_weights = np.random.uniform(0.0, 1.0, size=(3, 8)).astype(np.float32)
            alpha = float(np.random.uniform(self.alpha_min, self.alpha_max))
            img_patch_rgb = _ring_mix_channels(
                point_imgs, None, local_ring, alpha, b_weights,
                dark_noise_std=self.dark_noise_std,
            )
        else:
            # Legacy: pick 3 random light angles, stack as RGB.
            indices = sorted(random.sample(range(total_imgs), 3))
            channels = []
            for i in indices:
                img_gray = self._load_single_image_as_gray(image_paths[i])
                channels.append(img_gray[y:y+ps, x:x+ps])
            img_patch_rgb = np.stack(channels, axis=0)

        img_patch_rgb = img_patch_rgb.astype(np.float32, copy=False)

        # Random gamma applied in raw-intensity space, before z-score. Negatives
        # (from dark-noise) are clipped to 0 so the power is well-defined.
        if self.augment and self.gamma_min < self.gamma_max:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            img_patch_rgb = np.maximum(img_patch_rgb, 0.0).astype(np.float32) ** np.float32(gamma)

        # Normalize each channel
        mask_bool = mask_patch > 0.5
        for c in range(3):
            if mask_bool.any():
                m = img_patch_rgb[c][mask_bool].mean()
                s = img_patch_rgb[c][mask_bool].std() + 1e-8
                img_patch_rgb[c] = (img_patch_rgb[c] - m) / s

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img_patch_rgb = img_patch_rgb[:, :, ::-1].copy()
                normal_patch = normal_patch[:, ::-1, :].copy()
                normal_patch[:, :, 0] = -normal_patch[:, :, 0]
                mask_patch = mask_patch[:, ::-1].copy()

            # Per-channel brightness jitter — simulates real LED ring imbalance and
            # partial AWB correction from phone cameras (e.g. blue channel 2x brighter).
            # Applied BEFORE z-score so z-score can correct for it, matching inference.
            if random.random() > 0.3:
                for c in range(3):
                    img_patch_rgb[c] *= random.uniform(0.5, 2.0)
                # Re-normalise after jitter so the z-score baseline is preserved
                for c in range(3):
                    if mask_bool.any():
                        m = img_patch_rgb[c][mask_bool].mean()
                        s = img_patch_rgb[c][mask_bool].std() + 1e-8
                        img_patch_rgb[c] = (img_patch_rgb[c] - m) / s

            # Specular highlight simulation — random Gaussian blob in one channel.
            # Teaches the model to ignore specular hot-spots on shiny/plastic objects.
            if random.random() > 0.5:
                ps = img_patch_rgb.shape[1]
                c = random.randint(0, 2)
                cy, cx = random.randint(0, ps - 1), random.randint(0, ps - 1)
                sigma = random.uniform(3, ps // 4)
                strength = random.uniform(1.0, 3.0)
                yy, xx = np.meshgrid(np.arange(ps), np.arange(ps), indexing='ij')
                blob = strength * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
                img_patch_rgb[c] = img_patch_rgb[c] + blob.astype(np.float32)

        img_tensor = torch.from_numpy(img_patch_rgb.copy()).float()
        normal_tensor = torch.from_numpy(normal_patch).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)

        return img_tensor, normal_tensor, mask_tensor


class DiLiGentTestDataset(Dataset):
    def __init__(self, data_root: str, objects: list):
        self.data_root = data_root
        self.objects = objects
        self.object_info = []

        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            info = self._scan_object(obj_dir, obj_name)
            if info is not None:
                self.object_info.append(info)

    def _scan_object(self, obj_dir, obj_name):
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path): 
            return None

        mask_path = os.path.join(obj_dir, "mask.npy")
        if not os.path.exists(mask_path): 
            mask_path = None

        image_paths = DiLiGentDataset._scan_images(obj_dir)
        if not image_paths or len(image_paths) < 3: 
            return None

        return {
            "name": obj_name, 
            "normal_path": normal_path, 
            "mask_path": mask_path, 
            "image_paths": image_paths
        }

    def __len__(self): 
        return len(self.object_info)

    def __getitem__(self, idx):
        obj_info = self.object_info[idx]
        normal_gt = np.load(obj_info["normal_path"]).astype(np.float32)
        
        if obj_info["mask_path"]:
            ext = os.path.splitext(obj_info["mask_path"])[1].lower()
            if ext == ".npy":
                mask = np.load(obj_info["mask_path"]).astype(np.float32)
            else:
                mask = np.array(Image.open(obj_info["mask_path"]).convert("L")).astype(np.float32)
                mask = (mask > 128).astype(np.float32)
        else: 
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        if mask.ndim != 2: 
            mask = mask.reshape(normal_gt.shape[:2])
            
        if mask.dtype != np.float32: 
            mask = (mask > 0.5).astype(np.float32) if mask.max() <= 1.0 else (mask > 128).astype(np.float32)

        # Fix the first 3 light angles for consistent testing
        image_paths = obj_info["image_paths"]
        random.seed(42 + idx)
        indices = sorted(random.sample(range(len(image_paths)), 3))
        random.seed()
        
        channels = []
        for i in indices:
            img_gray = DiLiGentDataset._load_single_image_as_gray(image_paths[i])
            channels.append(img_gray)
            
        test_img_rgb = np.stack(channels, axis=0)

        mask_bool = mask > 0.5
        for c in range(3):
            if mask_bool.any():
                m = test_img_rgb[c][mask_bool].mean()
                s = test_img_rgb[c][mask_bool].std() + 1e-8
                test_img_rgb[c] = (test_img_rgb[c] - m) / s

        img_tensor = torch.from_numpy(test_img_rgb).float()
        normal_tensor = torch.from_numpy(normal_gt).float().permute(2, 0, 1)  
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  
        
        return img_tensor, normal_tensor, mask_tensor, obj_info["name"]


class DiLiGentMVTestDataset(Dataset):
    """
    Test dataset for DiLiGenT-MV benchmark.
    Structure: mvpmsData/{object}/view_{nn}/{001..096}.png + Normal_gt.mat + mask.png
    Each sample is one (object, view) pair.
    """

    def __init__(self, mv_root: str, objects: list = None, views_per_object: int = None,
                 use_ring_aggregation: bool = True, ring_tilt_deg: float = 30.0):
        """
        Args:
            mv_root: path to mvpmsData directory
            objects: list of object names (e.g. ['bearPNG', 'cowPNG']); None = all
            views_per_object: max views to use per object (None = all)
            use_ring_aggregation: match all available LEDs to the 24 canonical ring
                directions and sum 8 per R/G/B arc into 3 channels (uniform b=1),
                mirroring the ring-mixing training distribution. When False (or when
                light_directions.txt is missing / <24 images), falls back to the
                legacy 3-max-spread single-LED sampling.
            ring_tilt_deg: canonical ring tilt used for LED matching.
        """
        try:
            import scipy.io as sio
            self._sio = sio
        except ImportError:
            raise ImportError("scipy is required for DiLiGenT-MV: pip install scipy")

        self.mv_root = mv_root
        self.use_ring_aggregation = use_ring_aggregation
        self.ring_tilt_deg = ring_tilt_deg
        self._canonical = _canonical_led_directions(ring_tilt_deg) if use_ring_aggregation else None
        self.samples = []  # list of (obj_name, view_name, view_dir)

        all_objects = sorted(os.listdir(mv_root)) if objects is None else objects
        for obj in all_objects:
            obj_dir = os.path.join(mv_root, obj)
            if not os.path.isdir(obj_dir):
                continue
            view_dirs = sorted(
                d for d in os.listdir(obj_dir)
                if d.startswith("view_") and os.path.isdir(os.path.join(obj_dir, d))
            )
            if views_per_object is not None:
                view_dirs = view_dirs[:views_per_object]
            for view in view_dirs:
                view_path = os.path.join(obj_dir, view)
                mat_path = os.path.join(view_path, "Normal_gt.mat")
                mask_path = os.path.join(view_path, "mask.png")
                if os.path.exists(mat_path) and os.path.exists(mask_path):
                    self.samples.append((obj, view, view_path))

        mode = "RingAgg-24" if use_ring_aggregation else "3-max-spread"
        print(f"DiLiGentMVTestDataset: {len(self.samples)} view samples across {len(all_objects)} objects, {mode}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_light_dirs(light_dirs_path):
        """Load (N,3) unit direction vectors from a light_directions.txt file."""
        if not os.path.exists(light_dirs_path):
            return None
        dirs = np.loadtxt(light_dirs_path, dtype=np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs / (norms + 1e-8)

    def __getitem__(self, idx):
        obj_name, view_name, view_dir = self.samples[idx]

        # Load ground-truth normal (H, W, 3) in [-1, 1]
        mat = self._sio.loadmat(os.path.join(view_dir, "Normal_gt.mat"))
        normal_gt = mat["Normal_gt"].astype(np.float32)  # already unit normals

        # Load mask (H, W) in {0, 255} -> {0, 1}
        mask = np.array(Image.open(os.path.join(view_dir, "mask.png")).convert("L")).astype(np.float32)
        mask = (mask > 128).astype(np.float32)

        # Collect all numbered images
        all_pngs = sorted(
            p for p in glob.glob(os.path.join(view_dir, "*.png"))
            if os.path.basename(p)[0].isdigit()
        )

        light_dirs_path = os.path.join(view_dir, "light_directions.txt")
        dirs = self._load_light_dirs(light_dirs_path)

        if self.use_ring_aggregation and dirs is not None and len(all_pngs) >= 24:
            # Ring aggregation: match every available LED to the 24 canonical ring
            # directions, then sum 8 LEDs per R/G/B arc into 3 channels with uniform
            # weight (b=1). This mirrors the ring-mixing training distribution; the
            # per-channel z-score below absorbs the resulting scale difference.
            ring_indices = _match_canonical(dirs, self._canonical)  # (24,) into all_pngs
            unique_idx = sorted(set(int(i) for i in ring_indices))
            cache = {}
            for gi in unique_idx:
                cache[gi] = np.array(
                    Image.open(all_pngs[gi]).convert("L")
                ).astype(np.float32) / 255.0
            H0, W0 = next(iter(cache.values())).shape
            test_img_rgb = np.zeros((3, H0, W0), dtype=np.float32)
            for c in range(3):
                for i in range(8):
                    gi = int(ring_indices[c * 8 + i])
                    test_img_rgb[c] += cache[gi]
        else:
            # Legacy: 3 single LEDs whose directions are most spread apart.
            if dirs is not None:
                indices = _max_spread_indices(dirs)
            else:
                step = max(1, len(all_pngs) // 3)
                indices = [0, step, 2 * step]

            channels = []
            for i in indices:
                img = np.array(Image.open(all_pngs[i]).convert("L")).astype(np.float32) / 255.0
                channels.append(img)
            test_img_rgb = np.stack(channels, axis=0)  # (3, H, W)

        # Per-channel z-score normalisation on foreground pixels
        mask_bool = mask > 0.5
        for c in range(3):
            if mask_bool.any():
                m = test_img_rgb[c][mask_bool].mean()
                s = test_img_rgb[c][mask_bool].std() + 1e-8
                test_img_rgb[c] = (test_img_rgb[c] - m) / s

        img_tensor = torch.from_numpy(test_img_rgb).float()
        normal_tensor = torch.from_numpy(normal_gt).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        name = f"{obj_name}_{view_name}"
        return img_tensor, normal_tensor, mask_tensor, name


class SyntheticDataset(Dataset):
    def __init__(
        self,
        json_path,
        patch_size=256,
        alpha_min=0.4,
        alpha_max=0.7,
        augment=True,
        use_ring_mixing: bool = True,
        ring_tilt_deg: float = 30.0,
        dark_noise_std: float = 0.0,
        gamma_min: float = 1.0,
        gamma_max: float = 1.0,
    ):
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.patch_size = patch_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.augment = augment
        self.use_ring_mixing = use_ring_mixing
        self.ring_tilt_deg = ring_tilt_deg
        self.dark_noise_std = float(dark_noise_std)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self._canonical = _canonical_led_directions(ring_tilt_deg) if use_ring_mixing else None
        self._dirs_cache: dict = {}  # cam_folder -> (ring_indices, n_lights) or (None, n_lights)
        mode = "RingMix-24" if use_ring_mixing else "3-random"
        print(f"SyntheticDataset: {len(self.folder_list)} folders, {mode}, patch={patch_size}")

    def _ring_indices_for(self, cam_folder: str, n_available: int):
        """Cached lookup: returns 24-int ring indices for this cam folder, or None
        if directions aren't available (caller should fall back to random)."""
        if not self.use_ring_mixing:
            return None
        cached = self._dirs_cache.get(cam_folder)
        if cached is not None:
            return cached
        dirs = _load_prps_light_dirs_cam(cam_folder)
        if dirs is None or len(dirs) == 0:
            self._dirs_cache[cam_folder] = None
            return None
        # Clip to lights actually present on disk (n_available may be < len(dirs)
        # if some renders are missing).
        if n_available < len(dirs):
            dirs = dirs[:n_available]
        ring = _match_canonical(dirs, self._canonical)
        self._dirs_cache[cam_folder] = ring
        return ring

    def __len__(self): 
        return len(self.folder_list)

    def load_exr(self, path):
        if not os.path.exists(path): 
            return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: 
                return None
            if len(img.shape) == 2: 
                img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except: 
            return None

    def read_light_means(self, folder_path):
        mean_ip, mean_ienv = 1.0, 1.0
        clean_path = folder_path.rstrip('/\\')
        parent_dir = os.path.dirname(clean_path)
        config_path = os.path.join(parent_dir, "light_means.config")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split() 
                        if len(parts) >= 2:
                            if 'point_mean' in parts[0]: 
                                mean_ip = float(parts[1])
                            elif 'env_mean' in parts[0]: 
                                mean_ienv = float(parts[1])
            except: 
                pass
                
        return mean_ip, mean_ienv

    def __getitem__(self, idx):
        attempts = 0
        ps = self.patch_size
        
        while attempts < 20:
            folder_path = self.folder_list[idx]
            try:
                normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
                mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
                
                if normal_gt is None or mask is None: 
                    raise FileNotFoundError

                normal_gt = normal_gt[:, :, ::-1]       
                normal_gt = normal_gt * 2.0 - 1.0       
                norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
                normal_gt = normal_gt / norm
                if mask.ndim == 3: 
                    mask = mask[:, :, 0]

                H, W = normal_gt.shape[:2]
                y, x = random.randint(0, H-ps), random.randint(0, W-ps)
                
                for _ in range(30):
                    ty, tx = random.randint(0, H-ps), random.randint(0, W-ps)
                    if mask[ty:ty+ps, tx:tx+ps].sum() > ps*ps*0.05:
                        y, x = ty, tx
                        break

                mean_ip, mean_ienv = self.read_light_means(folder_path)

                all_files = os.listdir(folder_path)
                pl_files = sorted(
                    f for f in all_files if f.startswith("point_light_") and f.endswith(".exr")
                )
                env_files = sorted(
                    f for f in all_files if f.startswith("env_light_") and f.endswith(".exr")
                )
                if len(pl_files) < 3:
                    raise FileNotFoundError

                pl_indices = [int(f.split('_')[2].split('.')[0]) for f in pl_files]
                mask_patch = mask[y:y+ps, x:x+ps]
                mask_expanded = np.expand_dims(mask_patch, axis=0)

                if self.use_ring_mixing:
                    n_lights = len(pl_files)
                    ring = self._ring_indices_for(folder_path, n_lights)
                    if ring is None:
                        # Formatted-style folder with no light directions — random draw.
                        if n_lights >= 24:
                            ring = np.array(random.sample(range(n_lights), 24), dtype=np.int32)
                        else:
                            ring = np.array(
                                [random.randrange(n_lights) for _ in range(24)], dtype=np.int32
                            )

                    unique = sorted(set(int(i) for i in ring))
                    local_pos = {gi: li for li, gi in enumerate(unique)}
                    point_imgs = []
                    for gi in unique:
                        abs_idx = pl_indices[gi]
                        img_p = self.load_exr(os.path.join(folder_path, f"point_light_{abs_idx:05d}.exr"))
                        if img_p is None:
                            raise FileNotFoundError
                        patch_p = img_p[y:y+ps, x:x+ps] / (mean_ip + 1e-8)
                        gray = 0.2989 * patch_p[:, :, 2] + 0.5870 * patch_p[:, :, 1] + 0.1140 * patch_p[:, :, 0]
                        point_imgs.append(gray.astype(np.float32))
                    point_imgs = np.stack(point_imgs, axis=0)
                    local_ring = np.array([local_pos[int(gi)] for gi in ring], dtype=np.int32)

                    env_rgb = None
                    if env_files:
                        env_idx = int(random.choice(env_files).split('_')[2].split('.')[0])
                        img_e = self.load_exr(
                            os.path.join(folder_path, f"env_light_{env_idx:05d}.exr")
                        )
                        if img_e is not None:
                            patch_e = img_e[y:y+ps, x:x+ps] / (mean_ienv + 1e-8)
                            env_rgb = patch_e[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)

                    b_weights = np.random.uniform(0.0, 1.0, size=(3, 8)).astype(np.float32)
                    alpha = float(np.random.uniform(self.alpha_min, self.alpha_max))
                    object_lights_rgb = _ring_mix_channels(
                        point_imgs, env_rgb, local_ring, alpha, b_weights,
                        dark_noise_std=self.dark_noise_std,
                    )

                    # Background fill: when an env light was loaded, paste it outside the
                    # object so the seg branch sees a non-zero background like before.
                    if env_rgb is not None:
                        input_patch = (object_lights_rgb * mask_expanded) + (env_rgb * (1.0 - mask_expanded))
                    else:
                        input_patch = object_lights_rgb * mask_expanded
                    input_patch = input_patch.astype(np.float32, copy=False)
                else:
                    sampled_indices = random.sample(pl_indices, 3)
                    channels = []
                    for i in sampled_indices:
                        img_p = self.load_exr(os.path.join(folder_path, f"point_light_{i:05d}.exr"))
                        if img_p is None:
                            raise FileNotFoundError
                        patch_p = img_p[y:y+ps, x:x+ps] / (mean_ip + 1e-8)
                        gray = 0.2989 * patch_p[:, :, 2] + 0.5870 * patch_p[:, :, 1] + 0.1140 * patch_p[:, :, 0]
                        channels.append(gray)
                    object_3_lights_rgb = np.stack(channels, axis=0)
                    env_idx = random.randint(1, 10)
                    img_e = self.load_exr(os.path.join(folder_path, f"env_light_{env_idx:05d}.exr"))
                    if img_e is None:
                        raise FileNotFoundError
                    patch_e = img_e[y:y+ps, x:x+ps] / (mean_ienv + 1e-8)
                    patch_e_rgb = patch_e[:, :, ::-1].transpose(2, 0, 1)
                    input_patch = (object_3_lights_rgb * mask_expanded) + (patch_e_rgb * (1.0 - mask_expanded))
                    input_patch = input_patch.astype(np.float32, copy=False)

                # Random gamma in raw-intensity space, before clip + z-score.
                if self.augment and self.gamma_min < self.gamma_max:
                    gamma = random.uniform(self.gamma_min, self.gamma_max)
                    input_patch = np.maximum(input_patch, 0.0).astype(np.float32) ** np.float32(gamma)

                mask_bool = mask_patch > 0.5
                for c in range(3):
                    if mask_bool.any():
                        valid_pixels = input_patch[c, mask_bool]
                        p99 = np.percentile(valid_pixels, 99.0)
                        input_patch[c] = np.clip(input_patch[c], 0.0, p99)
                        
                        valid_pixels_clipped = input_patch[c, mask_bool]
                        m = valid_pixels_clipped.mean()
                        s = valid_pixels_clipped.std() + 1e-8
                        input_patch[c] = (input_patch[c] - m) / s

                normal_patch = normal_gt[y:y+ps, x:x+ps, :]

                return torch.from_numpy(input_patch.copy()), \
                       torch.from_numpy(normal_patch.copy()).permute(2, 0, 1), \
                       torch.from_numpy(mask_patch.copy()).unsqueeze(0)
            except:
                attempts += 1
                idx = random.randint(0, len(self.folder_list)-1)
        
        return torch.zeros((3, ps, ps)), torch.zeros((3, ps, ps)), torch.zeros((1, ps, ps))


class SyntheticTestDataset(Dataset):
    def __init__(self, json_path, alpha_min=0.35):
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.alpha_min = alpha_min

    def __len__(self): 
        return len(self.folder_list)

    def load_exr(self, path):
        if not os.path.exists(path): 
            return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: 
                return None
            if len(img.shape) == 2: 
                img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except: 
            return None

    def read_light_means(self, folder_path):
        mean_ip, mean_ienv = 1.0, 1.0
        clean_path = folder_path.rstrip('/\\')
        parent_dir = os.path.dirname(clean_path)
        config_path = os.path.join(parent_dir, "light_means.config")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split() 
                        if len(parts) >= 2:
                            if 'point_mean' in parts[0]: 
                                mean_ip = float(parts[1])
                            elif 'env_mean' in parts[0]: 
                                mean_ienv = float(parts[1])
            except: 
                pass
                
        return mean_ip, mean_ienv

    def __getitem__(self, idx):
        folder_path = self.folder_list[idx]
        normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
        mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
        
        if normal_gt is None or mask is None:
             return torch.zeros((3, 512, 512)), torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512)), "error"

        normal_gt = normal_gt[:, :, ::-1]
        normal_gt = normal_gt * 2.0 - 1.0
        norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
        normal_gt /= norm
        if mask.ndim == 3: 
            mask = mask[:, :, 0]
        
        mean_ip, mean_ienv = self.read_light_means(folder_path)
        
        all_files = os.listdir(folder_path)
        all_indices = [int(f.split('_')[2].split('.')[0]) for f in all_files if f.startswith("point_light_") and f.endswith(".exr")]
        
        random.seed(42 + idx)
        sampled_indices = random.sample(all_indices, 3) if len(all_indices) >= 3 else all_indices
        random.seed()

        channels = []
        for i in sampled_indices:
            img_p = self.load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
            if img_p is None: 
                channels.append(np.zeros_like(mask))
                continue
            img_p = img_p / (mean_ip + 1e-8)
            gray = 0.2989*img_p[:,:,2] + 0.5870*img_p[:,:,1] + 0.1140*img_p[:,:,0]
            channels.append(gray)

        object_3_lights_rgb = np.stack(channels, axis=0)
        
        img_e = self.load_exr(os.path.join(folder_path, f"env_light_00001.exr"))
        if img_e is not None:
            img_e = img_e / (mean_ienv + 1e-8)
            patch_e_rgb = img_e[:, :, ::-1].transpose(2, 0, 1)
        else:
            patch_e_rgb = np.zeros_like(object_3_lights_rgb)

        mask_expanded = np.expand_dims(mask, axis=0) 
        input_patch = (object_3_lights_rgb * mask_expanded) + (patch_e_rgb * (1.0 - mask_expanded))
        
        mask_bool = mask > 0.5
        for c in range(3):
            if mask_bool.any():
                valid_pixels = input_patch[c, mask_bool]
                p99 = np.percentile(valid_pixels, 99.0)
                input_patch[c] = np.clip(input_patch[c], 0.0, p99)
                
                valid_pixels_clipped = input_patch[c, mask_bool]
                m = valid_pixels_clipped.mean()
                s = valid_pixels_clipped.std() + 1e-8
                input_patch[c] = (input_patch[c] - m) / s

        name = os.path.basename(os.path.dirname(folder_path)) + "_" + os.path.basename(folder_path)
        
        return torch.from_numpy(input_patch.copy()), \
               torch.from_numpy(normal_gt.copy()).permute(2, 0, 1), \
               torch.from_numpy(mask.copy()).unsqueeze(0), name