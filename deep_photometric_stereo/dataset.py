"""
DiLiGenT dataset for uncalibrated photometric stereo.

Each sample: N grayscale images of one object ---> normal map ground truth.
Training uses random patches; testing uses full resolution.
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DiLiGentDataset(Dataset):
    """
    DiLiGenT dataset for patch-based training.

    Each __getitem__ returns:
        images: (N, 1, patch_h, patch_w) — N randomly sampled grayscale images
        normal_gt: (3, patch_h, patch_w) — ground truth normal patch
        mask: (1, patch_h, patch_w) — foreground mask patch
    """

    def __init__(
        self,
        data_root: str,
        objects: list,
        num_images: int = 32,
        min_images: int = 8,
        patch_size: int = 128,
        patches_per_epoch: int = 2000,
        augment: bool = True,
    ):
        self.data_root = data_root
        self.objects = objects
        self.num_images = num_images
        self.min_images = min_images
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        # Load all object data into memory
        self.object_data = []
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            data = self._load_object(obj_dir, obj_name)
            if data is not None:
                self.object_data.append(data)

        if not self.object_data:
            raise RuntimeError(f"No valid objects found in {data_root} for {objects}")

        print(f"DiLiGentDataset: {len(self.object_data)} objects, "
              f"N={min_images}-{num_images}, patch={patch_size}, "
              f"patches/epoch={patches_per_epoch}, augment={augment}")

    def _load_object(self, obj_dir: str, obj_name: str):
        """Load all images, normal GT, and mask for one object."""
        # Load normal GT
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path):
            print(f"  WARNING: {normal_path} not found, skipping {obj_name}")
            return None
        normal_gt = np.load(normal_path).astype(np.float32)  # (H, W, 3)

        # Load mask (try mask.npy, then mask.png, then inboundary.png)
        mask_path = os.path.join(obj_dir, "mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)  # (H, W)
        else:
            mask = None
            for mask_name in ["mask.png", "inboundary.png"]:
                mp = os.path.join(obj_dir, mask_name)
                if os.path.exists(mp):
                    mask = np.array(Image.open(mp).convert("L")).astype(np.float32)
                    mask = (mask > 128).astype(np.float32)
                    break
            if mask is None:
                print(f"  WARNING: No mask found for {obj_name}, using all pixels")
                mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        # Ensure mask is 2D float32 binary
        if mask.ndim != 2:
            mask = mask.reshape(normal_gt.shape[:2])
        if mask.dtype != np.float32:
            mask = (mask > 0.5).astype(np.float32) if mask.max() <= 1.0 else (mask > 128).astype(np.float32)

        # Load images — support multiple layouts:
        #   1. Flat: images (001.png..096.png) directly in obj_dir alongside metadata
        #   2. Subfolder: images in a *PNG subdirectory
        #   3. NPY subfolder: .npy images in a subdirectory
        images = self._load_images(obj_dir)

        if images is None or images.shape[0] == 0:
            print(f"  WARNING: No images found in {obj_dir}, skipping {obj_name}")
            return None

        print(f"  Loaded {obj_name}: {images.shape[0]} images, "
              f"shape={images.shape[1:]},"
              f" normal={normal_gt.shape}, mask_fg={int(mask.sum())}")

        return {
            "name": obj_name,
            "images": images,      # (96, H, W)
            "normal_gt": normal_gt, # (H, W, 3)
            "mask": mask,           # (H, W)
        }

    @staticmethod
    def _load_single_image(path):
        """Load a single image as grayscale float32 [0, 1]."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            img = np.load(path).astype(np.float32)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            if img.max() > 1.5:
                img = img / 255.0
        elif ext == ".tif" or ext == ".tiff":
            try:
                import tifffile
                img = tifffile.imread(path).astype(np.float32)
            except ImportError:
                img = np.array(Image.open(path)).astype(np.float32)
            if img.ndim == 3:
                # RGB/HDR -> grayscale via luminance
                if img.shape[2] >= 3:
                    img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                else:
                    img = img[:, :, 0]
            # Normalize HDR range to [0, 1]
            if img.max() > 1.5:
                img = img / img.max() if img.max() > 0 else img
        else:
            # PNG, JPG, etc.
            img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        return img

    @staticmethod
    def _load_images(obj_dir, max_images=None):
        """
        Load grayscale images from an object directory.
        Supports layouts:
          1. Flat numbered PNGs in obj_dir (DiLiGenT)
          2. *PNG subfolder (DiLiGenT)
          3. images_specular/ or images_metallic/ with .tif (PRPS)
          4. Subdirectory with .npy files (synthetic)
        """
        non_image_names = {"Normal_gt.png", "mask.png", "inboundary.png",
                           "onboundary.png", "gt_normal.tif"}

        def _is_numbered_image(path):
            name = os.path.basename(path)
            return name not in non_image_names and name[0].isdigit()

        def _load_from_paths(paths):
            if max_images:
                paths = paths[:max_images]
            img_list = []
            for p in paths:
                img = DiLiGentDataset._load_single_image(p)
                img_list.append(img)
            return np.stack(img_list, axis=0) if img_list else None

        # Strategy 1: Flat layout — numbered PNGs directly in obj_dir
        direct_pngs = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
        image_pngs = [p for p in direct_pngs if _is_numbered_image(p)]
        if image_pngs:
            return _load_from_paths(image_pngs)

        # Strategy 2: *PNG subfolder
        png_dirs = glob.glob(os.path.join(obj_dir, "*PNG"))
        if png_dirs:
            pngs = sorted(glob.glob(os.path.join(png_dirs[0], "*.png")))
            if pngs:
                return _load_from_paths(pngs)

        # Strategy 3: PRPS — images_specular/ or images_metallic/ with .tif
        for subname in ["images_specular", "images_metallic"]:
            sub = os.path.join(obj_dir, subname)
            if os.path.isdir(sub):
                tifs = sorted(glob.glob(os.path.join(sub, "*.tif")))
                if tifs:
                    return _load_from_paths(tifs)

        # Strategy 4: Any subdirectory with .npy or .tif files
        for subname in sorted(os.listdir(obj_dir)):
            sub = os.path.join(obj_dir, subname)
            if not os.path.isdir(sub):
                continue
            npy_paths = sorted(glob.glob(os.path.join(sub, "*.npy")))
            if npy_paths:
                return _load_from_paths(npy_paths)
            tif_paths = sorted(glob.glob(os.path.join(sub, "*.tif")))
            if tif_paths:
                return _load_from_paths(tif_paths)

        return None

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        # Pick a random object
        obj = random.choice(self.object_data)
        images = obj["images"]      # (96, H, W)
        normal_gt = obj["normal_gt"]  # (H, W, 3)
        mask = obj["mask"]            # (H, W)

        H, W = images.shape[1], images.shape[2]
        ps = self.patch_size

        # Random crop location — ensure at least some foreground pixels
        for _ in range(50):  # try up to 50 times to find a good patch
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
            mask_patch = mask[y:y+ps, x:x+ps]
            if mask_patch.sum() > ps * ps * 0.05:  # at least 5% foreground
                break

        # Randomly sample N images (variable N for robustness)
        if self.augment:
            n = random.randint(self.min_images, self.num_images)
        else:
            n = self.num_images
        total_imgs = images.shape[0]
        indices = sorted(random.sample(range(total_imgs), min(n, total_imgs)))

        # Extract patches
        img_patches = images[indices, y:y+ps, x:x+ps]  # (N, ps, ps)
        normal_patch = normal_gt[y:y+ps, x:x+ps, :]     # (ps, ps, 3)
        mask_patch = mask[y:y+ps, x:x+ps]                # (ps, ps)

        # Per-image normalization (zero mean, unit std within masked region)
        mask_bool = mask_patch > 0.5
        for i in range(img_patches.shape[0]):
            if mask_bool.any():
                m = img_patches[i][mask_bool].mean()
                s = img_patches[i][mask_bool].std() + 1e-8
                img_patches[i] = (img_patches[i] - m) / s

        # Augmentation: horizontal flip
        if self.augment and random.random() > 0.5:
            img_patches = img_patches[:, :, ::-1].copy()
            normal_patch = normal_patch[:, ::-1, :].copy()
            normal_patch[:, :, 0] = -normal_patch[:, :, 0]  # negate x-component
            mask_patch = mask_patch[:, ::-1].copy()

        # Convert to tensors
        # images: (N, 1, ps, ps)
        img_tensor = torch.from_numpy(img_patches).float().unsqueeze(1)
        # normal: (3, ps, ps)
        normal_tensor = torch.from_numpy(normal_patch).float().permute(2, 0, 1)
        # mask: (1, ps, ps)
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)

        return img_tensor, normal_tensor, mask_tensor


class DiLiGentTestDataset(Dataset):
    """
    Full-resolution dataset for evaluation.

    Each __getitem__ returns the full images and GT for one object.
    """

    def __init__(self, data_root: str, objects: list, num_images: int = 96):
        self.data_root = data_root
        self.objects = objects
        self.num_images = num_images
        self.object_data = []

        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            data = self._load_object(obj_dir, obj_name)
            if data is not None:
                self.object_data.append(data)

    def _load_object(self, obj_dir, obj_name):
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path):
            return None
        normal_gt = np.load(normal_path).astype(np.float32)

        mask_path = os.path.join(obj_dir, "mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        # Load images using shared helper
        images = DiLiGentDataset._load_images(obj_dir, max_images=self.num_images)
        if images is None:
            return None

        # Per-image normalization
        mask_bool = mask > 0.5
        for i in range(images.shape[0]):
            if mask_bool.any():
                m = images[i][mask_bool].mean()
                s = images[i][mask_bool].std() + 1e-8
                images[i] = (images[i] - m) / s

        return {
            "name": obj_name,
            "images": images,
            "normal_gt": normal_gt,
            "mask": mask,
        }

    def __len__(self):
        return len(self.object_data)

    def __getitem__(self, idx):
        obj = self.object_data[idx]
        img_tensor = torch.from_numpy(obj["images"]).float().unsqueeze(1)  # (N, 1, H, W)
        normal_tensor = torch.from_numpy(obj["normal_gt"]).float().permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(obj["mask"]).float().unsqueeze(0)  # (1, H, W)
        return img_tensor, normal_tensor, mask_tensor, obj["name"]


def collate_variable_n(batch):
    """
    Custom collate for variable-N batches.
    Pads image tensors to max_N in the batch with zeros.

    Returns:
        images: (B, max_N, 1, H, W)
        normals: (B, 3, H, W)
        masks: (B, 1, H, W)
        image_counts: (B,) — actual N for each sample
    """
    images_list, normals_list, masks_list = [], [], []
    counts = []

    for imgs, normal, mask in batch:
        images_list.append(imgs)
        normals_list.append(normal)
        masks_list.append(mask)
        counts.append(imgs.shape[0])

    max_n = max(counts)
    B = len(batch)
    _, C, H, W = images_list[0].shape

    # Pad images to max_N
    padded_images = torch.zeros(B, max_n, C, H, W)
    for i, imgs in enumerate(images_list):
        padded_images[i, :counts[i]] = imgs

    normals = torch.stack(normals_list, dim=0)
    masks = torch.stack(masks_list, dim=0)
    image_counts = torch.tensor(counts, dtype=torch.long)

    return padded_images, normals, masks, image_counts
