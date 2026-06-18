"""
Predict using trained TransUNetPS models.

Supports two modes:
  - normal (default): Photometric stereo (Sequence or RGB Multiplexing) -> normal map
  - synapse: Single image segmentation -> organ mask

Usage:
    # ============================================================================
    # RECOMMENDED COMMAND (real phone photo OR DiLiGenT image)
    # Just point at the image — no flags needed. The default pipeline auto-applies:
    #   - sRGB->linear conversion (camera JPEGs)
    #   - Otsu foreground normalization
    #   - downscale-to-training-scale (then upscales the result back)
    #   - ADAPTIVE seg-head mask: trusts the model mask when it agrees with
    #     brightness (in-distribution -> recovers dim object parts like dark
    #     corners/hands), and falls back to a conservative brightness grow when
    #     the head over-segments (out-of-distribution -> no background leak).
    # The model mask now improves BOTH the charger and the bear, so do NOT pass
    # --no_pred_mask for the 25-image checkpoints (it is only needed for the
    # legacy best_trans.pt). The --no_pred_mask variants below are kept only so
    # you can compare/contrast.
    # ============================================================================

    # --- RECOMMENDED (default, uses adaptive model mask) ---
    python predict.py --checkpoint checkpoints/run/best_trans_25.pt --images data/testing/thanh_test/target/APC_0014.jpg --output ./output/charge

    python predict.py --checkpoint checkpoints/run/best_trans_25.pt --images data/testing/thanh_test/mask_problem/input_object_image.png --output ./output/bear_fixed

    python predict.py --checkpoint checkpoints/run/best_trans_25.pt --images data/testing/thanh_test/charger_test_2/input_charger_2.png --output ./output/charger_2_output

    # --- For comparison only (brightness Otsu mask, no seg-head) ---
    python predict.py --checkpoint checkpoints/run/best_trans_25.pt --images data/testing/thanh_test/target/APC_0014.jpg --output ./output/charge_no_pred_mask --no_pred_mask

    python predict.py --checkpoint checkpoints/run/best_trans_25.pt --images data/testing/thanh_test/mask_problem/input_object_image.png --output ./output/bear_fixed_no_pred_mask --no_pred_mask

    # Checkpoints to compare:
    #   checkpoints/run/best_trans.pt     (transunet,   Val MAE 7.9 — LEGACY mask head, ALWAYS use --no_pred_mask)
    #   checkpoints/run/best_trans_25.pt  (transunet,   Val MAE 9.3)
    #   checkpoints/run/best_lw_25.pt     (lightweight, Val MAE 9.9)

    # DiLiGenT test sequence (folder of grayscale PNGs):
    python predict.py --checkpoint checkpoints/run/best_trans_25.pt \
        --input_dir data/testing/bootao2PNG --output ./output/bootao2 --no_pred_mask

    # Useful flags:
    #   --linearize auto|on|off   sRGB->linear ('auto' = on for .jpg/.jpeg only)
    #   --resize_max 1024         downscale longest side before predict (0 = off)
    #   --no_pred_mask            use Otsu mask instead of the model's mask head

    # Synapse: segment a single CT slice
    python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt --images image.png --output ./output/
"""
import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from config import Config, ModelConfig
from model import get_model
from utils import normal_to_rgb, load_checkpoint, detect_model_type, count_parameters


# ==============================================================================
# PHOTOMETRIC STEREO PREDICTION
# ==============================================================================

def _srgb_to_linear(x):
    """
    Inverse sRGB EOTF — converts gamma-encoded camera photos to linear radiance.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_images(input_dir=None, image_paths=None, max_images=96, linearize="auto"):
    """
    Load images for prediction.
    Supports two modes:
      1. Sequence Mode: A folder or list of grayscale images (simulating different light angles).
      2. RGB Multiplexing Mode: A SINGLE color image where R, G, B channels represent 3 light angles.
    Returns: (N, H, W) float32 [0, 1].
    """
    if image_paths:
        paths = image_paths
    elif input_dir:
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.exr"]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(input_dir, ext)))
        paths = sorted(paths)
    else:
        raise ValueError("Provide either --input_dir or --images")

    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir or image_paths}")

    paths = paths[:max_images]
    print(f"Loading {len(paths)} image file(s)...")
    
    # --- RGB MULTIPLEXING MODE CHECK ---
    # If exactly 1 image is provided, check if it's an RGB image
    if len(paths) == 1:
        p = paths[0]
        ext = os.path.splitext(p)[1].lower()
        if ext in (".tif", ".tiff"):
            # Single RGB TIFF = a linear RAW capture (e.g. the LightControl app's
            # Bayer->linear output, whose R/G/B channels carry the three
            # illumination arcs). Use it directly for RGB Multiplexing.
            # IMPORTANT: do NOT apply sRGB->linear here. RAW TIFF is already
            # linear, matching the linear training distribution; a second
            # de-gamma would distort the tonal curve. (The model z-scores its
            # input, which is invariant to scale/offset but NOT to gamma, so the
            # linear-vs-sRGB distinction genuinely changes the prediction.)
            try:
                import tifffile
                arr = tifffile.imread(p).astype(np.float32)
            except ImportError:
                arr = np.array(Image.open(p)).astype(np.float32)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]
                if arr.max() > 255.0:
                    arr = arr / 65535.0
                elif arr.max() > 1.5:
                    arr = arr / 255.0
                print("[*] Detected single RGB TIFF (linear RAW)! Enabling RGB Multiplexing Mode "
                      "(R,G,B -> 3 light angles; no sRGB->linear, already linear)")
                images = np.stack([arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]], axis=0)
                print(f"  Extracted 3 channels from single image. Shape: {images.shape}, dtype: {images.dtype}")
                return images
            # single-channel TIFF -> fall through to standard sequence handling below
        elif ext != ".exr":
            # Standard 8-bit image format (PNG/JPEG/BMP)
            img_pil = ImageOps.exif_transpose(Image.open(p))  # honour phone camera rotation
            if img_pil.mode in ("RGB", "RGBA"):
                print("[*] Detected single RGB image! Enabling RGB Multiplexing Mode (R,G,B -> 3 light angles)")
                img_rgb = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
                # sRGB -> linear: phone JPEGs are gamma-encoded; training data is linear.
                do_linearize = (linearize == "on") or (linearize == "auto" and ext in (".jpg", ".jpeg"))
                if do_linearize:
                    img_rgb = _srgb_to_linear(img_rgb)
                    print("[*] Applied sRGB -> linear conversion (camera photo detected)")
                # Split RGB into 3 separate grayscale channels (N, H, W) -> (3, H, W)
                images = np.stack([img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]], axis=0)
                print(f"  Extracted 3 channels from single image. Shape: {images.shape}, dtype: {images.dtype}")
                return images

    # --- STANDARD SEQUENCE MODE (Multiple Grayscale Images) ---
    images = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".tif", ".tiff"):
            try:
                import tifffile
                img = tifffile.imread(p).astype(np.float32)
            except ImportError:
                img = np.array(Image.open(p)).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                else:
                    img = img[:, :, 0]
            if img.max() > 1.5:
                img = img / img.max() if img.max() > 0 else img
        elif ext == ".exr":
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            import cv2
            img = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise ValueError(f"Cannot read EXR file: {p}")
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    img = 0.1140 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.2989 * img[:, :, 2]
                else:
                    img = img[:, :, 0]
        else:
            img = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
            do_linearize = (linearize == "on") or (linearize == "auto" and ext in (".jpg", ".jpeg"))
            if do_linearize:
                img = _srgb_to_linear(img)
        images.append(img)

    images = np.stack(images, axis=0)
    print(f"  Shape: {images.shape}, dtype: {images.dtype}")
    return images

def normalize_images(images, mask=None):
    """Z-score normalize (C, H, W) images using foreground pixel statistics — matches training."""
    images = images.copy().astype(np.float32)
    if images.ndim == 3:  # (C, H, W)
        fg = (mask > 0.5) if mask is not None else (images.max(axis=0) > 0.04)
        for i in range(images.shape[0]):
            ch = images[i]
            n = int(fg.sum())
            m = float(ch[fg].mean()) if n > 50 else float(ch.mean())
            s = (float(ch[fg].std()) + 1e-8) if n > 50 else (float(ch.std()) + 1e-8)
            images[i] = (ch - m) / s
    return images


def _normalize_tile_zscore(tile, fg=None):
    tile = tile.copy()
    if fg is None:
        fg = tile.max(axis=0) > 0.04  # legacy: background is unlit/dark in LED ring PS setup
    for c in range(tile.shape[0]):
        n = int(fg.sum())
        if n > 20:
            vals = tile[c][fg]
            # p99 clip on foreground (matches SyntheticDataset training path)
            p99 = float(np.percentile(vals, 99.0))
            tile[c] = np.clip(tile[c], 0.0, p99)
            vals = tile[c][fg]
            m = float(vals.mean())
            s = float(vals.std()) + 1e-8
        else:
            m = float(tile[c].mean())
            s = float(tile[c].std()) + 1e-8
        tile[c] = (tile[c] - m) / s
    return tile


def _otsu_threshold(lum_map):
    """Otsu's optimal threshold on a (H, W) luminance map — works for any background brightness.
    Uses direct linear indexing instead of np.histogram to avoid a numpy 2.x / Python 3.14
    boundary bug where bincount returns 257 elements for a 256-bin histogram.
    """
    lo, hi = float(lum_map.min()), float(lum_map.max())
    if hi <= lo:
        return lo
    n_bins = 256
    flat = lum_map.ravel().astype(np.float64)
    # Map [lo, hi] linearly onto [0, n_bins-1] — guaranteed to produce valid indices
    idx = np.floor((flat - lo) / (hi - lo) * (n_bins - 1)).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins).astype(np.float64)
    bin_mid = lo + (np.arange(n_bins) + 0.5) / n_bins * (hi - lo)
    total = hist.sum()
    mu_total = float((hist * bin_mid).sum())
    w0, mu0_sum, best_t, best_var = 0.0, 0.0, lo, 0.0
    for h, b in zip(hist, bin_mid):
        w0 += h
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0_sum += h * b
        mu0 = mu0_sum / w0
        mu1 = (mu_total - mu0_sum) / w1
        var = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = b
    return float(best_t)

def predict_normal(model, images, device, tile_size=512, overlap=0.75, max_images=96,
                   dark_frame=None, no_pred_mask=False, single_pass="auto",
                   single_pass_max=800):
    """
    Predict normal map using Sliding Window with high overlap for zero-tiling.
    Includes Auto-Masking from the Segmentation branch.

    Pass `no_pred_mask=True` to skip Stage B and use the pure brightness mask
    (required for `best_trans.pt` whose legacy coupled seg head is unreliable).

    single_pass: "auto" (default) runs one full-image forward pass when the image
        fits (max side <= single_pass_max), which is SHARPER than tiling
    """
    model.eval()

    # Normalize the canonical shape and value scale of `images`:
    #   - single RGB image  (1,H,W,3) or (H,W,3)  -> (3,H,W) by unpacking RGB channels
    #     (treats one tri-color shot as three PS images, for back-compat)
    #   - multi RGB images  (N,H,W,3)             -> (N,H,W) by per-image luminance
    #   - already (N,H,W)                         -> untouched
    if images.ndim == 4 and images.shape[0] == 1 and images.shape[3] >= 3:
        img_rgb = images[0]
        if img_rgb.max() > 2.0: img_rgb = img_rgb.astype(np.float32) / 255.0
        images = np.stack([img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]], axis=0)
    elif (images.ndim == 3 and images.shape[2] in (3, 4)
          and images.shape[2] < images.shape[0] and images.shape[0] != 3):
        # True (H, W, C) single image. The tightened guard (C in (3,4) AND C < H)
        # prevents (N, H, W) grayscale sequences from being misclassified as a
        # single RGB image — e.g. a (96, 768, 1024) stack must stay a sequence.
        if images.max() > 2.0: images = images.astype(np.float32) / 255.0
        images = np.stack([images[:, :, 0], images[:, :, 1], images[:, :, 2]], axis=0)
    elif images.ndim == 4 and images.shape[3] >= 3:
        if images.max() > 2.0: images = images.astype(np.float32) / 255.0
        images = (0.2989 * images[..., 0] + 0.5870 * images[..., 1]
                  + 0.1140 * images[..., 2]).astype(np.float32)

    # Normalize the dark frame to (H, W) luminance in the same value scale as `images`.
    if dark_frame is not None:
        df = np.asarray(dark_frame, dtype=np.float32)
        if df.max() > 2.0: df = df / 255.0
        if df.ndim == 3 and df.shape[2] >= 3:
            df = 0.2989 * df[..., 0] + 0.5870 * df[..., 1] + 0.1140 * df[..., 2]
        dark_frame = df.astype(np.float32)

    # Keep the full lit set around for the dark-reference mask before subsetting to 3.
    lit_full = images

    if images.shape[0] > 3:
        n = images.shape[0]
        if n >= 9:
            # Select one image from each third of the sequence so the 3 chosen lights
            # are maximally spread in azimuth (matches the R/G/B group layout of the ring).
            # For a 24-LED ring: picks LED 4, 12, 20 — centre of each colour group.
            idx = [n // 6, n // 2, 5 * n // 6]
        else:
            idx = [0, n // 3, 2 * n // 3]
        images = images[idx]
    elif images.shape[0] < 3:
        raise ValueError("ERROR: At least 3 images are required!")

    _, H, W = images.shape

    tile_size = min(max(64, (tile_size // 16) * 16), 1024)
    stride = max(1, int(tile_size * (1 - overlap)))

    print(f"[*] Predicting High-Res ({H}x{W}) | Tile: {tile_size} | Overlap: {overlap*100}%...")

    # ==============================================================
    # GLOBAL FOREGROUND ESTIMATE FOR NORMALIZATION
    # Training z-scores each patch on GROUND-TRUTH MASK pixels only. At
    # inference we approximate that mask with Otsu on luminance intersected
    # with the legacy brightness rule. On DiLiGenT (black background) this
    # matches the old behavior; on real photos it excludes the gamma-lifted
    # background that was corrupting the z-score statistics.
    # ==============================================================
    lum_full = 0.2989 * images[0] + 0.5870 * images[1] + 0.1140 * images[2]
    otsu_t = _otsu_threshold(lum_full)
    fg_global = (lum_full > otsu_t) & (images.max(axis=0) > 0.04)
    fg_frac = float(fg_global.mean())
    if fg_frac < 0.005 or fg_frac > 0.90:
        # Otsu collapsed (uniform image or inverted) — fall back to legacy rule
        fg_global = images.max(axis=0) > 0.04
        print(f"[!] Otsu foreground unreliable ({fg_frac:.1%}) — using legacy >0.04 rule "
              f"({fg_global.mean():.1%} fg)")
    else:
        print(f"[*] Foreground estimate for normalization: {fg_frac:.1%} of image "
              f"(otsu_t={otsu_t:.3f})")

    # ==============================================================
    # INFERENCE: single full-image pass (sharp) vs sliding-window tiling.
    # Single pass avoids the Hanning-blend averaging that low-pass-filters
    # the result, so small objects (DiLiGenT, the bear) come out noticeably
    # sharper. Large images stay on tiling for memory + scale reasons, which
    # keeps previously validated big-photo results unchanged.
    # ==============================================================
    if single_pass == "on":
        use_single = True
    elif single_pass == "off":
        use_single = False
    else:  # auto
        use_single = max(H, W) <= single_pass_max

    if use_single:
        s = _normalize_tile_zscore(images.copy(), fg=fg_global)
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        inp = np.pad(s, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().unsqueeze(0).to(device))[0].cpu().numpy()
        out = out[:, :H, :W]
        if out.shape[0] == 3:
            out = np.concatenate([out, np.zeros((1, H, W), dtype=np.float32)], axis=0)
        pred_final = out
        print(f"[*] Single-pass inference ({H}x{W}) — no tile blending (sharpest)")
    else:
        pad_top = tile_size // 2
        pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
        pad_left = tile_size // 2
        pad_right = (tile_size - W % stride) % stride + tile_size // 2

        images_pad = np.pad(images, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        fg_pad = np.pad(fg_global, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        _, pH, pW = images_pad.shape

        # FIX: Initialize 4-channel accumulation array (instead of 3) to capture the Mask channel
        pred_accum = np.zeros((4, pH, pW), dtype=np.float32)
        weight_accum = np.zeros((1, pH, pW), dtype=np.float32)

        window_1d = np.hanning(tile_size).astype(np.float32) ** 1.5
        window_2d = np.outer(window_1d, window_1d)
        window_3d = window_2d[np.newaxis, :, :]

        y_positions = list(range(0, pH - tile_size + 1, stride))
        x_positions = list(range(0, pW - tile_size + 1, stride))

        for y in y_positions:
            for x in x_positions:
                tile = images_pad[:, y:y + tile_size, x:x + tile_size].copy()
                fg_tile = fg_pad[y:y + tile_size, x:x + tile_size]
                tile = _normalize_tile_zscore(tile, fg=fg_tile)  # match training mask-restricted normalization
                tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_tile = model(tile_tensor)[0]
                    pred_np = pred_tile.cpu().numpy()

                # FIX: Accumulate all channels (3 Normal channels + 1 Mask channel)
                channels_out = pred_np.shape[0]
                pred_accum[:channels_out, y:y + tile_size, x:x + tile_size] += pred_np * window_3d
                weight_accum[:, y:y + tile_size, x:x + tile_size] += window_3d

        weight_accum = np.clip(weight_accum, 1e-5, None)
        pred_blended = pred_accum / weight_accum
        pred_final = pred_blended[:, pad_top:pad_top + H, pad_left:pad_left + W]

    # Extract the 3 Normal Map channels
    pred_normal = pred_final[:3]
    norms = np.linalg.norm(pred_normal, axis=0, keepdims=True)
    pred_normal = pred_normal / (norms + 1e-8)

    # ==============================================================
    # FOREGROUND MASK — two-stage "seed + hysteresis grow" design
    #
    # Stage A (seed): brightness threshold (dark-ref or Otsu). PRECISE —
    #   every pixel it keeps is definitely object — but it cuts off dark
    #   object parts (e.g. a shadowed bottom corner).
    # Stage B (grow): the model's seg head is accurate NEAR the object but
    #   hallucinated far away (OOD backgrounds in real photos). So we never
    #   use it standalone; instead we grow the seed through the model's
    #   high-confidence region via binary propagation. Dark corners connected
    #   to the seed are recovered; disconnected background noise is never
    #   reached. Skipped with --no_pred_mask or 3-channel legacy models.
    # ==============================================================
    from scipy.ndimage import (binary_closing, binary_fill_holes, gaussian_filter, label,
                               binary_propagation, binary_erosion, distance_transform_edt)

    # ---- Stage A: brightness seed ----
    if dark_frame is not None:
        # Dark-reference signal — per-pixel "max LED response above ambient"
        diff = np.maximum(0.0, lit_full - dark_frame[np.newaxis, ...])
        signal = diff.max(axis=0)
        mask_source = "dark-ref"
    else:
        signal = 0.2989 * images[0] + 0.5870 * images[1] + 0.1140 * images[2]
        mask_source = "Otsu"

    bg_floor = float(np.percentile(signal, 5))
    brightness_threshold = max(bg_floor * 1.5, min(_otsu_threshold(signal), 0.50))
    seed = signal > brightness_threshold

    # Close small gaps, fill holes, keep the largest connected component
    close_px = max(5, int(min(H, W) * 0.005))
    seed = binary_closing(seed, structure=np.ones((close_px, close_px)))
    seed = binary_fill_holes(seed)
    labeled, n_comp = label(seed)
    if n_comp > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background label
        seed = (labeled == sizes.argmax())
    elif n_comp == 0:
        print("[!] Brightness mask is empty — disabling foreground mask")
        seed = np.ones_like(seed, dtype=bool)

    bright_mask = seed

    # ---- Stage B: grow the bright seed to recover dim-but-lit object parts ----
    # Strategy depends on the ratio seg_fg / brightness_fg:
    #   ratio < 1.8  -> in-distribution (dark-bg PS). The seg head agrees with
    #                   brightness and is accurate (IoU ~0.98). TRUST it:
    #                   connectivity-grow the seed through all seg>0.5 pixels,
    #                   recovering dim object parts (shadowed hands/corners).
    #   ratio >= 1.8 -> OOD real photo. The seg head over-segments a CONFIDENT
    #                   halo of dark background fabric (seg>0.95 on brightness
    #                   ~0.03), so it cannot be used to grow without leaking.
    #                   Instead use BRIGHTNESS-HYSTERESIS: grow the seed into
    #                   pixels CONNECTED to it whose brightness exceeds a lower
    #                   floor (thr * 0.4). Dim-but-lit object faces (pink charger
    #                   sides) are recovered; the much darker fabric/cast-shadow
    #                   (~0.03, below the floor) stays out. No seg head, no
    #                   distance band — connectivity + a brightness floor are a
    #                   cleaner, more complete discriminator on OOD inputs.
    has_mask_channel = pred_final.shape[0] >= 4
    if has_mask_channel and not no_pred_mask and n_comp > 0:
        mask_logit = pred_final[3]
        mask_prob = 1.0 / (1.0 + np.exp(-mask_logit))
        seg_frac = float((mask_prob > 0.5).mean())
        bright_frac = max(float(seed.mean()), 1e-6)
        ratio = seg_frac / bright_frac

        if ratio < 1.8:
            # In-distribution: trust the seg head (connectivity grow, seg>0.5).
            seg_bin = mask_prob > 0.5
            grown = binary_propagation(seed, mask=(seed | seg_bin))
            grown = binary_closing(grown, structure=np.ones((close_px, close_px)))
            grown = binary_fill_holes(grown)
            glab, gn = label(grown)
            if gn > 1:
                gsizes = np.bincount(glab.ravel()); gsizes[0] = 0
                grown = (glab == gsizes.argmax())
            bright_mask = grown
            mask_source += f"+seg-trust(ratio={ratio:.2f})"
            if abs(float(grown.mean()) - float(seed.mean())) > 0.001:
                print(f"[*] Seg-trust grow: foreground {seed.mean():.1%} -> {grown.mean():.1%} "
                      f"(ratio={ratio:.2f})")
        else:
            # OOD: brightness-hysteresis grow (seg head unreliable here).
            # Grow the seed into connected pixels brighter than thr*0.4 — the
            # dim-but-lit object faces — while the darker fabric/shadow stays out.
            bright_floor = brightness_threshold * 0.4
            medium = signal > bright_floor
            grown = binary_propagation(seed, mask=medium)
            grown = binary_closing(grown, structure=np.ones((close_px, close_px)))
            grown = binary_fill_holes(grown)
            glab, gn = label(grown)
            if gn > 1:
                gsizes = np.bincount(glab.ravel()); gsizes[0] = 0
                grown = (glab == gsizes.argmax())
            recovered = float(grown.mean()) - float(seed.mean())
            if recovered > 0.001:
                print(f"[*] Brightness-hysteresis grow recovered {recovered:.1%} foreground "
                      f"(dim-but-lit object faces, floor={bright_floor:.3f}, ratio={ratio:.2f})")
            bright_mask = grown
            mask_source += f"+bright-hyst(ratio={ratio:.2f})"

    # ---- Soft Gaussian edge for natural blending ----
    blur_sigma = max(1.5, min(H, W) * 0.0015)
    mask_soft = gaussian_filter(bright_mask.astype(np.float32), sigma=blur_sigma)

    # Apply mask to the predicted normal map
    pred_normal = pred_normal * mask_soft[np.newaxis, :, :]
    print(f"[*] Applied {mask_source} mask (threshold={brightness_threshold:.3f}, "
          f"fg={bright_mask.mean():.1%})")
    # ==============================================================

    return pred_normal.transpose(1, 2, 0)


def predict_at_native_scale(model, images, device, dark_frame=None, tile_size=512,
                            overlap=0.75, max_images=96, no_pred_mask=False,
                            resize_max=1024, single_pass="auto"):
    """
    Normal-map prediction with CLI-equivalent scale matching, used by serve.py.

    If the longest side exceeds ``resize_max``, the image stack (and the optional
    dark frame) is downscaled to the training scale, the prediction is run, and
    the normal map is upscaled back to the native resolution and re-normalized to
    unit vectors. Mirrors run_normal_prediction so the REST server and the CLI
    produce matching results.

    images : (N, H, W), (N, H, W, 3), (1, H, W, 3) or (H, W, 3) float array.
    Returns: (H, W, 3) float32 normal map.
    """
    arr = np.asarray(images, dtype=np.float32)

    # Locate the spatial (H, W) axes for both stacked-grayscale and RGB layouts.
    if arr.ndim == 3 and arr.shape[2] in (3, 4) and arr.shape[2] < arr.shape[0]:
        orig_h, orig_w = arr.shape[0], arr.shape[1]     # (H, W, C)
    else:                                               # (N, H, W) or (N, H, W, C)
        orig_h, orig_w = arr.shape[1], arr.shape[2]

    def _resize_spatial(a, nh, nw):
        t = torch.from_numpy(np.asarray(a, dtype=np.float32))
        if a.ndim == 4:                                              # (N, H, W, C)
            t = F.interpolate(t.permute(0, 3, 1, 2), size=(nh, nw), mode="area")
            return t.permute(0, 2, 3, 1).numpy()
        if a.ndim == 3 and a.shape[2] in (3, 4) and a.shape[2] < a.shape[0]:  # (H, W, C)
            t = F.interpolate(t.permute(2, 0, 1).unsqueeze(0), size=(nh, nw), mode="area")
            return t[0].permute(1, 2, 0).numpy()
        if a.ndim == 3:                                             # (N, H, W)
            return F.interpolate(t.unsqueeze(0), size=(nh, nw), mode="area")[0].numpy()
        return F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(nh, nw), mode="area")[0, 0].numpy()  # (H, W)

    scaled = False
    if resize_max and max(orig_h, orig_w) > resize_max:
        scale = resize_max / max(orig_h, orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
        arr = _resize_spatial(arr, new_h, new_w)
        if dark_frame is not None:
            dark_frame = _resize_spatial(np.asarray(dark_frame, dtype=np.float32), new_h, new_w)
        scaled = True
        print(f"[*] Downscaled {orig_h}x{orig_w} -> {new_h}x{new_w} to match training scale "
              f"(disable with resize_max=0)")

    pred_normal = predict_normal(
        model, arr, device, tile_size=tile_size, overlap=overlap, max_images=max_images,
        dark_frame=dark_frame, no_pred_mask=no_pred_mask, single_pass=single_pass,
    )

    if scaled:
        t_pn = torch.from_numpy(pred_normal.transpose(2, 0, 1)).unsqueeze(0)
        t_pn = F.interpolate(t_pn, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_normal = t_pn[0].numpy().transpose(1, 2, 0)
        norms = np.linalg.norm(pred_normal, axis=2, keepdims=True)
        pred_normal = np.where(norms > 0.1, pred_normal / (norms + 1e-8), 0.0).astype(np.float32)
        print(f"[*] Upscaled normal map back to {orig_h}x{orig_w}")

    return pred_normal


def run_normal_prediction(args, device):
    """Run photometric stereo prediction."""
    config = Config(device=args.device)
    
    # Auto-detect or override model type
    if args.model_type != "auto":
        mt = args.model_type
        print(f"Forced model_type: {mt}")
    else:
        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        
    model = get_model(config.model, model_type=mt).to(device)
    epoch, val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    print(f"Parameters: {count_parameters(model):,}")

    images = load_images(input_dir=args.input_dir, image_paths=args.images,
                         max_images=args.max_images, linearize=args.linearize)

    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = np.array(Image.open(args.mask).convert("L")).astype(np.float32)
        mask = (mask > 128).astype(np.float32)
        print(f"Mask loaded: shape={mask.shape}, foreground={int(mask.sum())}")

    os.makedirs(args.output, exist_ok=True)

    input_tensor = images[:3].transpose(1, 2, 0)

    t_min = np.percentile(input_tensor, 1)
    t_max = np.percentile(input_tensor, 99)

    brightness_factor = 2.0

    png_tensor = ((input_tensor - t_min) / (t_max - t_min + 1e-8) * 255.0 * brightness_factor)
    png_tensor = png_tensor.clip(0, 255).astype(np.uint8)

    png_input_path = os.path.join(args.output, "input.png")
    Image.fromarray(png_tensor, mode='RGB').save(png_input_path)

    # ==============================================================
    # SCALE MATCHING: downscale large photos to training scale.
    # Training: 256px patches on ~512x612 images (object spans ~300-500px).
    # A 2048px phone photo has the object at ~2x that scale — the model
    # never saw geometry that "zoomed in". Downscale before prediction,
    # then upscale the normal map back to the original resolution.
    # ==============================================================
    orig_h, orig_w = images.shape[1], images.shape[2]
    scaled = False
    if args.resize_max and max(orig_h, orig_w) > args.resize_max:
        scale = args.resize_max / max(orig_h, orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
        t_imgs = torch.from_numpy(images).unsqueeze(0)  # (1, C, H, W)
        t_imgs = F.interpolate(t_imgs, size=(new_h, new_w), mode="area")
        images = t_imgs[0].numpy()
        scaled = True
        print(f"[*] Downscaled {orig_h}x{orig_w} -> {new_h}x{new_w} to match training scale "
              f"(disable with --resize_max 0)")

    pred_normal = predict_normal(model, images, device, tile_size=args.tile_size,
                                 overlap=args.overlap, no_pred_mask=args.no_pred_mask,
                                 single_pass=args.single_pass)

    # Upscale the normal map back to the original photo resolution
    if scaled:
        t_pn = torch.from_numpy(pred_normal.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, h, w)
        t_pn = F.interpolate(t_pn, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_normal = t_pn[0].numpy().transpose(1, 2, 0)
        # Re-normalize to unit vectors (interpolation shortens them); keep masked
        # background at zero instead of amplifying noise.
        norms = np.linalg.norm(pred_normal, axis=2, keepdims=True)
        pred_normal = np.where(norms > 0.1, pred_normal / (norms + 1e-8), 0.0).astype(np.float32)
        print(f"[*] Upscaled normal map back to {orig_h}x{orig_w}")

    if mask is not None:
        pred_normal = pred_normal * mask[..., None]

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    npy_path = os.path.join(args.output, "predicted_normal.npy")
    np.save(npy_path, pred_normal)
    print(f"Saved: {npy_path} (shape={pred_normal.shape})")

    rgb = normal_to_rgb(pred_normal)
    rgb_path = os.path.join(args.output, "predicted_normal.png")
    Image.fromarray(rgb).save(rgb_path)
    print(f"Saved: {rgb_path}")

    for i, name in enumerate(["nx", "ny", "nz"]):
        comp = pred_normal[:, :, i]
        comp_vis = ((comp + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(comp_vis).save(os.path.join(args.output, f"predicted_{name}.png"))

    print(f"\nNormal prediction complete. Results saved to {args.output}/")


# ==============================================================================
# SYNAPSE SEGMENTATION PREDICTION
# ==============================================================================

# Color map for Synapse organs (class -> RGB)
SYNAPSE_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # Aorta
    2: (0, 255, 0),      # Gallbladder
    3: (0, 0, 255),      # Kidney(L)
    4: (255, 255, 0),    # Kidney(R)
    5: (255, 128, 0),    # Liver
    6: (128, 0, 255),    # Pancreas
    7: (0, 255, 255),    # Spleen
    8: (255, 0, 255),    # Stomach
}

SYNAPSE_ORGAN_NAMES = {
    0: "Background", 1: "Aorta", 2: "Gallbladder", 3: "Kidney(L)",
    4: "Kidney(R)", 5: "Liver", 6: "Pancreas", 7: "Spleen", 8: "Stomach",
}

def decode_segmentation(mask, num_classes=9):
    """Convert 2D class mask to RGB image based on defined colors."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        idx = mask == c
        rgb[idx] = SYNAPSE_COLORS.get(c, (255, 255, 255))
    return rgb

def run_synapse_prediction(args, device):
    """Predict organ segmentation mask for Synapse CT slices."""
    if not args.images and not args.input_dir:
        raise ValueError("Must provide --images or --input_dir for synapse mode.")

    config = Config(device=args.device)
    
    if args.model_type != "auto":
        mt = args.model_type
    else:
        mt = detect_model_type(args.checkpoint)
        
    print(f"Auto-detected model_type: {mt}")
    model_cfg = ModelConfig(model_type=mt, mode="segmentation", num_classes=9, in_channels=1)
    model = get_model(model_cfg, model_type=mt).to(device)

    load_checkpoint(args.checkpoint, model)
    model.eval()

    paths = []
    if args.images:
        paths = args.images
    elif args.input_dir:
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not paths:
        print("No images found.")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"Predicting {len(paths)} images in synapse mode...")

    with torch.no_grad():
        for p in paths:
            img_pil = Image.open(p).convert("L")
            orig_w, orig_h = img_pil.size
            img_resized = img_pil.resize((224, 224), Image.BICUBIC)
            
            img_np = np.array(img_resized).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

            pred_logits = model(img_tensor)
            pred_class = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

            pred_mask_full = Image.fromarray(pred_class.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
            pred_mask_full_np = np.array(pred_mask_full)

            rgb_mask = decode_segmentation(pred_mask_full_np, num_classes=9)
            
            base_name = os.path.basename(p)
            name, ext = os.path.splitext(base_name)
            out_path = os.path.join(args.output, f"{name}_pred.png")
            Image.fromarray(rgb_mask).save(out_path)
            print(f"Saved {out_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Predict using trained UNetPS models")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "synapse"],
                        help="Prediction mode: 'normal' for PS, 'synapse' for segmentation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing input images")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Specific image file paths")
    parser.add_argument("--mask", type=str, default=None,
                        help="Optional mask image (for normal mode)")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--max_images", type=int, default=96,
                        help="Max images to use (for normal sequence mode)")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for Sliding Window (max 512)")
    parser.add_argument("--overlap", type=float, default=0.75, 
                        help="Overlap ratio (default: 0.75 for smooth blend)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "transunet", "lightweight", "swin", "globalattn"],
                        help="Model architecture (auto-detected from checkpoint by default)")
    parser.add_argument("--no_pred_mask", action="store_true",
                        help="Ignore the model's predicted mask head and use the "
                             "dark-ref / Otsu heuristic instead. Use this for legacy "
                             "checkpoints (e.g. best_trans.pt) with an unreliable seg head.")
    parser.add_argument("--linearize", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="sRGB -> linear conversion for camera photos. 'auto' converts "
                             ".jpg/.jpeg (phone photos are gamma-encoded; training data is "
                             "linear). 'on' forces it for all formats, 'off' disables.")
    parser.add_argument("--resize_max", type=int, default=1024,
                        help="Downscale inputs whose longest side exceeds this value, predict, "
                             "then upscale the normal map back. Matches phone photos to the "
                             "training scale. Set 0 to disable. (DiLiGenT 512x612 images are "
                             "unaffected by the 1024 default.)")
    parser.add_argument("--single_pass", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Run a single full-image forward pass (sharper, no tile-blend "
                             "averaging) instead of sliding-window tiling. 'auto' (default) "
                             "uses single-pass when the longest side <= 800 (covers DiLiGenT "
                             "objects and the bear); larger images (e.g. downscaled phone "
                             "photos at 1024) keep tiling so validated results are unchanged.")
    args = parser.parse_args()

    config = Config(device=args.device)
    device = config.resolve_device()
    print(f"Device: {device}")

    if args.mode == "synapse":
        run_synapse_prediction(args, device)
    else:
        run_normal_prediction(args, device)

if __name__ == "__main__":
    main()