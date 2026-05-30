"""
Predict using trained TransUNetPS models.

Supports two modes:
  - normal (default): Photometric stereo (Sequence or RGB Multiplexing) -> normal map
  - synapse: Single image segmentation -> organ mask

Usage:
    # Photometric stereo: predict normal map from a single RGB multiplexed image
    python predict.py --checkpoint checkpoints/best_trans.pt --images data/testing/batteryPNG/001.png --output ./output/

    # Photometric stereo: predict normal map from folder of grayscale images
    python predict.py --checkpoint checkpoints/best.pt --input_dir ./data/testing/batteryPNG --output ./output/

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

def load_images(input_dir=None, image_paths=None, max_images=96):
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
        if ext not in (".exr", ".tif", ".tiff"):
            # Standard image format
            img_pil = ImageOps.exif_transpose(Image.open(p))  # honour phone camera rotation
            if img_pil.mode in ("RGB", "RGBA"):
                print("[*] Detected single RGB image! Enabling RGB Multiplexing Mode (R,G,B -> 3 light angles)")
                img_rgb = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
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


def _normalize_tile_zscore(tile):
    """Z-score each channel using non-dark pixels — exactly matches DiLiGentDataset per-patch normalization."""
    tile = tile.copy()
    fg = tile.max(axis=0) > 0.04  # background is unlit/dark in LED ring PS setup
    for c in range(tile.shape[0]):
        n = int(fg.sum())
        m = float(tile[c][fg].mean()) if n > 20 else float(tile[c].mean())
        s = (float(tile[c][fg].std()) + 1e-8) if n > 20 else (float(tile[c].std()) + 1e-8)
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

def predict_normal(model, images, device, tile_size=512, overlap=0.75, max_images=96, dark_frame=None):
    """
    Predict normal map using Sliding Window with high overlap for zero-tiling.
    Includes Auto-Masking from the Segmentation branch.

    If `dark_frame` is provided (an HxW or HxWx3 image captured with all LEDs off),
    the segmentation mask is built from the per-pixel "max LED response above ambient"
    signal across the full input set, which is far more robust for low-contrast or
    transparent objects than the existing brightness-based Otsu fallback.
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
    elif images.ndim == 3 and images.shape[2] >= 3 and images.shape[0] != 3:
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

    pad_top = tile_size // 2
    pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
    pad_left = tile_size // 2
    pad_right = (tile_size - W % stride) % stride + tile_size // 2

    images_pad = np.pad(images, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
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
            tile = _normalize_tile_zscore(tile)  # match DiLiGentDataset per-patch normalization
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
    # FOREGROUND MASK
    # Two paths:
    #   1. Dark-reference (preferred): foreground is "any pixel that responded
    #      to at least one LED above ambient". Robust to low-contrast objects.
    #   2. Brightness Otsu (fallback): foreground is "bright in the lit shots".
    # Both share the same morphology (closing, fill, center-distance, blur).
    # ==============================================================
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label, center_of_mass

    if dark_frame is not None:
        # Per-pixel max LED-driven response across the full sequence.
        # Clipping at 0 throws away pixels where ambient actually decreased
        # (e.g. small AE drift between dark and lit shots).
        diff = np.maximum(0.0, lit_full - dark_frame[np.newaxis, ...])  # (N, H, W)
        signal = diff.max(axis=0)                                       # (H, W)
        mask_source = "dark-ref"
    else:
        # Luminance weights blue at only 11%, so dark-blue fabric backgrounds stay dim
        # while the LED-lit object (with strong R and G components) remains bright.
        signal = 0.2989 * images[0] + 0.5870 * images[1] + 0.1140 * images[2]
        mask_source = "Otsu"

    # Otsu picks the bimodal split between dim background and lit foreground.
    # Floor at max(5th-percentile * 1.5) prevents over-segmentation in uniform images.
    bg_floor = float(np.percentile(signal, 5))
    brightness_threshold = max(bg_floor * 1.5, min(_otsu_threshold(signal), 0.50))

    bright_mask = signal > brightness_threshold

    # Scale the closing kernel to image size so gaps get bridged regardless of resolution
    close_px = max(15, int(min(H, W) * 0.007))   # ~0.7% of shorter side
    bright_mask = binary_closing(bright_mask, structure=np.ones((close_px, close_px)))
    bright_mask = binary_fill_holes(bright_mask)

    # Remove any remaining stray-light blobs far from the image center
    labeled, n_comp = label(bright_mask)
    if n_comp > 1:
        img_cy, img_cx = H / 2.0, W / 2.0
        max_dist = np.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)
        min_size = int(bright_mask.size * 0.001)
        keep = np.zeros_like(bright_mask, dtype=bool)
        for i in range(1, n_comp + 1):
            comp = labeled == i
            if comp.sum() < min_size:
                continue
            cy_c, cx_c = center_of_mass(comp)
            if np.sqrt((cy_c - img_cy) ** 2 + (cx_c - img_cx) ** 2) / max_dist < 0.45:
                keep |= comp
        bright_mask = keep

    # Scale Gaussian blur radius to image resolution
    blur_sigma = max(3.0, min(H, W) * 0.002)
    mask_soft = gaussian_filter(bright_mask.astype(np.float32), sigma=blur_sigma)
    pred_normal = pred_normal * mask_soft[np.newaxis, :, :]
    print(f"[*] Applied {mask_source} mask (bg_floor={bg_floor:.3f}, threshold={brightness_threshold:.3f}, fg={bright_mask.mean():.1%})")
    # ==============================================================

    return pred_normal.transpose(1, 2, 0)

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

    images = load_images(input_dir=args.input_dir, image_paths=args.images, max_images=args.max_images)

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

    pred_normal = predict_normal(model, images, device, tile_size=args.tile_size, overlap=args.overlap)

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
                        choices=["auto", "transunet", "lightweight", "swin"],
                        help="Model architecture (auto-detected from checkpoint by default)")
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