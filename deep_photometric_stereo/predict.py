"""
Predict using trained TransUNetPS models.

Supports two modes:
  - normal (default): Multi-image photometric stereo -> normal map
  - synapse: Single image segmentation -> organ mask

Usage:
    # Photometric stereo: predict normal map from folder of images
    python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/batteryPNG --output ./output/

    # Photometric stereo: predict from specific image files
    python predict.py --checkpoint checkpoints/train/best.pt --images img1.png img2.png img3.png --output ./output/

    # Synapse: segment a single CT slice
    python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt --input image.png --output ./output/

    # Synapse: segment all slices in a folder
    python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt --input_dir ./ct_slices/ --output ./output/
"""
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image

from config import Config, ModelConfig
from model import get_model
from utils import normal_to_rgb, load_checkpoint, detect_model_type, count_parameters


# Photometric Stereo prediction (multi-image -> normal map)

def load_images(input_dir=None, image_paths=None, max_images=96):
    """Load grayscale images. Returns (N, H, W) float32 [0, 1]."""
    if image_paths:
        paths = image_paths
    elif input_dir:
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(input_dir, ext)))
        paths = sorted(paths)
    else:
        raise ValueError("Provide either --input_dir or --images")

    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    paths = paths[:max_images]
    print(f"Loading {len(paths)} images...")

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
        else:
            img = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
        images.append(img)

    images = np.stack(images, axis=0)
    print(f"  Shape: {images.shape}, dtype: {images.dtype}")
    return images


def normalize_images(images, mask=None):
    """Per-image zero-mean, unit-std normalization within mask."""
    mask_bool = mask > 0.5 if mask is not None else np.ones(images.shape[1:], dtype=bool)
    for i in range(images.shape[0]):
        if mask_bool.any():
            m = images[i][mask_bool].mean()
            s = images[i][mask_bool].std() + 1e-8
            images[i] = (images[i] - m) / s
    return images


@torch.no_grad()
def predict_normal(model, images, device, tile_size=128, max_images=96):
    """Predict normal map from multiple images using tiling."""
    model.eval()
    N, H, W = images.shape

    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, pH, pW = images.shape
    pred_normal = np.zeros((3, pH, pW), dtype=np.float32)
    tile_size = max(16, (tile_size // 16) * 16)

    if N > max_images:
        indices = np.linspace(0, N - 1, max_images).astype(int)
        images_sub = images[indices]
    else:
        images_sub = images

    n_used = images_sub.shape[0]
    print(f"Predicting with {n_used} images, tile_size={tile_size}, resolution={pH}x{pW}...")

    total_tiles = ((pH + tile_size - 1) // tile_size) * ((pW + tile_size - 1) // tile_size)
    tile_idx = 0

    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            y_end = min(y + tile_size, pH)
            x_end = min(x + tile_size, pW)

            tile = images_sub[:, y:y_end, x:x_end]
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(1).unsqueeze(0).to(device)
            counts = torch.tensor([n_used], dtype=torch.long, device=device)

            pred_tile = model(tile_tensor, counts)
            pred_normal[:, y:y_end, x:x_end] = pred_tile[0].cpu().numpy()

            tile_idx += 1
            if tile_idx % 10 == 0 or tile_idx == total_tiles:
                print(f"  Tile {tile_idx}/{total_tiles}")

    pred_normal = pred_normal[:, :H, :W]
    return pred_normal.transpose(1, 2, 0)  # (H, W, 3)


def run_normal_prediction(args, device):
    """Run photometric stereo prediction."""
    config = Config(device=args.device)
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

    images = normalize_images(images, mask)
    pred_normal = predict_normal(model, images, device, tile_size=args.tile_size)

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


# Synapse segmentation prediction (single image -> organ mask)

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


def label_to_color(label):
    """Convert label map to RGB color image."""
    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in SYNAPSE_COLORS.items():
        rgb[label == cls_id] = color
    return rgb


@torch.no_grad()
def predict_segmentation(model, image, device, img_size=224):
    """
    Predict segmentation mask for a single grayscale image.

    Args:
        model: trained TransUNetPS in segmentation mode
        image: (H, W) float32 numpy array
        device: torch device
        img_size: model input size

    Returns:
        pred_label: (H, W) uint8 class labels
    """
    model.eval()
    H, W = image.shape

    # Resize to model input
    img_resized = np.array(Image.fromarray(image).resize((img_size, img_size), Image.BICUBIC))
    inp = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 224, 224)

    out = model(inp)  # (1, num_classes, 224, 224)
    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (224, 224)

    # Resize back to original
    pred_full = np.array(Image.fromarray(pred).resize((W, H), Image.NEAREST))
    return pred_full


def run_synapse_prediction(args, device):
    """Run Synapse segmentation prediction."""
    mt = detect_model_type(args.checkpoint)
    print(f"Auto-detected model_type: {mt}")

    model_cfg = ModelConfig(model_type=mt, mode="segmentation", num_classes=9, in_channels=1)
    model = get_model(model_cfg, model_type=mt).to(device)
    epoch, val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: epoch={epoch}, loss={val_loss:.4f}")
    print(f"Parameters: {count_parameters(model):,}")

    # Collect input images
    if args.input_dir:
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.npz"]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        paths = sorted(paths)
    elif args.images:
        paths = args.images
    else:
        print("ERROR: Provide --input_dir or --images")
        return

    if not paths:
        print(f"ERROR: No images found")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"Processing {len(paths)} images...")

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1].lower()

        # Load image
        if ext == ".npz":
            data = np.load(p)
            image = data["image"].astype(np.float32)
        else:
            image = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0

        print(f"\n  {name}: {image.shape}")

        # Predict
        pred_label = predict_segmentation(model, image, device)

        # Save label map as .npy
        np.save(os.path.join(args.output, f"{name}_pred.npy"), pred_label)

        # Save color visualization
        color = label_to_color(pred_label)
        Image.fromarray(color).save(os.path.join(args.output, f"{name}_pred_color.png"))

        # Save grayscale label (scaled for visibility)
        label_vis = (pred_label.astype(np.float32) / 8.0 * 255).astype(np.uint8)
        Image.fromarray(label_vis).save(os.path.join(args.output, f"{name}_pred_label.png"))

        # Print detected organs
        unique = np.unique(pred_label)
        organs = [SYNAPSE_ORGAN_NAMES.get(c, f"Class{c}") for c in unique if c > 0]
        print(f"    Detected: {', '.join(organs) if organs else 'none'}")

    print(f"\nSegmentation prediction complete. Results saved to {args.output}/")


# Main

def main():
    parser = argparse.ArgumentParser(description="Predict with TransUNetPS")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "synapse"],
                        help="normal: photometric stereo, synapse: organ segmentation")
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
                        help="Max images to use (for normal mode)")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="Tile size (for normal mode)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="transunet",
                        choices=["transunet", "lightweight"],
                        help="Model architecture (auto-detected from checkpoint)")
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
