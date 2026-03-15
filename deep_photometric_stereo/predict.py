"""
Predict normal maps from input images using a trained TransUNetPS model.

Usage:
    # Predict from a folder of images
    python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --output ./output/
    python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/batteryPNG --output ./output/

    # Predict with specific number of images
    python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --max_images 32

    # Predict from specific image files
    python predict.py --checkpoint checkpoints/train/best.pt --images img1.png img2.png img3.png --output ./output/

    # Predict for all images in folder ( Tue, highly recommended to use that)
    python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/batteryPNG --output ./output/
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


def load_images(input_dir=None, image_paths=None, max_images=96):
    """
    Load grayscale images from a directory or list of paths.
    Returns numpy array (N, H, W) float32 [0, 1].
    """
    if image_paths:
        paths = image_paths
    elif input_dir:
        # Collect all image files
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
    if mask is not None:
        mask_bool = mask > 0.5
    else:
        mask_bool = np.ones(images.shape[1:], dtype=bool)

    for i in range(images.shape[0]):
        if mask_bool.any():
            m = images[i][mask_bool].mean()
            s = images[i][mask_bool].std() + 1e-8
            images[i] = (images[i] - m) / s
    return images


@torch.no_grad()
def predict(model, images, device, tile_size=128, max_images=96):
    """
    Predict normal map from multiple images using tiling.

    Args:
        model: trained TransUNetPS
        images: (N, H, W) numpy array
        device: torch device
        tile_size: spatial tile size for processing
        max_images: max images to use per tile

    Returns:
        pred_normal: (H, W, 3) numpy array
    """
    model.eval()
    N, H, W = images.shape

    # Pad to be divisible by 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, pH, pW = images.shape
    pred_normal = np.zeros((3, pH, pW), dtype=np.float32)

    # Ensure tile_size is divisible by 16
    tile_size = max(16, (tile_size // 16) * 16)

    # Subsample N if too many
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
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(1).unsqueeze(0)  # (1, N, 1, th, tw)
            tile_tensor = tile_tensor.to(device)
            counts = torch.tensor([n_used], dtype=torch.long, device=device)

            pred_tile = model(tile_tensor, counts)  # (1, 3, th, tw)
            pred_normal[:, y:y_end, x:x_end] = pred_tile[0].cpu().numpy()

            tile_idx += 1
            if tile_idx % 10 == 0 or tile_idx == total_tiles:
                print(f"  Tile {tile_idx}/{total_tiles}")

    # Remove padding
    pred_normal = pred_normal[:, :H, :W]
    return pred_normal.transpose(1, 2, 0)  # (H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="Predict normal maps with TransUNetPS")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing input images")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Specific image file paths")
    parser.add_argument("--mask", type=str, default=None,
                        help="Optional mask image (foreground region)")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--max_images", type=int, default=96,
                        help="Maximum number of images to use")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="Tile size for processing (must be divisible by 16)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--model_type", type=str, default="transunet",
                        choices=["transunet", "lightweight"],
                        help="Model architecture: transunet (~11M) or lightweight (~4.7M)")
    args = parser.parse_args()

    # Resolve device
    config = Config(device=args.device)
    device = config.resolve_device()
    print(f"Device: {device}")

    # Load model (auto-detect model_type from checkpoint)
    mt = detect_model_type(args.checkpoint)
    print(f"Auto-detected model_type: {mt}")
    model = get_model(config.model, model_type=mt).to(device)
    epoch, val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    print(f"Parameters: {count_parameters(model):,}")

    # Load images
    images = load_images(
        input_dir=args.input_dir,
        image_paths=args.images,
        max_images=args.max_images,
    )

    # Load mask if provided
    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = np.array(Image.open(args.mask).convert("L")).astype(np.float32)
        mask = (mask > 128).astype(np.float32)
        print(f"Mask loaded: shape={mask.shape}, foreground={int(mask.sum())}")

    # Normalize images
    images = normalize_images(images, mask)

    # Predict
    pred_normal = predict(model, images, device, tile_size=args.tile_size)

    # Apply mask if provided
    if mask is not None:
        pred_normal = pred_normal * mask[..., None]

    # Save outputs
    os.makedirs(args.output, exist_ok=True)

    # Save as .npy
    npy_path = os.path.join(args.output, "predicted_normal.npy")
    np.save(npy_path, pred_normal)
    print(f"Saved: {npy_path} (shape={pred_normal.shape})")

    # Save as RGB visualization
    rgb = normal_to_rgb(pred_normal)
    rgb_path = os.path.join(args.output, "predicted_normal.png")
    Image.fromarray(rgb).save(rgb_path)
    print(f"Saved: {rgb_path}")

    # Save visualization of x, y, z components
    for i, name in enumerate(["nx", "ny", "nz"]):
        comp = pred_normal[:, :, i]
        comp_vis = ((comp + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        comp_path = os.path.join(args.output, f"predicted_{name}.png")
        Image.fromarray(comp_vis).save(comp_path)

    print(f"\nPrediction complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
