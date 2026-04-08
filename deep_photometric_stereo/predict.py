"""
Predict using trained TransUNetPS models (Upgraded for RGB support).
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


# --- CẬP NHẬT: Load ảnh RGB thay vì Grayscale ---
def load_images(input_dir=None, image_paths=None, max_images=96):
    """Load RGB images. Returns (N, 3, H, W) float32 [0, 1]."""
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
        raise FileNotFoundError(f"No images found")

    paths = paths[:max_images]
    print(f"Loading {len(paths)} images in RGB mode...")

    img_list = []
    for p in paths:
        # Load as RGB
        img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        img_list.append(img)

    # images shape: (N, H, W, 3)
    images = np.stack(img_list, axis=0)
    # Permute to (N, 3, H, W) để khớp với đầu vào model
    images = images.transpose(0, 3, 1, 2)
    
    print(f"  Final Tensor Shape: {images.shape}, dtype: {images.dtype}")
    return images


def normalize_images(images, mask=None):
    """Per-image, per-channel zero-mean, unit-std normalization within mask."""
    # images: (N, 3, H, W)
    N, C, H, W = images.shape
    mask_bool = mask > 0.5 if mask is not None else np.ones((H, W), dtype=bool)
    
    for i in range(N):
        for c in range(3): # Duyệt qua 3 kênh RGB
            if mask_bool.any():
                m = images[i, c][mask_bool].mean()
                s = images[i, c][mask_bool].std() + 1e-8
                images[i, c] = (images[i, c] - m) / s
    return images


@torch.no_grad()
def predict_normal(model, images, device, tile_size=128, overlap=0.5, max_images=96):
    """
    Predict normal map using Sliding Window with Hanning Blending.
    """
    model.eval()
    N, C, H, W = images.shape

    # Giới hạn số lượng ảnh nếu quá nhiều
    if N > max_images:
        indices = np.linspace(0, N - 1, max_images).astype(int)
        images = images[indices]
    n_used = images.shape[0]

    tile_size = max(16, (tile_size // 16) * 16)
    stride = int(tile_size * (1 - overlap))

    print(f"Predicting with {n_used} images, tile_size={tile_size}, stride={stride}...")

    # Padding cho H và W
    pad_top = tile_size // 2
    pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
    pad_left = tile_size // 2
    pad_right = (tile_size - W % stride) % stride + tile_size // 2

    # Pad trên chiều H (axis 2) và W (axis 3)
    images_pad = np.pad(images, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
    _, _, pH, pW = images_pad.shape

    pred_normal_accum = np.zeros((3, pH, pW), dtype=np.float32)
    weight_accum = np.zeros((1, pH, pW), dtype=np.float32)

    window_1d = np.hanning(tile_size).astype(np.float32)
    window_2d = np.outer(window_1d, window_1d)
    window_3d = window_2d[np.newaxis, :, :]

    counts = torch.tensor([n_used], dtype=torch.long, device=device)

    y_positions = list(range(0, pH - tile_size + 1, stride))
    x_positions = list(range(0, pW - tile_size + 1, stride))
    total_tiles = len(y_positions) * len(x_positions)
    tile_idx = 0

    for y in y_positions:
        for x in x_positions:
            # tile shape: (N, 3, tile_size, tile_size)
            tile = images_pad[:, :, y:y + tile_size, x:x + tile_size]
            # Đưa lên device: (1, N, 3, tile_size, tile_size)
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)

            pred_tile = model(tile_tensor, counts)
            pred_np = pred_tile[0].cpu().numpy()  # (3, tile_size, tile_size)

            pred_normal_accum[:, y:y + tile_size, x:x + tile_size] += pred_np * window_3d
            weight_accum[:, y:y + tile_size, x:x + tile_size] += window_3d

            tile_idx += 1
            if tile_idx % 20 == 0 or tile_idx == total_tiles:
                print(f"   Tile {tile_idx}/{total_tiles}")

    weight_accum = np.clip(weight_accum, 1e-5, None)
    pred_normal_blended = pred_normal_accum / weight_accum
    pred_normal_final = pred_normal_blended[:, pad_top:pad_top + H, pad_left:pad_left + W]

    # Chuẩn hóa vector đơn vị
    norms = np.linalg.norm(pred_normal_final, axis=0, keepdims=True)
    pred_normal_final = pred_normal_final / (norms + 1e-8)

    return pred_normal_final.transpose(1, 2, 0)


def run_normal_prediction(args, device):
    """Run photometric stereo prediction."""
    mt = detect_model_type(args.checkpoint)
    print(f"Auto-detected model_type: {mt}")
    
    # --- FIX LỖI SIZE MISMATCH: Ép in_channels=3 ---
    model_cfg = ModelConfig(model_type=mt, in_channels=3)
    model = get_model(model_cfg, model_type=mt).to(device)
    
    epoch, val_loss = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")

    images = load_images(input_dir=args.input_dir, image_paths=args.images, max_images=args.max_images)

    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = np.array(Image.open(args.mask).convert("L")).astype(np.float32)
        mask = (mask > 128).astype(np.float32)

    images = normalize_images(images, mask)
    pred_normal = predict_normal(model, images, device, tile_size=args.tile_size)

    if mask is not None:
        pred_normal = pred_normal * mask[..., None]

    os.makedirs(args.output, exist_ok=True)
    npy_path = os.path.join(args.output, "predicted_normal.npy")
    np.save(npy_path, pred_normal)
    
    rgb = normal_to_rgb(pred_normal)
    rgb_path = os.path.join(args.output, "predicted_normal.png")
    Image.fromarray(rgb).save(rgb_path)
    print(f"Prediction complete. Saved to {args.output}/")


# --- PHẦN SYNAPSE GIỮ NGUYÊN (HOẶC CẬP NHẬT TƯƠNG TỰ NẾU CẦN) ---

def main():
    parser = argparse.ArgumentParser(description="Predict with TransUNetPS")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "synapse"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--max_images", type=int, default=12) # Gợi ý: dùng N nhỏ để khớp lúc train
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="transunet")
    args = parser.parse_args()

    config = Config(device=args.device)
    device = config.resolve_device()
    
    if args.mode == "synapse":
        # Synapse vẫn dùng 1 kênh (Grayscale CT)
        model_cfg = ModelConfig(model_type=args.model_type, mode="segmentation", in_channels=1)
        model = get_model(model_cfg, model_type=args.model_type).to(device)
        load_checkpoint(args.checkpoint, model)
        # run_synapse_prediction...
    else:
        run_normal_prediction(args, device)

if __name__ == "__main__":
    main()