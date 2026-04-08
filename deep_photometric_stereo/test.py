"""
Evaluation / inference script for TransUNetPS.
"""
import argparse
import os
import glob
import csv # Thêm thư viện csv
import numpy as np
import torch
from PIL import Image

from config import Config, DataConfig, ModelConfig
from dataset import DiLiGentTestDataset
from model import get_model
from utils import (
    mean_angular_error, angular_error_map, load_checkpoint, detect_model_type,
    count_parameters, normal_to_rgb,
)


@torch.no_grad()
def predict_full_resolution(model, images, device, tile_size=128, max_images=96):
    """Predict normal map using Sliding Window with Hanning Blending."""
    import numpy as _np

    model.eval()
    N, C, H, W = images.shape
    tile_size = max(16, (tile_size // 16) * 16)
    overlap = 0.5
    stride = int(tile_size * (1 - overlap))

    if N > max_images:
        indices = torch.linspace(0, N - 1, max_images).long()
        images = images[indices]
        n_used = max_images
    else:
        n_used = N

    pad_top = tile_size // 2
    pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
    pad_left = tile_size // 2
    pad_right = (tile_size - W % stride) % stride + tile_size // 2
    images = torch.nn.functional.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    _, _, pH, pW = images.shape

    window_1d = _np.hanning(tile_size).astype(_np.float32)
    window_2d = _np.outer(window_1d, window_1d)
    window_t = torch.from_numpy(window_2d)  

    pred_accum = torch.zeros(3, pH, pW)
    weight_accum = torch.zeros(1, pH, pW)
    counts = torch.tensor([n_used], dtype=torch.long, device=device)

    y_positions = list(range(0, pH - tile_size + 1, stride))
    x_positions = list(range(0, pW - tile_size + 1, stride))

    for y in y_positions:
        for x in x_positions:
            tile_imgs = images[:, :, y:y + tile_size, x:x + tile_size]
            tile_imgs = tile_imgs.unsqueeze(0).to(device)

            pred_tile = model(tile_imgs, counts)[0].cpu()  

            pred_accum[:, y:y + tile_size, x:x + tile_size] += pred_tile * window_t.unsqueeze(0)
            weight_accum[:, y:y + tile_size, x:x + tile_size] += window_t.unsqueeze(0)

    weight_accum = weight_accum.clamp(min=1e-5)
    pred_normal = pred_accum / weight_accum

    pred_normal = pred_normal[:, pad_top:pad_top + H, pad_left:pad_left + W]

    norm = pred_normal.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return pred_normal / norm


def evaluate_object(model, test_ds, obj_idx, device, save_dir=None):
    """Evaluate on one object, optionally save outputs. Returns full metrics dict."""
    images, normal_gt, mask, obj_name = test_ds[obj_idx]
    N, C, H, W = images.shape

    print(f"\nEvaluating: {obj_name} ({N} images, {H}x{W})")

    pred_normal = predict_full_resolution(model, images, device)

    pred_np = pred_normal.permute(1, 2, 0).numpy()
    gt_np = normal_gt.permute(1, 2, 0).numpy()
    mask_np = mask[0].numpy()

    mae = mean_angular_error(pred_np, gt_np, mask_np)
    err_map = angular_error_map(pred_np, gt_np, mask_np)

    valid_errors = err_map[mask_np > 0.5]
    
    # Gom toàn bộ metrics lại
    metrics = {
        "mae": mae,
        "median": float(np.median(valid_errors)),
        "max": float(np.max(valid_errors)),
        "p10": float((valid_errors < 10).mean() * 100),
        "p20": float((valid_errors < 20).mean() * 100),
        "p30": float((valid_errors < 30).mean() * 100)
    }
    
    print(f"  Mean Angular Error: {metrics['mae']:.2f}")
    print(f"  Median Error:       {metrics['median']:.2f}")
    print(f"  Max Error:          {metrics['max']:.2f}")
    print(f"  <10 pixels:         {metrics['p10']:.1f}%")
    print(f"  <20 pixels:         {metrics['p20']:.1f}%")
    print(f"  <30 pixels:         {metrics['p30']:.1f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pred_rgb = normal_to_rgb(pred_np * mask_np[..., None])
        Image.fromarray(pred_rgb).save(os.path.join(save_dir, f"{obj_name}_pred_normal.png"))
        gt_rgb = normal_to_rgb(gt_np * mask_np[..., None])
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"{obj_name}_gt_normal.png"))
        err_vis = (err_map / 90.0 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(err_vis).save(os.path.join(save_dir, f"{obj_name}_error_map.png"))
        np.save(os.path.join(save_dir, f"{obj_name}_pred_normal.npy"), pred_np)
        print(f"  Saved outputs to {save_dir}/")

    return metrics # Trả về dictionary thay vì chỉ MAE


def discover_objects(data_root):
    objects = []
    if not os.path.isdir(data_root):
        return objects
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        if os.path.exists(os.path.join(obj_dir, "Normal_gt.npy")) or os.path.exists(os.path.join(obj_dir, "Normal_gt.mat")):
            objects.append(name)
    return objects


def main():
    parser = argparse.ArgumentParser(description="Evaluate TransUNetPS")
    parser.add_argument("--mode", type=str, default="eval_all",
                        choices=["eval", "eval_all", "logo_eval", "synapse"],
                        help="eval: single object, eval_all: all objects, synapse: Synapse segmentation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory (for --mode logo_eval)")
    parser.add_argument("--test_object", type=str, default=None,
                        help="Object to evaluate on (for --mode eval)")
    parser.add_argument("--data_root", type=str, default="./data/training",
                        help="Directory containing test objects")
    parser.add_argument("--save_output", type=str, default=None,
                        help="Directory to save prediction outputs")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Path to save the CSV metrics file")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ... (Bỏ qua phần synapse cho gọn, bạn có thể copy từ code cũ của bạn nếu cần) ...

    config = Config(device=args.device)
    device = config.resolve_device()

    # eval: single object ======
    if args.mode == "eval":
        if not args.checkpoint or not args.test_object:
            print("ERROR: --checkpoint and --test_object required")
            return

        mt = detect_model_type(args.checkpoint)
        model = get_model(config.model, model_type=mt).to(device)
        load_checkpoint(args.checkpoint, model)

        test_ds = DiLiGentTestDataset(data_root=os.path.abspath(args.data_root), objects=[args.test_object])
        if len(test_ds) > 0:
            evaluate_object(model, test_ds, 0, device, save_dir=args.save_output)

    # eval_all: ALL OBJECTS & EXPORT CSV ======
    elif args.mode == "eval_all":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return

        data_root = os.path.abspath(args.data_root)
        objects = discover_objects(data_root)
        if not objects:
            print(f"ERROR: No prepared objects found in {data_root}")
            return

        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        model = get_model(config.model, model_type=mt).to(device)
        epoch, val_loss = load_checkpoint(args.checkpoint, model)
        
        all_results = {}
        for obj_name in objects:
            test_ds = DiLiGentTestDataset(data_root=data_root, objects=[obj_name])
            if len(test_ds) == 0:
                continue

            save_dir = os.path.join(args.save_output, obj_name) if args.save_output else None
            metrics = evaluate_object(model, test_ds, 0, device, save_dir=save_dir)
            all_results[obj_name] = metrics

        # XUẤT RA FILE CSV
        if not args.csv_out:
            model_name = os.path.basename(args.checkpoint).split('.')[0]
            csv_filename = f"metrics_{mt}_{model_name}.csv"
        else:
            csv_filename = args.csv_out

        print(f"\nĐang xuất dữ liệu ra file: {csv_filename}")
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Viết Header
            writer.writerow(["Object", "Mean_Angular_Error", "Median_Angular_Error", "Max_Error", "Pixels_under_10", "Pixels_under_20", "Pixels_under_30"])
            
            # Viết từng dòng data
            for obj, m in all_results.items():
                writer.writerow([obj, f"{m['mae']:.4f}", f"{m['median']:.4f}", f"{m['max']:.4f}", f"{m['p10']:.4f}", f"{m['p20']:.4f}", f"{m['p30']:.4f}"])
            
            # Tính toán và viết dòng Average cuối cùng
            avg_mae = np.mean([m['mae'] for m in all_results.values()])
            avg_median = np.mean([m['median'] for m in all_results.values()])
            avg_max = np.mean([m['max'] for m in all_results.values()])
            avg_p10 = np.mean([m['p10'] for m in all_results.values()])
            avg_p20 = np.mean([m['p20'] for m in all_results.values()])
            avg_p30 = np.mean([m['p30'] for m in all_results.values()])
            
            writer.writerow(["Average", f"{avg_mae:.4f}", f"{avg_median:.4f}", f"{avg_max:.4f}", f"{avg_p10:.4f}", f"{avg_p20:.4f}", f"{avg_p30:.4f}"])

        print(f"Đã xuất file CSV thành công!")

if __name__ == "__main__":
    main()