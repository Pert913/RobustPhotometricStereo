"""
Evaluation / inference script for TransUNetPS, LightweightUNetPS, and SwinUNetPS.
Multi-Task capability: Normal Estimation + Segmentation.

Usage Examples:
    # Evaluate a single checkpoint (Auto-detect model type from filename)
    python test.py --mode eval_all --checkpoint checkpoints/best.pt --data_root ./data/testing

    # FORCE a specific model type (if filename doesn't contain 'swin' or 'lightweight')
    python test.py --mode eval_all --checkpoint checkpoints/best_joint.pt --model_type swin --data_root ./data/testing
"""

import argparse
import os
import glob
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import Config, DataConfig, ModelConfig
from dataset import DiLiGentTestDataset, DiLiGentMVTestDataset, SyntheticTestDataset
from model import get_model
from utils import (
    mean_angular_error, angular_error_map, load_checkpoint, detect_model_type,
    count_parameters, normal_to_rgb,
)

# ==============================================================================
# SEGMENTATION METRICS
# ==============================================================================
def calculate_iou(pred_prob, true_mask, threshold=0.5):
    """Calculates the Intersection over Union (IoU) between prediction and ground truth."""
    pred_bin = (pred_prob > threshold).astype(np.float32)
    true_bin = (true_mask > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_bin * true_bin)
    union = np.sum(pred_bin) + np.sum(true_bin) - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

# ==============================================================================
# FULL RESOLUTION INFERENCE
# ==============================================================================
@torch.no_grad()
def predict_full_resolution(model, images, device, tile_size=128, max_long_side=None):
    """
    Predict normal map via sliding-window tiling with Hanning blending.

    max_long_side: if set and the image is larger on any side, resize it down
    to that length before tiling, then upsample the result back.  This keeps
    the number of tiles small and eliminates the visible tile-grid artifact
    that appears on high-resolution images.
    Normal maps are smooth surfaces, so bilinear upsampling back to native
    resolution is perceptually clean.
    """
    import numpy as _np

    model.eval()

    if images.dim() == 3:
        images = images.unsqueeze(0)

    B, C, H, W = images.shape
    orig_H, orig_W = H, W

    # --- optional downscale -------------------------------------------------
    if max_long_side is not None and max(H, W) > max_long_side:
        scale = max_long_side / max(H, W)
        # align to multiples of 16 (safe for most encoder/decoder strides)
        new_H = max(tile_size, int(H * scale) // 16 * 16)
        new_W = max(tile_size, int(W * scale) // 16 * 16)
        images = F.interpolate(images.float(), size=(new_H, new_W),
                               mode='bilinear', align_corners=False)
        H, W = new_H, new_W
        print(f"    [resize] {orig_H}x{orig_W} → {H}x{W} for tiling")
    # ------------------------------------------------------------------------

    tile_size = max(16, (tile_size // 16) * 16)
    overlap = 0.5
    stride = int(tile_size * (1 - overlap))

    pad_top = tile_size // 2
    pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
    pad_left = tile_size // 2
    pad_right = (tile_size - W % stride) % stride + tile_size // 2
    images = torch.nn.functional.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    _, _, pH, pW = images.shape

    window_1d = _np.hanning(tile_size).astype(_np.float32)
    window_2d = _np.outer(window_1d, window_1d)
    window_t = torch.from_numpy(window_2d)

    pred_accum = torch.zeros(4, pH, pW)
    weight_accum = torch.zeros(1, pH, pW)

    y_positions = list(range(0, pH - tile_size + 1, stride))
    x_positions = list(range(0, pW - tile_size + 1, stride))

    for y in y_positions:
        for x in x_positions:
            tile_imgs = images[:, :, y:y + tile_size, x:x + tile_size].to(device)
            pred_tile = model(tile_imgs)[0].cpu()

            if pred_tile.shape[0] == 3:
                pred_tile = torch.cat([pred_tile, torch.zeros(1, tile_size, tile_size)], dim=0)

            pred_accum[:, y:y + tile_size, x:x + tile_size] += pred_tile * window_t.unsqueeze(0)
            weight_accum[:, y:y + tile_size, x:x + tile_size] += window_t.unsqueeze(0)

    weight_accum = weight_accum.clamp(min=1e-5)
    pred_out = pred_accum / weight_accum
    pred_out = pred_out[:, pad_top:pad_top + H, pad_left:pad_left + W]

    # --- upsample back to original resolution --------------------------------
    if (H, W) != (orig_H, orig_W):
        pred_out = F.interpolate(pred_out.unsqueeze(0), size=(orig_H, orig_W),
                                 mode='bilinear', align_corners=False).squeeze(0)
        pred_out[:3] = F.normalize(pred_out[:3], p=2, dim=0)
    # -------------------------------------------------------------------------

    return pred_out

# ==============================================================================
# EVALUATION PIPELINE
# ==============================================================================
@torch.no_grad()
def evaluate_object(model, test_ds, obj_idx, device, save_dir=None, use_pred_mask=True):
    """Evaluates performance on a single object and saves visual results if requested."""
    model.eval()
    images, normal_gt, mask, obj_name = test_ds[obj_idx]
    C, H, W = images.shape

    print(f"\nEvaluating: {obj_name:>15s} ({C} channels, {H}x{W})")

    images = images.unsqueeze(0).to(device)
    pred_out = model(images)[0]

    # 1. PROCESS SEGMENTATION MASK
    iou = 0.0
    mask_np = mask[0].numpy() if mask.dim() > 2 else mask.numpy()

    if use_pred_mask and pred_out.shape[0] >= 4:
        pred_mask_logit = pred_out[3, :, :]
        pred_mask_prob = torch.sigmoid(pred_mask_logit).cpu().numpy()
        pred_mask_bin = (pred_mask_prob > 0.5).astype(np.float32)
        iou = calculate_iou(pred_mask_prob, mask_np)
    else:
        pred_mask_bin = mask_np  # fall back to GT mask

    # 2. PROCESS NORMAL MAP & APPLY MASK
    pred_normal = pred_out[:3, :, :]
    pred_normal = F.normalize(pred_normal, p=2, dim=0)
    pred_normal_masked = pred_normal * torch.from_numpy(pred_mask_bin).to(device)

    # 3. COMPUTE QUANTITATIVE METRICS
    pred_np = pred_normal_masked.cpu().permute(1, 2, 0).numpy()
    gt_np = normal_gt.permute(1, 2, 0).numpy()

    mae = mean_angular_error(pred_np, gt_np, mask_np)
    err_map = angular_error_map(pred_np, gt_np, mask_np)

    valid_errors = err_map[mask_np > 0.5]
    print(f"  Segmentation IoU:   {iou * 100:.2f}%")
    print(f"  Mean Angular Error: {mae:.2f}°")
    print(f"  <10 pixels:         {(valid_errors < 10).mean() * 100:.1f}%")

    # 4. SAVE VISUALIZATIONS FOR REPORTING
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pred_rgb = normal_to_rgb(pred_np)
        Image.fromarray(pred_rgb).save(os.path.join(save_dir, f"{obj_name}_pred_normal.png"))
        
        gt_rgb = normal_to_rgb(gt_np * mask_np[..., None])
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"{obj_name}_gt_normal.png"))
        
        mask_vis = (pred_mask_bin * 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(os.path.join(save_dir, f"{obj_name}_pred_mask.png"))

        err_vis = (err_map / 90.0 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(err_vis).save(os.path.join(save_dir, f"{obj_name}_error_map.png"))
        
        np.save(os.path.join(save_dir, f"{obj_name}_pred_normal.npy"), pred_np)

    return {
        "mae": float(mae), "iou": float(iou), "median": float(np.median(valid_errors)), 
        "max": float(np.max(valid_errors)),
        "p10": float((valid_errors < 10).mean() * 100), "p20": float((valid_errors < 20).mean() * 100),
        "p30": float((valid_errors < 30).mean() * 100)
    }
    

def discover_objects(data_root):
    """Scans the directory for valid objects containing the Normal_gt.npy file."""
    objects = []
    if not os.path.isdir(data_root):
        return objects
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        if os.path.exists(os.path.join(obj_dir, "Normal_gt.npy")):
            objects.append(name)
    return objects


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Task Photometric Stereo Models")
    parser.add_argument("--mode", type=str, default="eval_all",
                        choices=["eval", "eval_all", "logo_eval", "eval_json", "eval_mv"],
                        help="eval: single object, eval_all: all objects in dir, eval_json: via JSON split, eval_mv: DiLiGenT-MV benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    
    # [NEW]: Model Type Override parameter
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "transunet", "lightweight", "swin"],
                        help="Force specific model architecture (overrides auto-detect from filename)")
    
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for LOGO mode checkpoints")
    parser.add_argument("--test_object", type=str, default=None, help="Target object for 'eval' mode")
    parser.add_argument("--data_root", type=str, default="./data/training", help="Root directory containing datasets")
    parser.add_argument("--save_output", type=str, default=None, help="Directory to save visual results")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu, cuda, auto)")
    parser.add_argument("--json_file", type=str, default="./val_split.json", help="Path to validation JSON file")
    parser.add_argument("--csv_out", type=str, default="evaluation_results.csv", help="Path to save the resulting CSV metrics")
    parser.add_argument("--no_pred_mask", action="store_true",
                        help="Disable predicted segmentation mask; use GT mask instead (useful when pred mask cuts too much)")
    parser.add_argument("--max_inference_side", type=int, default=None,
                        help="Resize image so its longest side is at most this many pixels before tiling. "
                             "Eliminates tile-grid artifacts on large images. "
                             "Output is upsampled back to original resolution.")
    args = parser.parse_args()

    config = Config(device=args.device)
    device = config.resolve_device()

    # Determine Model Type
    if args.checkpoint:
        if args.model_type != "auto":
            mt = args.model_type
            print(f"[*] FORCED model_type: {mt.upper()}")
        else:
            mt = detect_model_type(args.checkpoint)
            print(f"[*] AUTO-DETECTED model_type: {mt.upper()}")
        
        model_cfg = ModelConfig(model_type=mt, num_classes=4, out_channels=4) 
        model = get_model(model_cfg, model_type=mt).to(device)
        try:
            epoch, val_loss = load_checkpoint(args.checkpoint, model)
            print(f"[*] Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
        except TypeError: # Backward compatibility if load_checkpoint doesn't return epoch/loss
            load_checkpoint(args.checkpoint, model)
            print("[*] Loaded checkpoint successfully.")

    # -------------------------------------------------------------------------
    # JSON VALIDATION MODE
    # -------------------------------------------------------------------------
    if args.mode == "eval_json":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for eval_json mode.")
            return

        try:
            print(f"Loading dataset from JSON split: {args.json_file}")
            val_dataset = SyntheticTestDataset(json_path=args.json_file)
            print(f"Successfully loaded {len(val_dataset)} objects for evaluation.")
        except Exception as e:
            print(f"ERROR: Failed to initialize SyntheticTestDataset. Details: {e}")
            return
        
        all_results = {}
        for idx in range(len(val_dataset)):
            save_dir = os.path.join(args.save_output, val_dataset.folder_list[idx].split('/')[-1]) if args.save_output else None
            metrics = evaluate_object(model, val_dataset, idx, device, save_dir=save_dir, use_pred_mask=not args.no_pred_mask)
            obj_name = val_dataset[idx][3] 
            all_results[obj_name] = metrics
            
        if all_results:
            csv_filename = args.csv_out
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Object", "IoU_Seg(%)", "MAE_Normal", "Median", "Max_Error", "P10", "P20", "P30"])
                for obj, m in all_results.items():
                    writer.writerow([obj, f"{m['iou']*100:.2f}", f"{m['mae']:.4f}", f"{m['median']:.4f}", f"{m['max']:.4f}", f"{m['p10']:.2f}", f"{m['p20']:.2f}", f"{m['p30']:.2f}"])
                
                avg_iou = np.mean([m['iou'] for m in all_results.values()]) * 100
                avg_mae = np.mean([m['mae'] for m in all_results.values()])
                avg_median = np.mean([m['median'] for m in all_results.values()])
                writer.writerow(["AVERAGE", f"{avg_iou:.2f}", f"{avg_mae:.4f}", f"{avg_median:.4f}", "", "", "", ""])
            print(f"\n SUCCESS! Metrics saved to: {csv_filename}")

    # -------------------------------------------------------------------------
    # SINGLE OBJECT MODE
    # -------------------------------------------------------------------------
    elif args.mode == "eval":
        if not args.checkpoint or not args.test_object:
            print("ERROR: Both --checkpoint and --test_object are required for eval mode.")
            return
        
        test_ds = DiLiGentTestDataset(data_root=os.path.abspath(args.data_root), objects=[args.test_object])
        if len(test_ds) == 0:
            print(f"ERROR: Could not load data for '{args.test_object}'")
            return
        evaluate_object(model, test_ds, 0, device, save_dir=args.save_output, use_pred_mask=not args.no_pred_mask)

    # -------------------------------------------------------------------------
    # ALL OBJECTS MODE (Data Root)
    # -------------------------------------------------------------------------
    elif args.mode == "eval_all":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for eval_all mode.")
            return

        data_root = os.path.abspath(args.data_root)
        objects = discover_objects(data_root)
        if not objects:
            print(f"ERROR: No valid objects found in {data_root}")
            return

        all_results = {}
        for obj_name in objects:
            test_ds = DiLiGentTestDataset(data_root=data_root, objects=[obj_name])
            if len(test_ds) == 0: continue
            save_dir = os.path.join(args.save_output, obj_name) if args.save_output else None
            metrics = evaluate_object(model, test_ds, 0, device, save_dir=save_dir, use_pred_mask=not args.no_pred_mask)
            all_results[obj_name] = metrics

        if all_results:
            print(f"\n{'='*60}\n  Evaluation Summary\n{'='*60}")
            for obj, metrics in all_results.items():
                print(f"  {obj:>20s}: {metrics['mae']:.2f}° (IoU: {metrics['iou']*100:.1f}%)")
            
            avg_mae = np.mean([m['mae'] for m in all_results.values()])
            avg_iou = np.mean([m['iou'] for m in all_results.values()]) * 100
            print(f"  {'':>20s}  --------")
            print(f"  {'Average MAE':>20s}: {avg_mae:.2f}°")
            print(f"  {'Average IoU':>20s}: {avg_iou:.2f}%\n{'='*60}")

            # [THÊM MỚI] Lưu kết quả ra file CSV
            csv_filename = args.csv_out
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Object", "IoU_Seg(%)", "MAE_Normal", "Median", "Max_Error", "P10", "P20", "P30"])
                for obj, m in all_results.items():
                    writer.writerow([obj, f"{m['iou']*100:.2f}", f"{m['mae']:.4f}", f"{m['median']:.4f}", f"{m['max']:.4f}", f"{m['p10']:.2f}", f"{m['p20']:.2f}", f"{m['p30']:.2f}"])
                
                avg_median = np.mean([m['median'] for m in all_results.values()])
                writer.writerow(["AVERAGE", f"{avg_iou:.2f}", f"{avg_mae:.4f}", f"{avg_median:.4f}", "", "", "", ""])
            print(f"\n[*] Đã lưu báo cáo CSV tại: {csv_filename}")

    # -------------------------------------------------------------------------
    # DILIGENT-MV BENCHMARK MODE
    # -------------------------------------------------------------------------
    elif args.mode == "eval_mv":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for eval_mv mode.")
            return

        mv_root = os.path.join(os.path.abspath(args.data_root), "mvpmsData")
        if not os.path.isdir(mv_root):
            print(f"ERROR: mvpmsData directory not found at {mv_root}")
            return

        mv_ds = DiLiGentMVTestDataset(mv_root=mv_root)
        if len(mv_ds) == 0:
            print(f"ERROR: No valid (object, view) samples found in {mv_root}")
            return

        all_results = {}
        for idx in range(len(mv_ds)):
            images, normal_gt, mask, name = mv_ds[idx]
            # name format: "{obj}_{view_NN}" where view part is always "view_NN"
            parts = name.rsplit("_view_", 1)
            obj_name = parts[0]
            view_name = "view_" + parts[1] if len(parts) == 2 else name

            C, H, W = images.shape
            print(f"\nEvaluating: {name:>30s} ({C}ch, {H}x{W})")

            pred_out = predict_full_resolution(model, images, device, max_long_side=args.max_inference_side)

            mask_np = mask[0].numpy()

            iou = 0.0
            if not args.no_pred_mask and pred_out.shape[0] >= 4:
                pred_mask_prob = torch.sigmoid(pred_out[3]).numpy()
                pred_mask_bin = (pred_mask_prob > 0.5).astype(np.float32)
                iou = calculate_iou(pred_mask_prob, mask_np)
            else:
                pred_mask_bin = mask_np  # fall back to GT mask

            pred_normal = F.normalize(pred_out[:3], p=2, dim=0)
            pred_normal_masked = pred_normal * torch.from_numpy(pred_mask_bin).to(pred_normal.device)

            pred_np = pred_normal_masked.permute(1, 2, 0).numpy()
            gt_np = normal_gt.permute(1, 2, 0).numpy()

            mae = mean_angular_error(pred_np, gt_np, mask_np)
            err_map = angular_error_map(pred_np, gt_np, mask_np)
            valid_errors = err_map[mask_np > 0.5]

            print(f"  MAE: {mae:.2f}°  IoU: {iou*100:.1f}%  <10°: {(valid_errors < 10).mean()*100:.1f}%")

            if args.save_output:
                save_dir = os.path.join(args.save_output, obj_name, view_name)
                os.makedirs(save_dir, exist_ok=True)
                pred_rgb = normal_to_rgb(pred_np)
                Image.fromarray(pred_rgb).save(os.path.join(save_dir, "pred_normal.png"))
                gt_rgb = normal_to_rgb(gt_np * mask_np[..., None])
                Image.fromarray(gt_rgb).save(os.path.join(save_dir, "gt_normal.png"))
                err_vis = (err_map / 90.0 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(err_vis).save(os.path.join(save_dir, "error_map.png"))

            all_results[name] = {
                "obj": obj_name, "view": view_name,
                "mae": float(mae), "iou": float(iou),
                "median": float(np.median(valid_errors)),
                "max": float(np.max(valid_errors)),
                "p10": float((valid_errors < 10).mean() * 100),
                "p20": float((valid_errors < 20).mean() * 100),
                "p30": float((valid_errors < 30).mean() * 100),
            }

        if all_results:
            # Per-object summary
            from collections import defaultdict
            obj_maes = defaultdict(list)
            obj_ious = defaultdict(list)
            for name, m in all_results.items():
                obj_maes[m["obj"]].append(m["mae"])
                obj_ious[m["obj"]].append(m["iou"])

            print(f"\n{'='*65}\n  DiLiGenT-MV Per-Object Summary\n{'='*65}")
            for obj in sorted(obj_maes):
                avg_mae = np.mean(obj_maes[obj])
                avg_iou = np.mean(obj_ious[obj]) * 100
                n_views = len(obj_maes[obj])
                print(f"  {obj:>15s} ({n_views} views): MAE={avg_mae:.2f}°  IoU={avg_iou:.1f}%")

            overall_mae = np.mean([m["mae"] for m in all_results.values()])
            overall_iou = np.mean([m["iou"] for m in all_results.values()]) * 100
            print(f"  {'Overall':>15s}: MAE={overall_mae:.2f}°  IoU={overall_iou:.1f}%\n{'='*65}")

            # Save CSV
            csv_filename = args.csv_out
            with open(csv_filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Object", "View", "IoU_Seg(%)", "MAE_Normal", "Median", "Max_Error", "P10", "P20", "P30"])
                for name, m in all_results.items():
                    writer.writerow([m["obj"], m["view"], f"{m['iou']*100:.2f}", f"{m['mae']:.4f}",
                                     f"{m['median']:.4f}", f"{m['max']:.4f}",
                                     f"{m['p10']:.2f}", f"{m['p20']:.2f}", f"{m['p30']:.2f}"])
                # Per-object averages
                writer.writerow([])
                writer.writerow(["--- Per-Object Average ---"])
                writer.writerow(["Object", "Views", "Avg_IoU(%)", "Avg_MAE"])
                for obj in sorted(obj_maes):
                    writer.writerow([obj, len(obj_maes[obj]),
                                     f"{np.mean(obj_ious[obj])*100:.2f}",
                                     f"{np.mean(obj_maes[obj]):.4f}"])
                writer.writerow(["OVERALL", len(all_results), f"{overall_iou:.2f}", f"{overall_mae:.4f}"])
            print(f"[*] CSV report saved to: {csv_filename}")


if __name__ == "__main__":
    main()