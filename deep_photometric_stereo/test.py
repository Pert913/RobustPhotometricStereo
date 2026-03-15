"""
Evaluation / inference script for TransUNetPS.

Usage:
    # Evaluate ONE checkpoint on ALL test objects (recommended)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

    # Evaluate on a single test object
    python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG

    # Evaluate and save predicted normal maps
    python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG --save_output ./output

    # Run on all objects (using LOGO checkpoints)
    python test.py --mode logo_eval --checkpoint_dir checkpoints/

    # Test on ALL training objects (10 objects)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

    # Test on ALL testing objects (5 objects)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing

    # Test on BOTH training + testing, save output images
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training --save_output ./output/training
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing --save_output ./output/testing
"""
import argparse
import os
import glob
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
    """
    Predict normal map for full-resolution images using tiling.

    Args:
        model: TransUNetPS or LightweightUNetPS model
        images: (N, 1, H, W) tensor
        device: torch device
        tile_size: tile size for processing
        max_images: max images to use

    Returns:
        pred_normal: (3, H, W) tensor on CPU
    """
    model.eval()
    N, C, H, W = images.shape

    # Pad H and W to be divisible by 16 (for encoder downsampling)
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    if pad_h > 0 or pad_w > 0:
        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, pH, pW = images.shape
    pred_normal = torch.zeros(3, pH, pW)

    # Use tile_size that's divisible by 16
    tile_size = max(16, (tile_size // 16) * 16)

    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            y_end = min(y + tile_size, pH)
            x_end = min(x + tile_size, pW)

            tile_imgs = images[:, :, y:y_end, x:x_end]

            # Subsample N if too many
            if N > max_images:
                indices = torch.linspace(0, N - 1, max_images).long()
                tile_imgs = tile_imgs[indices]
                n_used = max_images
            else:
                n_used = N

            tile_imgs = tile_imgs.unsqueeze(0).to(device)  # (1, N, 1, th, tw)
            counts = torch.tensor([n_used], dtype=torch.long, device=device)

            pred_tile = model(tile_imgs, counts)  # (1, 3, th, tw)
            pred_normal[:, y:y_end, x:x_end] = pred_tile[0].cpu()

    # Remove padding
    pred_normal = pred_normal[:, :H, :W]
    return pred_normal


def evaluate_object(model, test_ds, obj_idx, device, save_dir=None):
    """Evaluate on one object, optionally save outputs."""
    images, normal_gt, mask, obj_name = test_ds[obj_idx]
    N, C, H, W = images.shape

    print(f"\nEvaluating: {obj_name} ({N} images, {H}x{W})")

    # Predict
    pred_normal = predict_full_resolution(model, images, device)

    # Compute metrics
    pred_np = pred_normal.permute(1, 2, 0).numpy()
    gt_np = normal_gt.permute(1, 2, 0).numpy()
    mask_np = mask[0].numpy()

    mae = mean_angular_error(pred_np, gt_np, mask_np)
    err_map = angular_error_map(pred_np, gt_np, mask_np)

    # Statistics
    valid_errors = err_map[mask_np > 0.5]
    print(f"  Mean Angular Error: {mae:.2f}")
    print(f"  Median Error:       {np.median(valid_errors):.2f}")
    print(f"  Max Error:          {np.max(valid_errors):.2f}")
    print(f"  <10 pixels:         {(valid_errors < 10).mean() * 100:.1f}%")
    print(f"  <20 pixels:         {(valid_errors < 20).mean() * 100:.1f}%")
    print(f"  <30 pixels:         {(valid_errors < 30).mean() * 100:.1f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save predicted normal map as PNG
        pred_rgb = normal_to_rgb(pred_np * mask_np[..., None])
        Image.fromarray(pred_rgb).save(os.path.join(save_dir, f"{obj_name}_pred_normal.png"))

        # Save GT normal map as PNG
        gt_rgb = normal_to_rgb(gt_np * mask_np[..., None])
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"{obj_name}_gt_normal.png"))

        # Save error map as grayscale (scaled 0-90)
        err_vis = (err_map / 90.0 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(err_vis).save(os.path.join(save_dir, f"{obj_name}_error_map.png"))

        # Save predicted normals as .npy
        np.save(os.path.join(save_dir, f"{obj_name}_pred_normal.npy"), pred_np)

        print(f"  Saved outputs to {save_dir}/")

    return mae


def discover_objects(data_root):
    """Find all object directories that have Normal_gt.npy (prepared data)."""
    objects = []
    if not os.path.isdir(data_root):
        return objects
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        # Must have Normal_gt.npy (from prepare_data.py) and some images
        if os.path.exists(os.path.join(obj_dir, "Normal_gt.npy")):
            objects.append(name)
    return objects


def main():
    parser = argparse.ArgumentParser(description="Evaluate TransUNetPS")
    parser.add_argument("--mode", type=str, default="eval_all",
                        choices=["eval", "eval_all", "logo_eval"],
                        help="eval: single object, eval_all: all objects in data_root, logo_eval: LOGO folds")
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
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = Config(device=args.device)
    device = config.resolve_device()

    # eval: single object ======
    if args.mode == "eval":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return
        if not args.test_object:
            print("ERROR: --test_object required for --mode eval")
            return

        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        model = get_model(config.model, model_type=mt).to(device)
        epoch, val_loss = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
        print(f"Parameters: {count_parameters(model):,}")

        test_ds = DiLiGentTestDataset(
            data_root=os.path.abspath(args.data_root),
            objects=[args.test_object],
        )

        if len(test_ds) == 0:
            print(f"ERROR: Could not load test object '{args.test_object}'")
            return

        evaluate_object(model, test_ds, 0, device, save_dir=args.save_output)

    # eval_all: one checkpoint, all objects in data_root ==
    elif args.mode == "eval_all":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return

        data_root = os.path.abspath(args.data_root)
        objects = discover_objects(data_root)
        if not objects:
            print(f"ERROR: No prepared objects found in {data_root}")
            print("  (Each object needs Normal_gt.npy — run prepare_data.py first)")
            return

        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        model = get_model(config.model, model_type=mt).to(device)
        epoch, val_loss = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Data root: {data_root}")
        print(f"Objects found: {len(objects)} — {objects}")

        all_results = {}
        for obj_name in objects:
            test_ds = DiLiGentTestDataset(
                data_root=data_root,
                objects=[obj_name],
            )
            if len(test_ds) == 0:
                print(f"\n  WARNING: Could not load {obj_name}, skipping")
                continue

            save_dir = os.path.join(args.save_output, obj_name) if args.save_output else None
            mae = evaluate_object(model, test_ds, 0, device, save_dir=save_dir)
            all_results[obj_name] = mae

        # Print summary table
        if all_results:
            print(f"\n{'='*60}")
            print(f"  Evaluation Summary — {len(all_results)} objects")
            print(f"  Checkpoint: {args.checkpoint}")
            print(f"  Data root:  {data_root}")
            print(f"{'='*60}")
            for obj, mae in all_results.items():
                print(f"  {obj:>20s}: {mae:.2f}")
            avg = np.mean(list(all_results.values()))
            median = np.median(list(all_results.values()))
            best_obj = min(all_results, key=all_results.get)
            worst_obj = max(all_results, key=all_results.get)
            print(f"  {'':>20s}  --------")
            print(f"  {'Average':>20s}: {avg:.2f}")
            print(f"  {'Median':>20s}: {median:.2f}")
            print(f"  {'Best':>20s}: {all_results[best_obj]:.2f} ({best_obj})")
            print(f"  {'Worst':>20s}: {all_results[worst_obj]:.2f} ({worst_obj})")
            print(f"{'='*60}")

    # logo_eval: one checkpoint per fold
    elif args.mode == "logo_eval":
        data_root = os.path.abspath(args.data_root)
        objects = config.data.objects
        all_results = {}

        for test_obj in objects:
            ckpt_path = os.path.join(args.checkpoint_dir, f"logo_{test_obj}", "best.pt")
            if not os.path.exists(ckpt_path):
                print(f"WARNING: No checkpoint for {test_obj} at {ckpt_path}")
                continue

            mt = detect_model_type(ckpt_path)
            model = get_model(config.model, model_type=mt).to(device)
            load_checkpoint(ckpt_path, model)

            test_ds = DiLiGentTestDataset(
                data_root=data_root,
                objects=[test_obj],
            )
            if len(test_ds) == 0:
                continue

            save_dir = os.path.join(args.save_output, test_obj) if args.save_output else None
            mae = evaluate_object(model, test_ds, 0, device, save_dir=save_dir)
            all_results[test_obj] = mae

        if all_results:
            print(f"\n{'='*50}")
            print("  LOGO Evaluation Summary")
            print(f"{'='*50}")
            for obj, mae in all_results.items():
                print(f"  {obj:>20s}: {mae:.2f}")
            avg = np.mean(list(all_results.values()))
            print(f"  {'Average':>20s}: {avg:.2f}")
            print(f"{'='*50}")


if __name__ == "__main__":
    main()
