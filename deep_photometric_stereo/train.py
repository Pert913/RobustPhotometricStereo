"""
Training script for UNetPS — Uncalibrated Photometric Stereo.

Usage:
    # Train with leave-one-object-out cross-validation
    python train.py --mode logo_cv --epochs 200

    # Train on specific objects, test on another
    python train.py --mode train --train_objects buddha bunny --test_object cat --epochs 200

    # Quick test run
    python train.py --mode train --epochs 5 --patches_per_epoch 100

    # Train (TransUNet model — default)
    python train.py --mode logo_cv --epochs 150

    # Train (Lightweight model — faster, smaller)
    python train.py --mode logo_cv --epochs 200 --model_type lightweight

    # General training command using our approach UNet-Transformer ( lightweight model, 150 epochs)
    python train.py --mode train \
      --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
      --test_object bootaoPNG \
      --model_type lightweight \
      --epochs 150

    # General training command using exactly TransUnet in the Paper
    python train.py --mode train \
      --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
      --test_object bootaoPNG \
      --model_type transunet \
      --epochs 150

    # Tue, you can use this command to train  if your computer is good enough (using exactly TransUnet in the Paper) (NOTE: only support DiLiGentDataset + New Pictures from Dropbox)
    python train.py --mode train \
      --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
      --test_object ballPNG \
      --model_type transunet \
      --epochs 150 \
      --batch_size 4 \
      --patches_per_epoch 2000

    # OR Lightweight model (faster, ~4.6M params) (NOTE: only support DiLiGentDataset + New Pictures from Dropbox))
    python train.py --mode train \
          --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
          --test_object ballPNG \
          --model_type lightweight \
          --epochs 150 \
          --batch_size 4 \
          --patches_per_epoch 2000

    # Train Synapse segmentation (following TransUNet paper exactly)
    python train.py --mode synapse --model_type transunet --epochs 150

    # With custom batch size (lower if OOM)
    python train.py --mode synapse --model_type transunet --epochs 150 --batch_size 2

    # Checkpoint saved to: checkpoints/synapse/best.pt
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset import DiLiGentDataset, DiLiGentTestDataset, collate_variable_n
from model import get_model
from losses import CombinedNormalLoss, SegmentationLoss
from utils import (
    mean_angular_error, save_checkpoint, load_checkpoint,
    count_parameters, normal_to_rgb,
)


def poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9):
    """Polynomial LR decay (TransUNet trainer.py)."""
    lr = base_lr * (1.0 - iter_num / max_iters) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=1.0, log_interval=50,
                    base_lr=0.01, iter_offset=0, max_iters=10000, lr_power=0.9):
    """Train for one epoch with per-iteration polynomial LR. Returns (avg_loss, iter_count)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    iter_num = iter_offset

    for batch_idx, (images, normals, masks, image_counts) in enumerate(loader):
        images = images.to(device)
        normals = normals.to(device)
        masks = masks.to(device)
        image_counts = image_counts.to(device)

        optimizer.zero_grad()
        pred = model(images, image_counts)
        loss = criterion(pred, normals, masks)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Polynomial LR decay per iteration (following TransUNet)
        iter_num += 1
        poly_lr(optimizer, base_lr, iter_num, max_iters, lr_power)

        total_loss += loss.item()
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}, loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1), iter_num


@torch.no_grad()
def evaluate(model, test_dataset, device, max_images=96):
    """
    Evaluate on full-resolution test objects.
    Processes images in chunks to handle memory constraints.

    Returns dict: {object_name: mean_angular_error_degrees}
    """
    model.eval()
    results = {}

    for idx in range(len(test_dataset)):
        images, normal_gt, mask, obj_name = test_dataset[idx]
        # images: (N, 1, H, W), normal_gt: (3, H, W), mask: (1, H, W)

        N, C, H, W = images.shape

        # Process in spatial tiles to avoid OOM on full resolution
        tile_size = 128
        pred_normal = torch.zeros(3, H, W)

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)

                tile_imgs = images[:, :, y:y_end, x:x_end]  # (N, 1, th, tw)

                # Limit N for memory
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

        # Compute angular error
        pred_np = pred_normal.permute(1, 2, 0).numpy()  # (H, W, 3)
        gt_np = normal_gt.permute(1, 2, 0).numpy()
        mask_np = mask[0].numpy()

        mae = mean_angular_error(pred_np, gt_np, mask_np)
        results[obj_name] = mae
        print(f"  {obj_name}: MAE = {mae:.2f}°")

    return results


def train_fold(
    train_objects, test_object, config, fold_name="fold",
):
    """Train on train_objects, evaluate on test_object."""
    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  {fold_name}: train={train_objects}, test={test_object}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # Dataset
    train_ds = DiLiGentDataset(
        data_root=config.data.data_root,
        objects=train_objects,
        num_images=config.data.num_input_images,
        min_images=config.data.min_input_images,
        patch_size=config.data.patch_size,
        patches_per_epoch=config.data.patches_per_epoch,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_variable_n,
        drop_last=True,
    )

    test_ds = DiLiGentTestDataset(
        data_root=config.data.data_root,
        objects=[test_object],
        num_images=config.data.num_lights,
    )

    # Model
    model = get_model(config.model, model_type=config.model.model_type).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer — SGD with momentum (following TransUNet trainer.py)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.train.lr,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )
    # Polynomial LR decay (following TransUNet trainer.py)
    max_iters = config.train.epochs * len(train_loader)
    lr_power = config.train.lr_power
    base_lr = config.train.lr

    # Loss
    criterion = CombinedNormalLoss(
        angular_w=config.train.angular_weight,
        cosine_w=config.train.cosine_weight,
        l1_w=config.train.l1_weight,
    )

    # Training loop
    best_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    iter_num = 0

    save_dir = os.path.join(config.train.save_dir, fold_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        train_loss, iter_num = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.train.grad_clip,
            log_interval=config.train.log_interval,
            base_lr=base_lr, iter_offset=iter_num,
            max_iters=max_iters, lr_power=lr_power,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.train.epochs} — "
              f"loss={train_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")

        # Evaluate every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == config.train.epochs:
            results = evaluate(model, test_ds, device)
            mae = list(results.values())[0] if results else float("inf")

            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(save_dir, "best.pt")
                save_checkpoint(model, optimizer, epoch, mae, ckpt_path,
                                model_type=config.model.model_type)
                print(f"  ★ New best: {mae:.2f}° (saved)")
            else:
                patience_counter += 10

            if patience_counter >= config.train.patience:
                print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    print(f"\n{fold_name} done. Best MAE: {best_mae:.2f}° at epoch {best_epoch}")
    return best_mae


def logo_cv(config):
    """Leave-One-Object-Out cross-validation."""
    objects = config.data.objects
    results = {}

    for test_obj in objects:
        train_objs = [o for o in objects if o != test_obj]
        mae = train_fold(train_objs, test_obj, config, fold_name=f"logo_{test_obj}")
        results[test_obj] = mae

    print(f"\n{'='*60}")
    print("  LOGO Cross-Validation Results")
    print(f"{'='*60}")
    for obj, mae in results.items():
        print(f"  {obj:>10s}: {mae:.2f}°")
    avg = np.mean(list(results.values()))
    print(f"  {'Average':>10s}: {avg:.2f}°")
    print(f"{'='*60}\n")

    return results


# Synapse segmentation training (following TransUNet trainer.py)

class SynapseDataset(torch.utils.data.Dataset):
    """
    Synapse dataset loader (following TransUNet datasets/dataset_synapse.py).
    Loads .npz slices for training.
    """

    def __init__(self, base_dir, list_file, img_size=224):
        self.base_dir = base_dir
        self.img_size = img_size
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
        print(f"SynapseDataset: {len(self.sample_list)} slices from {base_dir}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data = np.load(os.path.join(self.base_dir, slice_name + ".npz"))
        image = data["image"].astype(np.float32)  # (H, W)
        label = data["label"].astype(np.float32)  # (H, W)

        # Resize to img_size
        from PIL import Image as PILImage
        image = np.array(PILImage.fromarray(image).resize(
            (self.img_size, self.img_size), PILImage.BICUBIC))
        label = np.array(PILImage.fromarray(label).resize(
            (self.img_size, self.img_size), PILImage.NEAREST))

        # Random augmentation (following TransUNet RandomGenerator)
        if np.random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()
        if np.random.random() > 0.5:
            from scipy.ndimage import rotate as scipy_rotate
            angle = np.random.uniform(-20, 20)
            image = scipy_rotate(image, angle, order=3, reshape=False)
            label = scipy_rotate(label, angle, order=0, reshape=False)

        # To tensor: (1, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).float()
        return image, label


def train_synapse(config, args):
    """
    Train on Synapse dataset (following TransUNet trainer.py exactly).
    Loss: 0.5 * CE + 0.5 * Dice
    Optimizer: SGD(lr=0.01, momentum=0.9, wd=1e-4)
    LR: polynomial decay (power=0.9)
    """
    device = config.resolve_device()
    synapse_dir = os.path.abspath(args.synapse_root)
    num_classes = args.num_classes
    img_size = 224

    print(f"\n{'='*60}")
    print(f"  Synapse Segmentation Training")
    print(f"  data: {synapse_dir}")
    print(f"  classes: {num_classes}, img_size: {img_size}")
    print(f"  model_type: {args.model_type}, device: {device}")
    print(f"{'='*60}\n")

    # Dataset
    train_list = os.path.join(synapse_dir, "train.txt")
    train_npz_dir = os.path.join(synapse_dir, "train_npz")
    train_ds = SynapseDataset(train_npz_dir, train_list, img_size=img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Model in segmentation mode
    model_cfg = ModelConfig(
        model_type=args.model_type,
        mode="segmentation",
        num_classes=num_classes,
        in_channels=1,
    )
    model = get_model(model_cfg, model_type=args.model_type).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer — SGD (TransUNet trainer.py)
    base_lr = args.lr if args.lr != 1e-3 else 0.01  # default 0.01 for synapse
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    # Loss — 0.5 * CE + 0.5 * Dice (TransUNet trainer.py)
    criterion = SegmentationLoss(num_classes=num_classes)

    # Training loop
    max_epochs = args.epochs if args.epochs != 200 else 150  # default 150 for synapse
    max_iters = max_epochs * len(train_loader)
    iter_num = 0

    save_dir = os.path.join(args.save_dir, "synapse")
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)   # (B, 1, H, W)
            labels = labels.to(device)   # (B, H, W)

            optimizer.zero_grad()
            outputs = model(images)      # (B, num_classes, H, W)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Poly LR per iteration
            iter_num += 1
            poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9)

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx+1}/{len(train_loader)}, loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{max_epochs} — loss={avg_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")

        # Save best + periodic checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path,
                            model_type=args.model_type)
            print(f"  New best loss: {avg_loss:.4f} (saved)")

        if epoch % 50 == 0 or epoch == max_epochs:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path,
                            model_type=args.model_type)

    print(f"\nSynapse training done. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train TransUNetPS")
    parser.add_argument("--mode", type=str, default="logo_cv",
                        choices=["train", "logo_cv", "synapse"],
                        help="Training mode: train, logo_cv, or synapse")
    parser.add_argument("--data_root", type=str, default="./data/training")
    parser.add_argument("--synapse_root", type=str, default="./data/Synapse",
                        help="Synapse dataset root (for --mode synapse)")
    parser.add_argument("--num_classes", type=int, default=9,
                        help="Number of segmentation classes (for --mode synapse)")
    parser.add_argument("--train_objects", nargs="+", default=None,
                        help="Objects to train on (for --mode train)")
    parser.add_argument("--test_object", type=str, default=None,
                        help="Object to test on (for --mode train)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_epoch", type=int, default=2000)
    parser.add_argument("--num_images", type=int, default=32,
                        help="Max number of input images per sample")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="transunet",
                        choices=["transunet", "lightweight"],
                        help="Model architecture: transunet (~11M) or lightweight (~4.7M)")
    args = parser.parse_args()

    if args.mode == "synapse":
        config = Config(device=args.device)
        train_synapse(config, args)
    else:
        # Photometric stereo modes
        config = Config(
            data=DataConfig(
                data_root=os.path.abspath(args.data_root),
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                patches_per_epoch=args.patches_per_epoch,
                num_input_images=args.num_images,
            ),
            model=ModelConfig(model_type=args.model_type),
            train=TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                save_dir=args.save_dir,
            ),
            device=args.device,
        )

        if args.mode == "logo_cv":
            logo_cv(config)
        elif args.mode == "train":
            all_objs = config.data.objects
            train_objs = args.train_objects or all_objs[:-1]
            test_obj = args.test_object or all_objs[-1]
            train_fold(train_objs, test_obj, config, fold_name="train")


if __name__ == "__main__":
    main()
