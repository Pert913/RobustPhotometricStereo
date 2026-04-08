"""
Training script for UNetPS — Uncalibrated Photometric Stereo.
Supports both DiLiGenT (Old) and Synthetic 8-8-8 LED (New) datasets.

Usage:
    # --- OLD EXPERIMENTS ---
    python train.py --mode logo_cv --epochs 200
    python train.py --mode train --train_objects buddha bunny --test_object cat --epochs 200

    # --- NEW SYNTHETIC 300GB DATASET ---
    python train.py --mode synthetic --json_path valid_samples.json --batch_size 8 --epochs 50
"""
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import Config, DataConfig, ModelConfig, TrainConfig

# [BỔ SUNG MỚI]: Import thêm SyntheticDataset và SyntheticTestDataset
from dataset import (
    DiLiGentDataset, DiLiGentTestDataset, collate_variable_n,
    SyntheticDataset, SyntheticTestDataset
)
from model import get_model
from losses import CombinedNormalLoss, SegmentationLoss
from utils import (
    mean_angular_error, save_checkpoint, load_checkpoint,
    count_parameters, normal_to_rgb, plot_training_curve
)

# ==============================================================================
# 1. SHARED UTILS
# ==============================================================================
def poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9):
    """Polynomial LR decay (TransUNet trainer.py)."""
    lr = base_lr * (1.0 - iter_num / max_iters) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ==============================================================================
# 2. OLD PIPELINE (DiLiGenT) - GIỮ NGUYÊN 100%
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=1.0, log_interval=50,
                    base_lr=0.01, iter_offset=0, max_iters=10000, lr_power=0.9,
                    use_poly_lr=True):
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

        iter_num += 1
        if use_poly_lr:
            poly_lr(optimizer, base_lr, iter_num, max_iters, lr_power)

        total_loss += loss.item()
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}, loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1), iter_num

@torch.no_grad()
def evaluate(model, test_dataset, device, max_images=96):
    model.eval()
    results = {}

    for idx in range(len(test_dataset)):
        images, normal_gt, mask, obj_name = test_dataset[idx]
        N, C, H, W = images.shape

        tile_size = 128
        pred_normal = torch.zeros(3, H, W)

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)

                tile_imgs = images[:, :, y:y_end, x:x_end]

                if N > max_images:
                    indices = torch.linspace(0, N - 1, max_images).long()
                    tile_imgs = tile_imgs[indices]
                    n_used = max_images
                else:
                    n_used = N

                tile_imgs = tile_imgs.unsqueeze(0).to(device) 
                counts = torch.tensor([n_used], dtype=torch.long, device=device)

                pred_tile = model(tile_imgs, counts) 
                pred_normal[:, y:y_end, x:x_end] = pred_tile[0].cpu()

        pred_np = pred_normal.permute(1, 2, 0).numpy()
        gt_np = normal_gt.permute(1, 2, 0).numpy()
        mask_np = mask[0].numpy()

        mae = mean_angular_error(pred_np, gt_np, mask_np)
        results[obj_name] = mae
        print(f"  {obj_name}: MAE = {mae:.2f}°")

    return results

def train_fold(train_objects, test_object, config, fold_name="fold"):
    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  {fold_name}: train={train_objects}, test={test_object}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

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
        train_ds, batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.num_workers, collate_fn=collate_variable_n, drop_last=True,
    )

    test_ds = DiLiGentTestDataset(
        data_root=config.data.data_root, objects=[test_object], num_images=config.data.num_lights,
    )

    if hasattr(config.model, 'in_channels'):
        config.model.in_channels = 3
    
    model = get_model(config.model, model_type=config.model.model_type).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    use_adamw = config.train.optimizer == "adamw"
    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)

    max_iters = config.train.epochs * len(train_loader)
    lr_power = config.train.lr_power
    base_lr = config.train.lr

    criterion = CombinedNormalLoss(angular_w=config.train.angular_weight, cosine_w=config.train.cosine_weight, l1_w=config.train.l1_weight)

    best_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    iter_num = 0

    save_dir = os.path.join(config.train.save_dir, fold_name)
    os.makedirs(save_dir, exist_ok=True)

    history_losses = [] 
    val_epochs = []
    val_maes = []

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        train_loss, iter_num = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.train.grad_clip, log_interval=config.train.log_interval,
            base_lr=base_lr, iter_offset=iter_num, max_iters=max_iters, lr_power=lr_power,
            use_poly_lr=not use_adamw,
        )

        if use_adamw: scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.train.epochs} — loss={train_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")
        history_losses.append(train_loss)

        if epoch % 10 == 0 or epoch == config.train.epochs:
            results = evaluate(model, test_ds, device, max_images=12)
            mae = list(results.values())[0] if results else float("inf")

            if mae != float("inf"):
                val_epochs.append(epoch)
                val_maes.append(mae)
                
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(save_dir, "best.pt")
                save_checkpoint(model, optimizer, epoch, mae, ckpt_path, model_type=config.model.model_type)
                print(f"  ★ New best: {mae:.2f}° (saved)")
            else:
                patience_counter += 10

            if patience_counter >= config.train.patience:
                print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    print(f"\n{fold_name} done. Best MAE: {best_mae:.2f}° at epoch {best_epoch}")
    plot_path = os.path.join(save_dir, "training_val_curve.png")
    plot_training_curve(history_losses, val_epochs, val_maes, plot_path)
    return best_mae

def logo_cv(config):
    objects = config.data.objects
    results = {}
    for test_obj in objects:
        train_objs = [o for o in objects if o != test_obj]
        mae = train_fold(train_objs, test_obj, config, fold_name=f"logo_{test_obj}")
        results[test_obj] = mae
    return results


# ==============================================================================
# 3. SYNAPSE SEGMENTATION - GIỮ NGUYÊN 100%
# ==============================================================================
class SynapseDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, list_file, img_size=224):
        self.base_dir = base_dir
        self.img_size = img_size
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
    def __len__(self): return len(self.sample_list)
    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data = np.load(os.path.join(self.base_dir, slice_name + ".npz"))
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.float32)
        from PIL import Image as PILImage
        image = np.array(PILImage.fromarray(image).resize((self.img_size, self.img_size), PILImage.BICUBIC))
        label = np.array(PILImage.fromarray(label).resize((self.img_size, self.img_size), PILImage.NEAREST))
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).float()
        return image, label

def train_synapse(config, args):
    device = config.resolve_device()
    synapse_dir = os.path.abspath(args.synapse_root)
    num_classes = args.num_classes
    img_size = 224

    train_list = os.path.join(synapse_dir, "train.txt")
    train_npz_dir = os.path.join(synapse_dir, "train_npz")
    train_ds = SynapseDataset(train_npz_dir, train_list, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model_cfg = ModelConfig(model_type=args.model_type, mode="segmentation", num_classes=num_classes, in_channels=1)
    model = get_model(model_cfg, model_type=args.model_type).to(device)

    base_lr = args.lr if args.lr != 1e-3 else 0.01 
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = SegmentationLoss(num_classes=num_classes)

    max_epochs = args.epochs if args.epochs != 200 else 150 
    max_iters = max_epochs * len(train_loader)
    iter_num = 0
    save_dir = os.path.join(args.save_dir, "synapse")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_num += 1
            poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9)
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(save_dir, "best.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path, model_type=args.model_type)


# ==============================================================================
# 4. [THÊM MỚI] NEW SYNTHETIC PIPELINE (8-8-8 LED)
# ==============================================================================
def train_one_epoch_synthetic(model, loader, criterion, optimizer, device,
                              grad_clip=1.0, log_interval=50,
                              base_lr=0.01, iter_offset=0, max_iters=10000, lr_power=0.9,
                              use_poly_lr=True, scaler=None):
    """
    Hàm train tối ưu hóa cho dữ liệu Synthetic.
    Đã tích hợp Mixed Precision và Real-time MAE logging.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    iter_num = iter_offset

    for batch_idx, (images, normals, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        normals = normals.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # --- 1. MIXED PRECISION (Tăng tốc cho card RTX) ---
        with autocast():
            try:
                pred = model(images)
            except TypeError:
                # Fallback cho kiến trúc cũ cần image_counts
                counts = torch.ones(images.shape[0], dtype=torch.long, device=device)
                pred = model(images.unsqueeze(1), counts) if images.ndim == 4 else model(images, counts)

            # --- 2. TÍNH LOSS CHIẾN THUẬT (Cosine + L1) ---
            mask_bool = masks > 0.5
            if mask_bool.any():
                # Ép phẳng dữ liệu để tính toán vector nhanh nhất
                # pred: (B, 3, H, W) -> (Pixels, 3)
                p = pred.permute(0, 2, 3, 1)[mask_bool.squeeze(1)]
                g = normals.permute(0, 2, 3, 1)[mask_bool.squeeze(1)]
                
                # Chuẩn hóa vector (Vô cùng quan trọng)
                p = F.normalize(p, p=2, dim=-1, eps=1e-8)
                g = F.normalize(g, p=2, dim=-1, eps=1e-8)

                # Loss kết hợp
                loss_l1 = F.l1_loss(p, g)
                cos_sim = F.cosine_similarity(p, g, dim=-1)
                loss_cos = 1.0 - cos_sim.mean()
                
                loss = loss_l1 + loss_cos * 0.5
                
                # Tính nhanh MAE (độ) để log terminal
                with torch.no_grad():
                    # Góc lệch = acos(dot_product)
                    # Clip để tránh lỗi NaN của acos tại 1.0000001
                    angle_err = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7))
                    batch_mae = torch.mean(angle_err) * (180.0 / np.pi)
            else:
                loss = criterion(pred, normals, masks)
                batch_mae = 0.0

        # --- 3. BACKWARD VỚI SCALER (Nếu dùng Mixed Precision) ---
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Cập nhật LR (Poly)
        iter_num += 1
        if use_poly_lr:
            poly_lr(optimizer, base_lr, iter_num, max_iters, lr_power)

        total_loss += loss.item()
        total_mae += batch_mae.item() if isinstance(batch_mae, torch.Tensor) else batch_mae
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | MAE: {batch_mae:.2f}°")

    return total_loss / max(n_batches, 1), iter_num

@torch.no_grad()
def evaluate_synthetic(model, test_dataset, device):
    """
    Hàm đánh giá MAE và xuất ảnh debug_visual.png gồm 3 phần:
    Dự đoán - Ground Truth - Ảnh đầu vào (Input).
    """
    model.eval()
    maes = []
    
    # Eval nhanh 100 mẫu, nhưng chỉ lưu ảnh mẫu đầu tiên
    max_eval = min(len(test_dataset), 100) 
    print(f"  [Eval] Bắt đầu đánh giá trên {max_eval} samples...")

    for idx in range(max_eval):
        images, normal_gt, mask, folder_name = test_dataset[idx]
        
        # 1. Dự đoán
        img_input = images.unsqueeze(0).to(device)
        from torch.cuda.amp import autocast
        with autocast():
            try:
                pred = model(img_input)
            except TypeError:
                counts = torch.ones(1, dtype=torch.long, device=device)
                pred = model(img_input.unsqueeze(1), counts)

        # 2. Xử lý dữ liệu Numpy
        pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)
        gt_np = normal_gt.numpy().transpose(1, 2, 0)
        mask_np = mask[0].numpy()
        
        # Chuẩn hóa vector dự đoán về độ dài 1
        norm = np.linalg.norm(pred_np, axis=-1, keepdims=True) + 1e-8
        pred_np = pred_np / norm

        mae = mean_angular_error(pred_np, gt_np, mask_np)
        maes.append(mae)

        # 3. LOGIC VISUAL DEBUG (Lưu ảnh tại mẫu đầu tiên)
        if idx == 0:
            # Chuẩn bị ảnh Normal ([-1, 1] -> [0, 1])
            vis_pred = (pred_np + 1.0) / 2.0
            vis_gt = (gt_np + 1.0) / 2.0
            
            # Chuẩn bị ảnh Input (RGB)
            # Vì ảnh input đã bị chuẩn hóa Mean/Std, ta cần đưa nó về dải [0, 1] để xem
            vis_input = images.numpy().transpose(1, 2, 0)
            vis_input = (vis_input - vis_input.min()) / (vis_input.max() - vis_input.min() + 1e-8)
            
            # Áp mask
            vis_pred[mask_np == 0] = 0
            vis_gt[mask_np == 0] = 0
            vis_input[mask_np == 0] = 0

            plt.figure(figsize=(20, 7))
            
            # Cột 1: Dự đoán của AI
            plt.subplot(1, 3, 1)
            plt.imshow(vis_pred)
            plt.title(f"Prediction (MAE: {mae:.2f}°)")
            plt.axis('off')
            
            # Cột 2: Ground Truth (Mục tiêu)
            plt.subplot(1, 3, 2)
            plt.imshow(vis_gt)
            plt.title("Ground Truth (Target)")
            plt.axis('off')
            
            # Cột 3: Ảnh đầu vào (Cái mà AI nhìn thấy)
            plt.subplot(1, 3, 3)
            plt.imshow(vis_input)
            plt.title("Model Input (Simulated 8-8-8)")
            plt.axis('off')
            
            plt.savefig("debug_visual.png", bbox_inches='tight')
            plt.close()
            print(f"  🖼️ Đã lưu ảnh debug 3 cột tại: debug_visual.png")

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    -> Progress: {idx+1}/{max_eval}. Current MAE = {mae:.2f}°")

    avg_mae = np.mean(maes)
    print(f"  [Eval] MAE trung bình: {avg_mae:.2f}°")
    return avg_mae


def train_synthetic_pipeline(config, args):
    """Luồng huấn luyện chính tự động phân tách 90/10 dữ liệu 300GB."""
    device = config.resolve_device()
    print(f"\n{'='*70}")
    print(f"  🚀 SYNTHETIC 8-8-8 LED TRAINING PIPELINE")
    print(f"  JSON List: {args.json_path}")
    print(f"  Device: {device} | Model: {args.model_type}")
    print(f"{'='*70}\n")

    # 1. Đọc và chia Split 90/10
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"Không tìm thấy file: {args.json_path}")
        
    with open(args.json_path, 'r') as f:
        all_folders = json.load(f)
        
    random.seed(42) # Cố định seed để chia train/val luôn giống nhau mỗi lần chạy
    random.shuffle(all_folders)
    if len(all_folders) == 1:
        # Nếu chỉ có 1 mẫu (đang debug), dùng mẫu đó cho cả train và val
        train_folders = all_folders
        val_folders = all_folders 
    else:
        # Chia 90/10 bình thường nhưng đảm bảo tập Train có ít nhất 1 mẫu
        split_idx = int(len(all_folders) * 0.9)
        if split_idx == 0: split_idx = 1
        
        train_folders = all_folders[:split_idx]
        val_folders = all_folders[split_idx:]

    save_dir = os.path.join(config.train.save_dir, "synthetic_run")
    os.makedirs(save_dir, exist_ok=True)

    train_json = os.path.join(save_dir, 'train_split.json')
    val_json = os.path.join(save_dir, 'val_split.json')
    with open(train_json, 'w') as f: json.dump(train_folders, f)
    with open(val_json, 'w') as f: json.dump(val_folders, f)

    print(f"  📦 Đã chia dữ liệu: {len(train_folders)} Train | {len(val_folders)} Validation")

    # 2. Khởi tạo Datasets
    train_ds = SyntheticDataset(
        json_path=train_json,
        patch_size=config.data.patch_size,
        alpha_min=0.4,
        augment=False # [SỬA] TẮT FLIP ẢNH ĐỂ ĐẢM BẢO ÁNH SÁNG CHUẨN XÁC 100%
    )
    
    # [QUAN TRỌNG]: Không dùng collate_fn=collate_variable_n nữa
    train_loader = DataLoader(
            train_ds, 
            batch_size=config.data.batch_size, 
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last=(len(train_folders) > config.data.batch_size),
            
            # --- BỘ 3 CỜ TURBO ---
            pin_memory=True,            # Ép dùng RAM tốc độ cao để bắn thẳng data vào GPU (Bỏ qua trạm trung chuyển)
            persistent_workers=True if config.data.num_workers > 0 else False, # Giữ công nhân sống sót giữa các Epoch, không bắt họ nghỉ việc rồi thuê lại
            prefetch_factor=3 if config.data.num_workers > 0 else None         # Ép CPU nhào nặn sẵn 3 batch đút túi chờ GPU tính xong là nhét vào miệng luôn
    )

    val_ds = SyntheticTestDataset(json_path=val_json, alpha_min=0.4)

    # 3. Khởi tạo Model
    if hasattr(config.model, 'in_channels'):
        config.model.in_channels = 3 # Khóa cứng in_channels = 3 vì đầu vào là 1 ảnh RGB
        
    model = get_model(config.model, model_type=config.model.model_type).to(device)
    print(f"  🧠 Model params: {count_parameters(model):,}")

    # 4. Optimizer & Loss
    use_adamw = config.train.optimizer == "adamw"
    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)

    criterion = CombinedNormalLoss(angular_w=config.train.angular_weight, cosine_w=config.train.cosine_weight, l1_w=config.train.l1_weight)

    start_epoch = 1
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        print(f"  ♻️ Đang tải checkpoint để chạy tiếp: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"  🚀 Sẽ bắt đầu huấn luyện tiếp từ Epoch {start_epoch}")

    # 5. Vòng lặp Training
    best_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    iter_num = 0
    max_iters = config.train.epochs * len(train_loader)

    history_losses = [] 
    val_epochs = []
    val_maes = []

    for epoch in range(start_epoch, config.train.epochs + 1):
        t0 = time.time()
        train_loss, iter_num = train_one_epoch_synthetic(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.train.grad_clip, log_interval=config.train.log_interval,
            base_lr=config.train.lr, iter_offset=iter_num, max_iters=max_iters, lr_power=config.train.lr_power,
            use_poly_lr=not use_adamw,
        )

        if use_adamw: scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.train.epochs} — loss={train_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")
        history_losses.append(train_loss)

        # Đánh giá MAE mỗi 5 epoch (chạy Validation mất thời gian nên không chạy mỗi epoch)
        if epoch % 5 == 0 or epoch == config.train.epochs:
            mae = evaluate_synthetic(model, val_ds, device)
            
            val_epochs.append(epoch)
            val_maes.append(mae)
                
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = os.path.join(save_dir, "best_synthetic.pt")
                save_checkpoint(model, optimizer, epoch, mae, ckpt_path, model_type=config.model.model_type)
                print(f"  ★ New best: {mae:.2f}° (saved)")
            else:
                patience_counter += 5

            if patience_counter >= config.train.patience:
                print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    print(f"\n✅ Synthetic Training done. Best MAE: {best_mae:.2f}° at epoch {best_epoch}")
    plot_path = os.path.join(save_dir, "synthetic_training_curve.png")
    plot_training_curve(history_losses, val_epochs, val_maes, plot_path)


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train UNetPS")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["train", "logo_cv", "synapse", "synthetic"],
                        help="Training mode: train, logo_cv, synapse, or synthetic (for 300GB dataset)")
    
    parser.add_argument("--json_path", type=str, default="valid_samples_ssd_clean.json",
                        help="Path to the JSON file containing valid 300GB folder paths")

    parser.add_argument("--data_root", type=str, default="./data/training")
    parser.add_argument("--synapse_root", type=str, default="./data/Synapse")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--train_objects", nargs="+", default=None)
    parser.add_argument("--test_object", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8) # Tăng default batch size vì giờ input rất nhẹ
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_epoch", type=int, default=2000)
    parser.add_argument("--num_images", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="lightweight",
                        choices=["transunet", "lightweight"])
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--num_workers", type=int, default=4, help="Số luồng CPU để nạp dữ liệu")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn tới file .pt để chạy tiếp")
    args = parser.parse_args()

    # Khởi tạo Config chung
    config = Config(
        data=DataConfig(
            data_root=os.path.abspath(args.data_root),
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            patches_per_epoch=args.patches_per_epoch,
            num_input_images=args.num_images,
            num_workers=args.num_workers,
        ),
        model=ModelConfig(model_type=args.model_type),
        train=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
            optimizer=args.optimizer,
        ),
        device=args.device,
    )

    if args.mode == "synapse":
        train_synapse(config, args)
    elif args.mode == "synthetic":
        # CHẠY LUỒNG MỚI
        train_synthetic_pipeline(config, args)
    else:
        # CÁC LUỒNG CŨ (Photometric Stereo cơ bản)
        if args.mode == "logo_cv":
            logo_cv(config)
        elif args.mode == "train":
            all_objs = config.data.objects
            train_objs = args.train_objects or all_objs[:-1]
            test_obj = args.test_object or all_objs[-1]
            train_fold(train_objs, test_obj, config, fold_name="train")

if __name__ == "__main__":
    main()