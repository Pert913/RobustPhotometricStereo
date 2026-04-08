"""
Utility functions for UNetPS.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def angular_error_map(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel angular error in degrees.

    Args:
        pred: (H, W, 3) predicted normals (unit vectors)
        gt: (H, W, 3) ground truth normals (unit vectors)
        mask: (H, W) binary mask

    Returns:
        (H, W) angular error in degrees (0 outside mask)
    """
    # Normalize
    pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gt_norm = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)

    cos_sim = np.sum(pred_norm * gt_norm, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    error = np.degrees(np.arccos(cos_sim))

    return error * mask


def mean_angular_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Compute mean angular error (degrees) over masked region."""
    error_map = angular_error_map(pred, gt, mask)
    n_valid = mask.sum()
    if n_valid == 0:
        return 0.0
    return float(error_map.sum() / n_valid)


def normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    """
    Convert normal map to RGB visualization.
    Maps [-1,1] ---> [0,255] for each channel.

    Args:
        normal: (H, W, 3) unit normal vectors

    Returns:
        (H, W, 3) uint8 RGB image
    """
    rgb = ((normal + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def save_checkpoint(model, optimizer, epoch, val_loss, path, model_type=None):
    """Save model checkpoint with model_type for auto-detection on load."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_type": model_type or type(model).__name__,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint. Returns epoch and val_loss."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def detect_model_type(path):
    """Detect model_type from a checkpoint file. Returns 'transunet' or 'lightweight'."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Check saved model_type field
    saved = ckpt.get("model_type", "")
    if "Lightweight" in saved or saved == "lightweight":
        return "lightweight"
    if "TransUNet" in saved or saved == "transunet":
        return "transunet"
    # Fallback: detect from state_dict keys
    keys = list(ckpt.get("model_state_dict", {}).keys())
    if any(k.startswith("enc1.") for k in keys):
        return "lightweight"
    if any(k.startswith("encoder.enc1.") for k in keys):
        return "transunet"
    return "transunet"  # default


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curve(train_losses, val_epochs=None, val_maes=None, save_path="training_curve.png"):
    """
    Vẽ biểu đồ 2 trục (Twin Axes): 
    - Trục trái (Xanh): Training Loss
    - Trục phải (Đỏ): Validation MAE (Đường chữ U)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Vẽ Training Loss (Trục trái)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, linestyle='-', color='#1f77b4', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, color='#1f77b4', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Vẽ Validation MAE (Trục phải) - Nếu có dữ liệu
    if val_epochs and val_maes and len(val_epochs) > 0:
        ax2 = ax1.twinx()  # Tạo trục Y thứ 2 dùng chung trục X
        ax2.plot(val_epochs, val_maes, linestyle='--', marker='o', color='#d62728', linewidth=2, label='Validation MAE')
        ax2.set_ylabel('Validation MAE (Degrees)', fontsize=12, color='#d62728', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Gộp chú thích (Legend) của cả 2 đường vào 1 hộp
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    else:
        ax1.legend(loc='upper right')

    plt.title('Model Convergence: Training Loss vs Validation MAE', fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📉 Đã lưu biểu đồ Đánh giá Training tại: {save_path}")