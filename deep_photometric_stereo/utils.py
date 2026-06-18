"""
Utility functions for UNetPS.
"""
import os
import numpy as np
import torch


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
    """
    Loads model weights and handles backward compatibility for old checkpoints.
    Performs on-the-fly tensor surgery for channel and shape mismatches.
    """
    import torch
    import torch.nn as nn
    print(f"=> Loading checkpoint '{path}'")
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # 1. Rename keys for Up-sampling blocks (Decoder)
        if k.startswith("up") and "conv.block." in k:
            new_k = k.replace("conv.block.", "conv.")
            
        # 2. Rename keys for Head block
        elif k == "head.weight":
            new_k = "head.conv.weight"
        elif k == "head.bias":
            new_k = "head.conv.bias"
            
        new_state_dict[new_k] = v

    # =================================================================
    # TENSOR SURGERY (Upgrade old Head to new Head)
    # =================================================================
    if "head.conv.weight" in new_state_dict:
        hw = new_state_dict["head.conv.weight"]
        # If old matrix shape is [3, 32, 1, 1], upgrade to [4, 32, 3, 3]
        if hw.shape == torch.Size([3, 32, 1, 1]):
            print("[ADAPTER] Upgrading head.conv.weight from 1x1 (3 ch) to 3x3 (4 ch)")
            # Create a new matrix initialized with zeros
            new_hw = torch.zeros(4, 32, 3, 3, dtype=hw.dtype, device=hw.device)
            # Place the old 1x1 matrix at the center of the new 3x3 matrix
            new_hw[:3, :, 1:2, 1:2] = hw
            new_state_dict["head.conv.weight"] = new_hw
            
    if "head.conv.bias" in new_state_dict:
        hb = new_state_dict["head.conv.bias"]
        # If old bias shape is [3], upgrade to [4]
        if hb.shape == torch.Size([3]):
            print("[ADAPTER] Upgrading head.conv.bias from 3 to 4")
            new_hb = torch.zeros(4, dtype=hb.dtype, device=hb.device)
            new_hb[:3] = hb
            # Set Mask channel bias = 5.0 (Sigmoid(5) ~ 0.99 -> Assume whole image is foreground)
            new_hb[3] = 5.0
            new_state_dict["head.conv.bias"] = new_hb

    # --- NEW SURGERY FOR DECOUPLED HEAD COMPATIBILITY ---
    legacy_head = False
    if "head.conv.weight" in new_state_dict:
        print("[ADAPTER] Detected legacy coupled MultiTaskHead. Patching model architecture on-the-fly...")
        legacy_head = True
        
        hw = new_state_dict["head.conv.weight"]
        out_ch, in_ch, kH, kW = hw.shape
        
        # Dynamically add the old conv layer to the new model (Will be moved to device on first pass)
        model.head.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kH, padding=kH//2)
        
        # Override the forward method of the head to bypass normal_head and mask_head
        def legacy_forward(x):
            # CẬP NHẬT: Tự động di chuyển lớp Conv cũ sang đúng Device (MPS/CUDA) của ảnh đầu vào
            if next(model.head.conv.parameters()).device != x.device:
                model.head.conv = model.head.conv.to(x.device)
            return model.head.conv(x)
        
        model.head.forward = legacy_forward
    # =================================================================

    # Load the updated state_dict into the model
    # strict=False avoids crashing on missing "normal_head" / "mask_head" keys for old checkpoints
    model.load_state_dict(new_state_dict, strict=not legacy_head)
    
    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", 0.0)
    
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"=> Loaded optimizer state")
        
    return epoch, val_loss

def detect_model_type(path):
    """Detect model_type from a checkpoint file.

    Returns 'transunet', 'lightweight', 'globalattn' (CNN encoder + wide
    global-attention bottleneck), or 'swin' (genuine Swin Transformer backbone).

    The state_dict structure is checked first for the two attention models because
    'swin' is historically ambiguous: legacy checkpoints saved the global-attention
    model under model_type='swin', so the saved string alone cannot be trusted.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt.get("model_state_dict", {}).keys())

    # 1) Structural detection (ground truth, disambiguates the legacy 'swin' label)
    if any(k.startswith("stem1.") for k in keys) and any(k.startswith("encoder.layers.") for k in keys):
        return "swin"          # genuine Swin backbone (SwinBackboneUNetPS)
    if any(k.startswith("patch_proj.") for k in keys) and any(k.startswith("pos_row.") for k in keys):
        return "globalattn"    # global-attention bottleneck (formerly mislabelled 'swin')

    # 2) Saved model_type field
    saved = ckpt.get("model_type", "")
    if "SwinBackbone" in saved or saved == "swin" or saved == "swin_real":
        return "swin"
    if "GlobalAttn" in saved or "SwinUNetPS" in saved or saved == "globalattn":
        return "globalattn"
    if "Lightweight" in saved or saved == "lightweight":
        return "lightweight"
    if "TransUNet" in saved or saved == "transunet":
        return "transunet"

    # 3) Remaining structural fallbacks
    if any(k.startswith("enc1.") for k in keys):
        return "lightweight"
    if any(k.startswith("encoder.enc1.") for k in keys):
        return "transunet"
    return "transunet"  # default


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)