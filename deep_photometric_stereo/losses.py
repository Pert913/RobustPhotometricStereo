"""
Loss functions for normal map prediction.
"""
import torch
import torch.nn as nn


class MaskedAngularLoss(nn.Module):
    """Angular error loss computed only over masked (foreground) pixels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) predicted normals (should be L2-normalized)
            target: (B, 3, H, W) ground truth normals
            mask: (B, 1, H, W) binary mask
        """
        # Normalize predictions
        pred = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
        target = torch.nn.functional.normalize(target, dim=1, eps=1e-8)

        # Cosine similarity per pixel
        cos_sim = (pred * target).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Angular error in radians
        angular_err = torch.acos(cos_sim)  # (B, 1, H, W)

        # Apply mask
        masked_err = angular_err * mask
        n_valid = mask.sum().clamp(min=1.0)

        return masked_err.sum() / n_valid


class MaskedCosineLoss(nn.Module):
    """1 - cosine similarity loss over masked pixels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
        target = torch.nn.functional.normalize(target, dim=1, eps=1e-8)

        cos_sim = (pred * target).sum(dim=1, keepdim=True)
        loss = (1.0 - cos_sim) * mask
        n_valid = mask.sum().clamp(min=1.0)

        return loss.sum() / n_valid


class MaskedL1Loss(nn.Module):
    """L1 loss on normal vectors over masked pixels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
        target = torch.nn.functional.normalize(target, dim=1, eps=1e-8)

        l1 = torch.abs(pred - target).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        masked_l1 = l1 * mask
        n_valid = mask.sum().clamp(min=1.0)

        return masked_l1.sum() / n_valid


class CombinedNormalLoss(nn.Module):
    """Weighted combination of angular, cosine, and L1 losses."""

    def __init__(self, angular_w=0.6, cosine_w=0.3, l1_w=0.1):
        super().__init__()
        self.angular_loss = MaskedAngularLoss()
        self.cosine_loss = MaskedCosineLoss()
        self.l1_loss = MaskedL1Loss()
        self.angular_w = angular_w
        self.cosine_w = cosine_w
        self.l1_w = l1_w

    def forward(self, pred, target, mask):
        loss_a = self.angular_loss(pred, target, mask)
        loss_c = self.cosine_loss(pred, target, mask)
        loss_l = self.l1_loss(pred, target, mask)
        return self.angular_w * loss_a + self.cosine_w * loss_c + self.l1_w * loss_l
