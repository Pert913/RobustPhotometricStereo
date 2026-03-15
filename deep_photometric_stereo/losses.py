"""
Loss functions for normal map prediction and segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Segmentation losses (following TransUNet utils.py)

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation (TransUNet utils.py:9-45).
    """

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes, H, W) raw logits
            targets: (B, H, W) integer class labels
        """
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, num_classes, H, W)
        targets_onehot = F.one_hot(targets.long(), self.num_classes)  # (B, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Per-class dice
        total_loss = 0.0
        for c in range(self.num_classes):
            pred_c = inputs[:, c]
            gt_c = targets_onehot[:, c]
            intersect = (pred_c * gt_c).sum()
            z_sum = pred_c.sum()
            y_sum = gt_c.sum()
            dice = (2.0 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
            total_loss += (1.0 - dice)

        return total_loss / self.num_classes


class SegmentationLoss(nn.Module):
    """
    Combined CE + Dice loss (TransUNet trainer.py).
    loss = 0.5 * CrossEntropy + 0.5 * Dice
    """

    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes, H, W) raw logits
            targets: (B, H, W) integer class labels
        """
        loss_ce = self.ce_loss(inputs, targets.long())
        loss_dice = self.dice_loss(inputs, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
