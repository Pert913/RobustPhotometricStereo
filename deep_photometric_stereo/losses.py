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
        pred = F.normalize(pred, dim=1, eps=1e-8)
        target = F.normalize(target, dim=1, eps=1e-8)
        mask = mask.to(pred.dtype)  # Safe type casting in case mask is boolean

        # Cosine similarity per pixel
        cos_sim = (pred * target).sum(dim=1, keepdim=True)
        cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Angular error in radians
        angular_err = torch.acos(cos_sim)

        # Apply mask
        masked_err = angular_err * mask
        n_valid = mask.sum().clamp(min=1.0)

        return masked_err.sum() / n_valid


class MaskedCosineLoss(nn.Module):
    """1 - cosine similarity loss over masked pixels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=1, eps=1e-8)
        target = F.normalize(target, dim=1, eps=1e-8)
        mask = mask.to(pred.dtype)

        cos_sim = (pred * target).sum(dim=1, keepdim=True)
        loss = (1.0 - cos_sim) * mask
        n_valid = mask.sum().clamp(min=1.0)

        return loss.sum() / n_valid


class MaskedL1Loss(nn.Module):
    """L1 loss on normal vectors over masked pixels."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=1, eps=1e-8)
        target = F.normalize(target, dim=1, eps=1e-8)
        mask = mask.to(pred.dtype)

        l1 = torch.abs(pred - target).sum(dim=1, keepdim=True)
        masked_l1 = l1 * mask
        n_valid = mask.sum().clamp(min=1.0)

        return masked_l1.sum() / n_valid


class MaskedGradientLoss(nn.Module):
    """
    Computes L1 loss on the spatial gradients of the normal maps.
    This encourages the model to preserve sharp edges and high-frequency details.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels for x and y directions
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Reshape to (out_channels, in_channels, kH, kW)
        # Applied to each of the 3 normal channels independently
        self.register_buffer('weight_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('weight_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=1, eps=1e-8)
        target = F.normalize(target, dim=1, eps=1e-8)
        mask = mask.to(pred.dtype)

        # Pad to keep spatial dimensions consistent
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        target_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')

        weight_x = self.weight_x.to(pred_pad.device)
        weight_y = self.weight_y.to(pred_pad.device)

        # Compute gradients using grouped convolution (groups=3)
        pred_grad_x = F.conv2d(pred_pad, weight_x, groups=3)
        pred_grad_y = F.conv2d(pred_pad, weight_y, groups=3)
        
        target_grad_x = F.conv2d(target_pad, self.weight_x, groups=3)
        target_grad_y = F.conv2d(target_pad, self.weight_y, groups=3)

        # Compute L1 difference on gradients
        loss_grad_x = torch.abs(pred_grad_x - target_grad_x).sum(dim=1, keepdim=True)
        loss_grad_y = torch.abs(pred_grad_y - target_grad_y).sum(dim=1, keepdim=True)
        
        loss_grad = loss_grad_x + loss_grad_y

        # Apply mask (erode mask slightly to avoid boundary artifacts)
        mask_eroded = F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
        mask_eroded = 1.0 - mask_eroded
        
        masked_loss = loss_grad * mask_eroded
        n_valid = mask_eroded.sum().clamp(min=1.0)

        return masked_loss.sum() / n_valid


class CombinedNormalLoss(nn.Module):
    """Weighted combination of angular, cosine, L1, and Gradient losses."""

    def __init__(self, angular_w=0.5, cosine_w=0.2, l1_w=0.1, grad_w=0.2):
        super().__init__()
        self.angular_loss = MaskedAngularLoss()
        self.cosine_loss = MaskedCosineLoss()
        self.l1_loss = MaskedL1Loss()
        self.grad_loss = MaskedGradientLoss()
        
        self.angular_w = angular_w
        self.cosine_w = cosine_w
        self.l1_w = l1_w
        self.grad_w = grad_w

    def forward(self, pred, target, mask):
        loss_a = self.angular_loss(pred, target, mask)
        loss_c = self.cosine_loss(pred, target, mask)
        loss_l = self.l1_loss(pred, target, mask)
        loss_g = self.grad_loss(pred, target, mask)
        
        total_loss = (self.angular_w * loss_a + 
                      self.cosine_w * loss_c + 
                      self.l1_w * loss_l + 
                      self.grad_w * loss_g)
        return total_loss


# ==============================================================================
# SEGMENTATION LOSSES
# ==============================================================================

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    Optimized: Replaced loops with vectorized tensor operations.
    """

    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes, H, W) raw logits
            targets: (B, H, W) or (B, 1, H, W) integer class labels
        """
        if targets.dim() == 4:  # Handle cases where target has an extra channel dimension
            targets = targets.squeeze(1)
            
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, num_classes, H, W)
        targets_onehot = F.one_hot(targets.long(), self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        # Vectorized per-class dice (eliminating the for loop)
        intersect = (inputs * targets_onehot).sum(dim=(0, 2, 3))
        z_sum = inputs.sum(dim=(0, 2, 3))
        y_sum = targets_onehot.sum(dim=(0, 2, 3))
        
        dice = (2.0 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)
        
        return (1.0 - dice).mean()


class SegmentationLoss(nn.Module):
    """
    Combined CE + Dice loss.
    loss = ce_weight * CrossEntropy + dice_weight * Dice
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
            targets: (B, H, W) or (B, 1, H, W) integer class labels
        """
        if targets.dim() == 4:  # Handle extra channel dimension to avoid CrossEntropy error
            targets = targets.squeeze(1)
            
        loss_ce = self.ce_loss(inputs, targets.long())
        loss_dice = self.dice_loss(inputs, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
    

class MultiTaskLoss(nn.Module):
    """
    Primary Multi-Task Loss optimizing both Normal Estimation and Mask Segmentation.
    """
    def __init__(self, normal_loss_module, seg_weight=0.5):
        super().__init__()
        self.normal_loss = normal_loss_module
        self.seg_loss = nn.BCEWithLogitsLoss() 
        self.seg_weight = seg_weight

    def forward(self, pred_4c, target_normal, target_mask):
        pred_normal = pred_4c[:, :3, :, :]
        pred_mask_logit = pred_4c[:, 3:, :, :]

        loss_normal = self.normal_loss(pred_normal, target_normal, target_mask)
        loss_seg = self.seg_loss(pred_mask_logit, target_mask.float())

        total_loss = loss_normal + (self.seg_weight * loss_seg)
        
        # Return all three values for logging and charting
        return total_loss, loss_normal, loss_seg