"""
Deep Learning Models for Light Direction Prediction (v3)
========================================================
PyTorch-based CNN models for predicting light direction from images.

Models:
1. LightCNN - Custom CNN with gradient input
2. ResNetLight - Transfer learning with ResNet18
3. EfficientNetLight - Transfer learning with EfficientNet-B0

Key v3 Innovations (target: <15 degree CV error):
1. 3-CHANNEL GRADIENT INPUT: [normalized_img, sobel_x, sobel_y]
   - Object-invariant: focuses on shading patterns, not object identity
   - Natural 3-channel for pretrained models (no conv1 hack needed)
   - Gradients directly encode light direction info (Lambertian shading)

2. MIXUP AUGMENTATION: Blends images from different objects
   - Prevents memorizing specific object appearances
   - Physically motivated: blending illuminations ~ intermediate lighting

3. HORIZONTAL FLIP WITH LABEL CORRECTION:
   - Doubles effective training data
   - Correctly negates x-component of light direction

4. 2-PHASE TRANSFER LEARNING + EARLY STOPPING IN CV FOLDS
5. ANGULAR LOSS + GRADIENT CLIPPING
6. HIGHER REGULARIZATION (dropout=0.5, weight_decay=5e-4)
"""

import numpy as np
import os
import glob
import random
from PIL import Image, ImageOps
from scipy import ndimage
import warnings

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Deep learning models will not be available. "
                  "Install with: pip install torch torchvision")

if TORCH_AVAILABLE:

    # ==================== DATASET (3-CHANNEL GRADIENT INPUT) ====================

    class LightDirectionDataset(Dataset):
        """
        3-channel gradient-based dataset for light direction prediction.

        Input channels: [normalized_image, sobel_x, sobel_y]

        Why gradient input?
        - ML models achieve ~10-15 deg error using gradient-based features
        - Raw pixel DL models get ~24 deg because they learn object identity
        - Gradient channels make the input object-invariant
        - Pretrained models expect 3 channels - no conv1 modification needed
        """

        def __init__(self, img_paths, light_directions, img_size=224, augment=False):
            self.img_paths = img_paths
            self.light_directions = torch.tensor(light_directions, dtype=torch.float32)
            self.img_size = img_size
            self.augment = augment

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert('L')
            light = self.light_directions[idx].clone()

            if self.augment:
                # Resize larger for random crop
                pad = 32
                img = img.resize((self.img_size + pad, self.img_size + pad), Image.BILINEAR)

                # Random crop
                i = random.randint(0, pad)
                j = random.randint(0, pad)
                img = img.crop((j, i, j + self.img_size, i + self.img_size))

                # Random horizontal flip WITH label correction
                if random.random() > 0.5:
                    img = ImageOps.mirror(img)
                    light[0] = -light[0]  # Negate x-component

                # Brightness/contrast jitter (invariant after normalization)
                img_arr = np.array(img, dtype=np.float32)
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                mean_val = img_arr.mean()
                img_arr = (img_arr - mean_val) * contrast + mean_val
                img_arr = img_arr * brightness
                img_arr = np.clip(img_arr, 0, 255)
            else:
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                img_arr = np.array(img, dtype=np.float32)

            # Per-image normalization (zero mean, unit std)
            img_norm = (img_arr - img_arr.mean()) / (img_arr.std() + 1e-8)

            # Compute Sobel gradients (on normalized image)
            grad_x = ndimage.sobel(img_norm, axis=1)  # horizontal edges
            grad_y = ndimage.sobel(img_norm, axis=0)  # vertical edges

            # Normalize gradients to [-1, 1]
            max_grad = max(np.abs(grad_x).max(), np.abs(grad_y).max(), 1e-8)
            grad_x = grad_x / max_grad
            grad_y = grad_y / max_grad

            # Clip normalized image to [-1, 1]
            img_norm = np.clip(img_norm, -3, 3) / 3.0

            # Stack as 3 channels: [image, gradient_x, gradient_y]
            tensor = torch.tensor(
                np.stack([img_norm, grad_x, grad_y]),
                dtype=torch.float32
            )

            # Random erasing (prevents learning object shape)
            if self.augment and random.random() > 0.7:
                h, w = self.img_size, self.img_size
                rh = random.randint(h // 10, h // 4)
                rw = random.randint(w // 10, w // 4)
                ry = random.randint(0, h - rh)
                rx = random.randint(0, w - rw)
                tensor[:, ry:ry+rh, rx:rx+rw] = 0

            return tensor, light


    # ==================== MIXUP AUGMENTATION ====================

    def mixup_data(x, y, alpha=0.4):
        """
        Mixup: blend images and labels from different samples.

        For light direction: interpolates between direction vectors.
        Physically motivated - blending illuminations approximates
        intermediate lighting for Lambertian surfaces.
        """
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        # Normalize direction vectors
        mixed_y = mixed_y / (torch.norm(mixed_y, dim=1, keepdim=True) + 1e-8)

        return mixed_x, mixed_y


    # ==================== CUSTOM CNN (3-CHANNEL INPUT) ====================

    class LightCNN(nn.Module):
        """Custom CNN with 3-channel gradient input."""

        def __init__(self, img_size=224, dropout=0.5):
            super(LightCNN, self).__init__()
            self.img_size = img_size

            # 3-channel input: [image, grad_x, grad_y]
            self.features = nn.Sequential(
                self._conv_block(3, 32, dropout * 0.5),
                self._conv_block(32, 64, dropout * 0.5),
                self._conv_block(64, 128, dropout * 0.75),
                self._conv_block(128, 256, dropout),
            )
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3)
            )

        def _conv_block(self, in_ch, out_ch, dropout):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)


    # ==================== RESNET (3-CHANNEL - NO CONV1 HACK) ====================

    class ResNetLight(nn.Module):
        """
        ResNet18 with 3-channel gradient input.

        No need to modify conv1 - input is already 3 channels!
        Pretrained ImageNet weights provide excellent initialization
        since the first conv filters detect edges/gradients naturally.
        """

        def __init__(self, pretrained=True, dropout=0.5):
            super(ResNetLight, self).__init__()
            self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

            # NO conv1 modification needed - 3 channels in, 3 channels expected

            # Better head with BatchNorm
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                nn.Linear(num_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 3)
            )

        def get_backbone_params(self):
            backbone = [self.resnet.conv1, self.resnet.bn1,
                        self.resnet.layer1, self.resnet.layer2,
                        self.resnet.layer3, self.resnet.layer4]
            params = []
            for module in backbone:
                params.extend(module.parameters())
            return params

        def get_head_params(self):
            return list(self.resnet.fc.parameters())

        def freeze_backbone(self):
            """Freeze deep layers but keep conv1+bn1 trainable (new input format)."""
            for layer in [self.resnet.layer1, self.resnet.layer2,
                          self.resnet.layer3, self.resnet.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False

        def unfreeze_backbone(self):
            for param in self.get_backbone_params():
                param.requires_grad = True

        def forward(self, x):
            x = self.resnet(x)
            return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)


    # ==================== EFFICIENTNET (3-CHANNEL - NO CONV1 HACK) ====================

    class EfficientNetLight(nn.Module):
        """EfficientNet-B0 with 3-channel gradient input."""

        def __init__(self, pretrained=True, dropout=0.5):
            super(EfficientNetLight, self).__init__()
            self.efficientnet = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None
            )

            # NO first conv modification needed - 3 channels expected

            # Better classifier head
            num_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                nn.Linear(num_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 3)
            )

        def get_backbone_params(self):
            return list(self.efficientnet.features.parameters())

        def get_head_params(self):
            return list(self.efficientnet.classifier.parameters())

        def freeze_backbone(self):
            """Freeze features except first conv (needs adaptation to gradient input)."""
            for i, block in enumerate(self.efficientnet.features):
                if i == 0:
                    continue  # Keep first conv trainable
                for param in block.parameters():
                    param.requires_grad = False

        def unfreeze_backbone(self):
            for param in self.efficientnet.features.parameters():
                param.requires_grad = True

        def forward(self, x):
            x = self.efficientnet(x)
            return x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)


    # ==================== LOSS FUNCTIONS ====================

    class AngularLoss(nn.Module):
        """Direct angular loss - minimizes angle in radians."""

        def forward(self, pred, target):
            pred_n = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
            target_n = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
            cos_sim = torch.clamp(torch.sum(pred_n * target_n, dim=1), -1 + 1e-7, 1 - 1e-7)
            return torch.acos(cos_sim).mean()


    class CombinedLoss(nn.Module):
        """Angular + MSE combined loss."""

        def __init__(self, angular_weight=0.7, mse_weight=0.3):
            super(CombinedLoss, self).__init__()
            self.aw = angular_weight
            self.mw = mse_weight
            self.angular = AngularLoss()
            self.mse = nn.MSELoss()

        def forward(self, pred, target):
            return self.aw * self.angular(pred, target) + self.mw * self.mse(pred, target)


    # ==================== TRAINING UTILITIES ====================

    class EarlyStopping:
        """Early stopping to prevent overfitting."""

        def __init__(self, patience=15, min_delta=0.0005):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.best_state = None
            self.early_stop = False

        def __call__(self, val_loss, model):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                self.counter = 0

        def reset(self):
            self.counter = 0
            self.best_loss = None
            self.best_state = None
            self.early_stop = False

        def restore(self, model):
            if self.best_state is not None:
                model.load_state_dict(self.best_state)


    def compute_angular_error(pred, target):
        """Compute angular error in degrees."""
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.clip(np.sum(pred_norm * target_norm, axis=1), -1, 1)
        return np.arccos(cos_sim) * 180 / np.pi


    # ==================== DEEP LEARNING TRAINER ====================

    class DeepLightPredictor:
        """
        Deep Learning Light Direction Predictor (v3).

        Key features:
        - 3-channel gradient input for object-invariant prediction
        - Mixup augmentation to prevent object memorization
        - 2-phase transfer learning with early stopping
        - Angular loss for direct metric optimization
        """

        def __init__(self, model_type='resnet', img_size=224, device=None):
            self.model_type = model_type
            self.img_size = img_size
            self.model = None
            self.is_fitted = False

            if device is None:
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)

            print(f"Using device: {self.device}")

        def _create_model(self):
            if self.model_type == 'cnn':
                return LightCNN(img_size=self.img_size, dropout=0.5)
            elif self.model_type == 'resnet':
                return ResNetLight(pretrained=True, dropout=0.5)
            elif self.model_type == 'efficientnet':
                return EfficientNetLight(pretrained=True, dropout=0.5)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        def _is_transfer_learning(self):
            return self.model_type in ['resnet', 'efficientnet']

        def load_data(self, training_folder):
            all_img_paths, all_light_dirs, all_groups = [], [], []
            object_folders = sorted(glob.glob(os.path.join(training_folder, '*PNG')))
            print(f"Found {len(object_folders)} objects for training:")

            for gid, folder in enumerate(object_folders):
                name = os.path.basename(folder)
                light_file = os.path.join(folder, 'light_directions.txt')
                if not os.path.exists(light_file):
                    continue
                lights = np.loadtxt(light_file)
                imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
                imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
                n = min(len(imgs), len(lights))
                imgs, lights = imgs[:n], lights[:n]
                print(f"  {name}: {n} images")
                all_img_paths.extend(imgs)
                all_light_dirs.extend(lights)
                all_groups.extend([gid] * n)

            return all_img_paths, np.array(all_light_dirs, dtype=np.float32), np.array(all_groups)

        def _train_one_phase(self, model, train_loader, val_loader, criterion,
                             optimizer, scheduler, epochs, early_stopping,
                             use_mixup=False, grad_clip=1.0, verbose=True, phase_name=""):
            """Train for one phase with early stopping and optional mixup."""
            history = {'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []}

            for epoch in range(epochs):
                # Training
                model.train()
                train_losses, train_preds, train_targets = [], [], []

                for images, targets in train_loader:
                    images, targets = images.to(self.device), targets.to(self.device)

                    # Mixup augmentation
                    if use_mixup and random.random() > 0.5:
                        images, targets = mixup_data(images, targets, alpha=0.4)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_targets.extend(targets.cpu().numpy())

                # Validation
                model.eval()
                val_losses, val_preds, val_targets = [], [], []

                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_err = compute_angular_error(np.array(train_preds), np.array(train_targets)).mean()
                val_err = compute_angular_error(np.array(val_preds), np.array(val_targets)).mean()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_error'].append(train_err)
                history['val_error'].append(val_err)

                if scheduler:
                    scheduler.step()

                early_stopping(val_loss, model)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"  {phase_name} Epoch {epoch+1:3d}/{epochs} | "
                          f"Train: {train_err:.1f}° | Val: {val_err:.1f}°")

                if early_stopping.early_stop:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

            early_stopping.restore(model)
            model.to(self.device)
            return history

        def train(self, img_paths, light_directions, groups=None,
                  epochs=100, batch_size=32, lr=1e-4, val_split=0.15,
                  use_augmentation=True, verbose=True):
            """
            Train with 2-phase strategy for transfer learning + mixup.

            Phase 1: Freeze deep backbone, train head + conv1 (fast)
            Phase 2: Unfreeze all, fine-tune with differential LR
            """
            self.model = self._create_model().to(self.device)

            # Split data
            n = len(img_paths)
            indices = np.random.permutation(n)
            n_val = int(n * val_split)
            val_idx, train_idx = indices[:n_val], indices[n_val:]

            train_paths = [img_paths[i] for i in train_idx]
            train_lights = light_directions[train_idx]
            val_paths = [img_paths[i] for i in val_idx]
            val_lights = light_directions[val_idx]

            train_ds = LightDirectionDataset(train_paths, train_lights, self.img_size, augment=use_augmentation)
            val_ds = LightDirectionDataset(val_paths, val_lights, self.img_size, augment=False)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            criterion = CombinedLoss(angular_weight=0.7, mse_weight=0.3)

            if verbose:
                print(f"\nTraining {self.model_type.upper()} (3-channel gradient input)...")
                print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Image: {self.img_size}")
                print("-" * 60)

            if self._is_transfer_learning():
                # Phase 1: Freeze deep backbone, train head + conv1
                if verbose:
                    print("Phase 1: Training head + conv1 (deep layers frozen)...")

                self.model.freeze_backbone()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                optimizer = optim.AdamW(trainable, lr=lr * 10, weight_decay=5e-4)
                p1_epochs = min(40, epochs // 3)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1_epochs, eta_min=lr)
                es = EarlyStopping(patience=10)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, p1_epochs, es,
                                      use_mixup=True, verbose=verbose, phase_name="P1")

                # Phase 2: Unfreeze and fine-tune all
                if verbose:
                    print("\nPhase 2: Fine-tuning all layers...")

                self.model.unfreeze_backbone()
                backbone_params = list(self.model.get_backbone_params())
                head_params = list(self.model.get_head_params())

                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': lr * 0.1},
                    {'params': head_params, 'lr': lr},
                ], weight_decay=5e-4)

                p2_epochs = epochs - p1_epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=1e-6)
                es = EarlyStopping(patience=20)

                history = self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                                optimizer, scheduler, p2_epochs, es,
                                                use_mixup=True, verbose=verbose, phase_name="P2")
            else:
                # CNN: Single phase
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                es = EarlyStopping(patience=25)

                history = self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                                optimizer, scheduler, epochs, es,
                                                use_mixup=True, verbose=verbose)

            self.is_fitted = True
            if verbose:
                print("-" * 60)
                print(f"Best validation error: {min(history['val_error']):.2f}°")

            return history

        def _train_for_cv(self, train_paths, train_lights, epochs, batch_size, lr):
            """
            Train for one CV fold with internal val split + early stopping + mixup.
            Uses 2-phase training for transfer learning models.
            """
            self.model = self._create_model().to(self.device)

            # Internal val split
            n = len(train_paths)
            indices = np.random.permutation(n)
            n_val = max(int(n * 0.12), 10)
            val_idx, train_idx = indices[:n_val], indices[n_val:]

            t_paths = [train_paths[i] for i in train_idx]
            t_lights = train_lights[train_idx]
            v_paths = [train_paths[i] for i in val_idx]
            v_lights = train_lights[val_idx]

            train_ds = LightDirectionDataset(t_paths, t_lights, self.img_size, augment=True)
            val_ds = LightDirectionDataset(v_paths, v_lights, self.img_size, augment=False)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            criterion = CombinedLoss()

            if self._is_transfer_learning():
                # Phase 1: Head + conv1 only
                self.model.freeze_backbone()
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                optimizer = optim.AdamW(trainable, lr=lr * 10, weight_decay=5e-4)
                p1_epochs = min(25, epochs // 3)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1_epochs, eta_min=lr)
                es = EarlyStopping(patience=8)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, p1_epochs, es,
                                      use_mixup=True, verbose=False)

                # Phase 2: Fine-tune all
                self.model.unfreeze_backbone()
                backbone_params = list(self.model.get_backbone_params())
                head_params = list(self.model.get_head_params())
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': lr * 0.1},
                    {'params': head_params, 'lr': lr},
                ], weight_decay=5e-4)
                p2_epochs = epochs - p1_epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=1e-6)
                es = EarlyStopping(patience=15)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, p2_epochs, es,
                                      use_mixup=True, verbose=False)
            else:
                # CNN: Single phase
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                es = EarlyStopping(patience=20)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, epochs, es,
                                      use_mixup=True, verbose=False)

        def cross_validate(self, img_paths, light_directions, groups,
                          epochs=80, batch_size=32, lr=1e-4, verbose=True):
            """
            Leave-One-Object-Out cross-validation with proper training per fold.
            Each fold uses 2-phase training + early stopping + mixup.
            """
            unique_groups = np.unique(groups)
            all_errors = np.zeros(len(img_paths))
            per_object_errors = {}

            if verbose:
                print(f"\nLOGO Cross-validation ({self.model_type.upper()}) - 3ch gradient input")
                print(f"Image: {self.img_size}, Epochs/fold: {epochs}")
                print("=" * 60)

            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask

                train_paths = [img_paths[i] for i in range(len(img_paths)) if train_mask[i]]
                train_lights = light_directions[train_mask]
                test_paths = [img_paths[i] for i in range(len(img_paths)) if test_mask[i]]
                test_lights = light_directions[test_mask]

                if verbose:
                    print(f"\nFold {test_group}: test={len(test_paths)}, train={len(train_paths)}...")

                # Train with proper 2-phase + early stopping + mixup
                self._train_for_cv(train_paths, train_lights, epochs, batch_size, lr)

                # Evaluate on test set
                test_ds = LightDirectionDataset(test_paths, test_lights, self.img_size, augment=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

                self.model.eval()
                preds = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        outputs = self.model(images.to(self.device))
                        preds.extend(outputs.cpu().numpy())

                errors = compute_angular_error(np.array(preds), test_lights)
                all_errors[test_mask] = errors
                per_object_errors[test_group] = errors.mean()

                if verbose:
                    print(f"  Object {test_group}: {errors.mean():.2f}° ± {errors.std():.2f}°")

            if verbose:
                print("\n" + "=" * 60)
                print(f"Overall CV error: {all_errors.mean():.2f}° ± {all_errors.std():.2f}°")

            return all_errors, per_object_errors

        def predict(self, img_paths):
            if not self.is_fitted:
                raise ValueError("Model not trained. Call train() first.")

            dummy = np.zeros((len(img_paths), 3), dtype=np.float32)
            dataset = LightDirectionDataset(img_paths, dummy, self.img_size, augment=False)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            self.model.eval()
            predictions = []
            with torch.no_grad():
                for images, _ in loader:
                    outputs = self.model(images.to(self.device))
                    predictions.extend(outputs.cpu().numpy())

            return np.array(predictions)

        def predict_from_folder(self, folder, output_file=None):
            imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
            imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
            print(f"Predicting light directions for {len(imgs)} images...")
            lights = self.predict(imgs)
            if output_file:
                np.savetxt(output_file, lights, fmt='%.4f')
                print(f"Saved to: {output_file}")
            return lights

        def save(self, path):
            if not self.is_fitted:
                raise ValueError("Model not trained.")
            torch.save({
                'model_type': self.model_type,
                'img_size': self.img_size,
                'model_state_dict': self.model.state_dict(),
            }, path)
            print(f"Model saved to: {path}")

        def load(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model_type = checkpoint['model_type']
            self.img_size = checkpoint['img_size']
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.is_fitted = True
            print(f"Model loaded from: {path}")


# ==================== COMPARISON FUNCTION ====================

def compare_dl_models(training_folder, epochs=80, batch_size=32):
    """Compare all deep learning models."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available.")
        return {}

    print("=" * 60)
    print("COMPARING DL MODELS (3-channel gradient input)")
    print("=" * 60)

    results = {}

    for model_type in ['cnn', 'resnet', 'efficientnet']:
        print(f"\n{'=' * 40}")
        print(f"Model: {model_type.upper()}")
        print("=" * 40)

        predictor = DeepLightPredictor(model_type=model_type, img_size=224)
        img_paths, light_directions, groups = predictor.load_data(training_folder)

        errors, per_object = predictor.cross_validate(
            img_paths, light_directions, groups,
            epochs=epochs, batch_size=batch_size, verbose=True
        )

        results[model_type] = errors.mean()

    print("\n" + "=" * 60)
    print("DL MODELS SUMMARY")
    print("=" * 60)
    for m, err in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {m:15s}: {err:.2f}°")

    best = min(results, key=results.get)
    print(f"\nBest DL model: {best}")

    return results


# ==================== MAIN ====================

if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch is required for deep learning models.")
        print("Install with: pip install torch torchvision")
    else:
        TRAINING_FOLDER = './data/training/'
        results = compare_dl_models(TRAINING_FOLDER, epochs=80, batch_size=32)
