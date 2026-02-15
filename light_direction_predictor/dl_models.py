"""
Deep Learning Models for Light Direction Prediction (v2 - Improved)
====================================================================
PyTorch-based CNN models for predicting light direction from images.

Models:
1. LightCNN - Custom lightweight CNN designed for light estimation
2. ResNetLight - Transfer learning with ResNet18
3. EfficientNetLight - Transfer learning with EfficientNet-B0

Key Improvements over v1:
- 2-phase training for transfer learning (freeze backbone → fine-tune)
- Early stopping within CV folds
- Larger image size (224) for pretrained models
- Gradient clipping for stable training
- Stronger data augmentation
- Angular loss function
- OneCycleLR scheduler
"""

import numpy as np
import os
import glob
from PIL import Image
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

    # ==================== DATASET ====================

    class LightDirectionDataset(Dataset):
        """PyTorch Dataset for light direction prediction."""

        def __init__(self, img_paths, light_directions, transform=None, img_size=224):
            self.img_paths = img_paths
            self.light_directions = torch.tensor(light_directions, dtype=torch.float32)
            self.transform = transform
            self.img_size = img_size

            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485], std=[0.229])
                ])

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = Image.open(self.img_paths[idx]).convert('L')
            if self.transform:
                img = self.transform(img)
            light = self.light_directions[idx]
            return img, light


    # ==================== DATA AUGMENTATION ====================

    def get_train_transforms(img_size=224):
        """Stronger data augmentation for training."""
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
            ], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])


    def get_val_transforms(img_size=224):
        """Validation transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])


    # ==================== CUSTOM CNN MODEL ====================

    class LightCNN(nn.Module):
        """Custom lightweight CNN for light direction prediction."""

        def __init__(self, img_size=224, dropout=0.4):
            super(LightCNN, self).__init__()
            self.img_size = img_size

            self.features = nn.Sequential(
                self._conv_block(1, 32, dropout * 0.5),
                self._conv_block(32, 64, dropout * 0.5),
                self._conv_block(64, 128, dropout * 0.75),
                self._conv_block(128, 256, dropout * 0.75),
                self._conv_block(256, 512, dropout),
            )

            self.global_pool = nn.AdaptiveAvgPool2d(1)

            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
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
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x


    # ==================== RESNET TRANSFER LEARNING ====================

    class ResNetLight(nn.Module):
        """ResNet18-based model with proper transfer learning."""

        def __init__(self, pretrained=True, dropout=0.4):
            super(ResNetLight, self).__init__()

            self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

            # Grayscale input
            original_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    self.resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

            # Better head with BatchNorm
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, 3)
            )

        def get_backbone_params(self):
            """Get backbone parameters (for differential LR)."""
            backbone = [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1,
                        self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
            params = []
            for module in backbone:
                params.extend(module.parameters())
            return params

        def get_head_params(self):
            """Get head parameters."""
            return self.resnet.fc.parameters()

        def freeze_backbone(self):
            """Freeze all backbone layers."""
            for param in self.get_backbone_params():
                param.requires_grad = False

        def unfreeze_backbone(self):
            """Unfreeze all backbone layers."""
            for param in self.get_backbone_params():
                param.requires_grad = True

        def forward(self, x):
            x = self.resnet(x)
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x


    # ==================== EFFICIENTNET TRANSFER LEARNING ====================

    class EfficientNetLight(nn.Module):
        """EfficientNet-B0 based model with proper transfer learning."""

        def __init__(self, pretrained=True, dropout=0.4):
            super(EfficientNetLight, self).__init__()

            self.efficientnet = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None
            )

            # Grayscale input
            original_conv = self.efficientnet.features[0][0]
            self.efficientnet.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    self.efficientnet.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

            # Better classifier head
            num_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, 3)
            )

        def get_backbone_params(self):
            """Get backbone (features) parameters."""
            return self.efficientnet.features.parameters()

        def get_head_params(self):
            """Get head (classifier) parameters."""
            return self.efficientnet.classifier.parameters()

        def freeze_backbone(self):
            for param in self.get_backbone_params():
                param.requires_grad = False

        def unfreeze_backbone(self):
            for param in self.get_backbone_params():
                param.requires_grad = True

        def forward(self, x):
            x = self.efficientnet(x)
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x


    # ==================== LOSS FUNCTIONS ====================

    class AngularLoss(nn.Module):
        """
        Direct angular loss - minimizes the angle between predicted and target.
        More meaningful than cosine loss for direction prediction.
        """

        def __init__(self):
            super(AngularLoss, self).__init__()

        def forward(self, pred, target):
            # Normalize
            pred_n = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
            target_n = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
            # Cosine similarity
            cos_sim = torch.clamp(torch.sum(pred_n * target_n, dim=1), -1 + 1e-7, 1 - 1e-7)
            # Angular error in radians
            angles = torch.acos(cos_sim)
            return angles.mean()


    class CombinedLoss(nn.Module):
        """Combined: Angular loss + MSE for stability."""

        def __init__(self, angular_weight=0.7, mse_weight=0.3):
            super(CombinedLoss, self).__init__()
            self.angular_weight = angular_weight
            self.mse_weight = mse_weight
            self.angular = AngularLoss()
            self.mse = nn.MSELoss()

        def forward(self, pred, target):
            return self.angular_weight * self.angular(pred, target) + \
                   self.mse_weight * self.mse(pred, target)


    # ==================== TRAINING UTILITIES ====================

    class EarlyStopping:
        """Early stopping to prevent overfitting."""

        def __init__(self, patience=15, min_delta=0.0005, restore_best=True):
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best = restore_best
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
            if self.restore_best and self.best_state is not None:
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
        Improved Deep Learning Light Direction Predictor.

        Key improvements:
        - 2-phase training for transfer learning
        - Early stopping in CV folds
        - Gradient clipping
        - Larger image size (224) for pretrained models
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
                return LightCNN(img_size=self.img_size, dropout=0.4)
            elif self.model_type == 'resnet':
                return ResNetLight(pretrained=True, dropout=0.4)
            elif self.model_type == 'efficientnet':
                return EfficientNetLight(pretrained=True, dropout=0.4)
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
                             grad_clip=1.0, verbose=True, phase_name=""):
            """Train for one phase with early stopping."""
            history = {'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []}

            for epoch in range(epochs):
                # Training
                model.train()
                train_losses, train_preds, train_targets = [], [], []

                for images, targets in train_loader:
                    images, targets = images.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Gradient clipping
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
                train_error = compute_angular_error(np.array(train_preds), np.array(train_targets)).mean()
                val_error = compute_angular_error(np.array(val_preds), np.array(val_targets)).mean()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_error'].append(train_error)
                history['val_error'].append(val_error)

                if scheduler:
                    scheduler.step()

                early_stopping(val_loss, model)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"  {phase_name} Epoch {epoch+1:3d}/{epochs} | "
                          f"Train: {train_error:.1f}° | Val: {val_error:.1f}°")

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
            Train with 2-phase strategy for transfer learning models.

            Phase 1: Freeze backbone, train head (fast convergence)
            Phase 2: Unfreeze backbone, fine-tune all (better accuracy)
            """
            self.model = self._create_model().to(self.device)

            # Split data
            n_samples = len(img_paths)
            indices = np.random.permutation(n_samples)
            n_val = int(n_samples * val_split)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            train_paths = [img_paths[i] for i in train_indices]
            train_lights = light_directions[train_indices]
            val_paths = [img_paths[i] for i in val_indices]
            val_lights = light_directions[val_indices]

            # Create datasets
            train_transform = get_train_transforms(self.img_size) if use_augmentation else get_val_transforms(self.img_size)
            val_transform = get_val_transforms(self.img_size)

            train_dataset = LightDirectionDataset(train_paths, train_lights, train_transform, self.img_size)
            val_dataset = LightDirectionDataset(val_paths, val_lights, val_transform, self.img_size)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            criterion = CombinedLoss(angular_weight=0.7, mse_weight=0.3)

            if verbose:
                print(f"\nTraining {self.model_type.upper()} model...")
                print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Image size: {self.img_size}")
                print("-" * 60)

            if self._is_transfer_learning():
                # ---- PHASE 1: Freeze backbone, train head ----
                if verbose:
                    print("Phase 1: Training head (backbone frozen)...")

                self.model.freeze_backbone()
                head_params = list(self.model.get_head_params())
                optimizer = optim.AdamW(head_params, lr=lr * 10, weight_decay=1e-4)
                phase1_epochs = min(30, epochs // 3)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=lr)
                early_stop = EarlyStopping(patience=10, min_delta=0.001)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, phase1_epochs, early_stop,
                                      verbose=verbose, phase_name="P1")

                # ---- PHASE 2: Unfreeze and fine-tune all ----
                if verbose:
                    print("\nPhase 2: Fine-tuning all layers...")

                self.model.unfreeze_backbone()
                backbone_params = list(self.model.get_backbone_params())

                # Differential learning rates: backbone gets lower LR
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': lr * 0.1},
                    {'params': head_params, 'lr': lr},
                ], weight_decay=1e-4)

                phase2_epochs = epochs - phase1_epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
                early_stop = EarlyStopping(patience=20, min_delta=0.0005)

                history = self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                                optimizer, scheduler, phase2_epochs, early_stop,
                                                verbose=verbose, phase_name="P2")
            else:
                # CNN: Single-phase training
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                early_stop = EarlyStopping(patience=25, min_delta=0.0005)

                history = self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                                optimizer, scheduler, epochs, early_stop,
                                                verbose=verbose, phase_name="")

            self.is_fitted = True

            if verbose:
                print("-" * 60)
                print(f"Best validation error: {min(history['val_error']):.2f}°")

            return history

        def _train_for_cv(self, train_paths, train_lights, epochs, batch_size, lr):
            """
            Train a model for one CV fold with internal validation + early stopping.
            Uses 2-phase training for transfer learning models.
            """
            self.model = self._create_model().to(self.device)

            # Split train into train/val (use 15% for validation within the fold)
            n = len(train_paths)
            indices = np.random.permutation(n)
            n_val = max(int(n * 0.12), 10)
            val_idx, train_idx = indices[:n_val], indices[n_val:]

            t_paths = [train_paths[i] for i in train_idx]
            t_lights = train_lights[train_idx]
            v_paths = [train_paths[i] for i in val_idx]
            v_lights = train_lights[val_idx]

            train_transform = get_train_transforms(self.img_size)
            val_transform = get_val_transforms(self.img_size)

            train_dataset = LightDirectionDataset(t_paths, t_lights, train_transform, self.img_size)
            val_dataset = LightDirectionDataset(v_paths, v_lights, val_transform, self.img_size)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            criterion = CombinedLoss(angular_weight=0.7, mse_weight=0.3)

            if self._is_transfer_learning():
                # Phase 1: Head only
                self.model.freeze_backbone()
                head_params = list(self.model.get_head_params())
                optimizer = optim.AdamW(head_params, lr=lr * 10, weight_decay=1e-4)
                phase1_epochs = min(20, epochs // 3)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=lr)
                early_stop = EarlyStopping(patience=8, min_delta=0.001)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, phase1_epochs, early_stop,
                                      verbose=False)

                # Phase 2: Fine-tune all
                self.model.unfreeze_backbone()
                backbone_params = list(self.model.get_backbone_params())
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': lr * 0.1},
                    {'params': head_params, 'lr': lr},
                ], weight_decay=1e-4)
                phase2_epochs = epochs - phase1_epochs
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)
                early_stop = EarlyStopping(patience=15, min_delta=0.0005)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, phase2_epochs, early_stop,
                                      verbose=False)
            else:
                # CNN single phase
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
                early_stop = EarlyStopping(patience=20, min_delta=0.0005)

                self._train_one_phase(self.model, train_loader, val_loader, criterion,
                                      optimizer, scheduler, epochs, early_stop,
                                      verbose=False)

        def cross_validate(self, img_paths, light_directions, groups,
                          epochs=80, batch_size=32, lr=1e-4, verbose=True):
            """
            Leave-One-Object-Out cross-validation with proper training per fold.
            """
            unique_groups = np.unique(groups)
            all_errors = np.zeros(len(img_paths))
            per_object_errors = {}

            if verbose:
                print(f"\nLeave-One-Object-Out Cross-validation ({self.model_type.upper()})")
                print(f"Image size: {self.img_size}, Epochs/fold: {epochs}")
                print("=" * 60)

            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask

                train_paths = [img_paths[i] for i in range(len(img_paths)) if train_mask[i]]
                train_lights = light_directions[train_mask]
                test_paths = [img_paths[i] for i in range(len(img_paths)) if test_mask[i]]
                test_lights = light_directions[test_mask]

                if verbose:
                    print(f"\nFold {test_group}: Testing on {len(test_paths)} images, training on {len(train_paths)}...")

                # Train with proper 2-phase training + early stopping
                self._train_for_cv(train_paths, train_lights, epochs, batch_size, lr)

                # Evaluate on test set
                test_transform = get_val_transforms(self.img_size)
                test_dataset = LightDirectionDataset(test_paths, test_lights, test_transform, self.img_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                self.model.eval()
                test_preds = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.to(self.device)
                        outputs = self.model(images)
                        test_preds.extend(outputs.cpu().numpy())

                test_preds = np.array(test_preds)
                errors = compute_angular_error(test_preds, test_lights)
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

            dummy_lights = np.zeros((len(img_paths), 3), dtype=np.float32)
            transform = get_val_transforms(self.img_size)
            dataset = LightDirectionDataset(img_paths, dummy_lights, transform, self.img_size)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            self.model.eval()
            predictions = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)
                    outputs = self.model(images)
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
        print("PyTorch not available. Cannot compare DL models.")
        return {}

    print("=" * 60)
    print("COMPARING DEEP LEARNING MODELS")
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
    print("DEEP LEARNING MODELS SUMMARY")
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
