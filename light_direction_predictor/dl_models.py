"""
Deep Learning Models for Light Direction Prediction
====================================================
PyTorch-based CNN models for predicting light direction from images.

Models:
1. LightCNN - Custom lightweight CNN designed for light estimation
2. ResNetLight - Transfer learning with ResNet18
3. EfficientNetLight - Transfer learning with EfficientNet-B0

Key Features:
- Data augmentation for small datasets
- Learning rate scheduling
- Early stopping
- Cosine similarity loss for direction prediction
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

        def __init__(self, img_paths, light_directions, transform=None, img_size=128):
            """
            Args:
                img_paths: List of image file paths
                light_directions: Numpy array of shape (N, 3)
                transform: Optional torchvision transforms
                img_size: Image size for resizing
            """
            self.img_paths = img_paths
            self.light_directions = torch.tensor(light_directions, dtype=torch.float32)
            self.transform = transform
            self.img_size = img_size

            # Default transform if none provided
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
                ])

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            # Load image
            img = Image.open(self.img_paths[idx]).convert('L')  # Grayscale

            # Apply transforms
            if self.transform:
                img = self.transform(img)

            # Get light direction (already normalized)
            light = self.light_directions[idx]

            return img, light


    # ==================== DATA AUGMENTATION ====================

    def get_train_transforms(img_size=128):
        """
        Data augmentation for training.

        Key augmentations for light direction:
        - Random horizontal flip (must flip light x-component)
        - Random rotation (small angles)
        - Brightness/contrast variations
        - Gaussian noise
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
        ])


    def get_val_transforms(img_size=128):
        """Validation transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])


    # ==================== CUSTOM CNN MODEL ====================

    class LightCNN(nn.Module):
        """
        Custom lightweight CNN for light direction prediction.

        Architecture:
        - 4 conv blocks with batch norm and dropout
        - Global average pooling
        - 2 fully connected layers
        - Output: 3D light direction vector
        """

        def __init__(self, img_size=128, dropout=0.3):
            super(LightCNN, self).__init__()

            self.img_size = img_size

            # Convolutional blocks
            self.conv1 = self._conv_block(1, 32, dropout)      # 128 -> 64
            self.conv2 = self._conv_block(32, 64, dropout)     # 64 -> 32
            self.conv3 = self._conv_block(64, 128, dropout)    # 32 -> 16
            self.conv4 = self._conv_block(128, 256, dropout)   # 16 -> 8
            self.conv5 = self._conv_block(256, 512, dropout)   # 8 -> 4

            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 3)  # Output: 3D light direction
            )

        def _conv_block(self, in_channels, out_channels, dropout):
            """Convolutional block with BatchNorm, ReLU, MaxPool, Dropout."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            )

        def forward(self, x):
            # Convolutional layers
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)

            # Global pooling and FC
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            # Normalize to unit vector
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

            return x


    # ==================== RESNET TRANSFER LEARNING ====================

    class ResNetLight(nn.Module):
        """
        ResNet18-based model for light direction prediction.

        Uses pretrained ImageNet weights and fine-tunes for light estimation.
        Modified first conv layer for grayscale input.
        """

        def __init__(self, pretrained=True, dropout=0.3):
            super(ResNetLight, self).__init__()

            # Load pretrained ResNet18
            self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

            # Modify first conv layer for grayscale (1 channel instead of 3)
            original_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize new conv with average of pretrained weights
            if pretrained:
                with torch.no_grad():
                    self.resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

            # Replace final FC layer
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3)
            )

        def forward(self, x):
            x = self.resnet(x)
            # Normalize to unit vector
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x


    # ==================== EFFICIENTNET TRANSFER LEARNING ====================

    class EfficientNetLight(nn.Module):
        """
        EfficientNet-B0 based model for light direction prediction.

        Efficient architecture with good accuracy/compute tradeoff.
        """

        def __init__(self, pretrained=True, dropout=0.3):
            super(EfficientNetLight, self).__init__()

            # Load pretrained EfficientNet-B0
            self.efficientnet = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None
            )

            # Modify first conv layer for grayscale
            original_conv = self.efficientnet.features[0][0]
            self.efficientnet.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

            # Initialize with average of pretrained weights
            if pretrained:
                with torch.no_grad():
                    self.efficientnet.features[0][0].weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )

            # Replace classifier
            num_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 3)
            )

        def forward(self, x):
            x = self.efficientnet(x)
            # Normalize to unit vector
            x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
            return x


    # ==================== LOSS FUNCTIONS ====================

    class CosineSimilarityLoss(nn.Module):
        """
        Cosine similarity loss for direction prediction.

        Loss = 1 - cos(predicted, target)

        This is more appropriate than MSE for direction vectors.
        """

        def __init__(self):
            super(CosineSimilarityLoss, self).__init__()
            self.cos_sim = nn.CosineSimilarity(dim=1)

        def forward(self, pred, target):
            return 1 - self.cos_sim(pred, target).mean()


    class CombinedLoss(nn.Module):
        """
        Combined loss: Cosine similarity + MSE.

        This helps with both direction accuracy and magnitude stability.
        """

        def __init__(self, cos_weight=0.7, mse_weight=0.3):
            super(CombinedLoss, self).__init__()
            self.cos_weight = cos_weight
            self.mse_weight = mse_weight
            self.cos_sim = nn.CosineSimilarity(dim=1)
            self.mse = nn.MSELoss()

        def forward(self, pred, target):
            cos_loss = 1 - self.cos_sim(pred, target).mean()
            mse_loss = self.mse(pred, target)
            return self.cos_weight * cos_loss + self.mse_weight * mse_loss


    # ==================== TRAINING UTILITIES ====================

    class EarlyStopping:
        """Early stopping to prevent overfitting."""

        def __init__(self, patience=15, min_delta=0.001, restore_best=True):
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

        def restore(self, model):
            if self.restore_best and self.best_state is not None:
                model.load_state_dict(self.best_state)


    def compute_angular_error(pred, target):
        """Compute angular error in degrees."""
        # Normalize predictions
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity
        cos_sim = np.clip(np.sum(pred_norm * target_norm, axis=1), -1, 1)

        # Angular error in degrees
        return np.arccos(cos_sim) * 180 / np.pi


    # ==================== DEEP LEARNING TRAINER ====================

    class DeepLightPredictor:
        """
        Deep Learning-based Light Direction Predictor.

        Supports multiple architectures:
        - 'cnn': Custom LightCNN
        - 'resnet': ResNet18 transfer learning
        - 'efficientnet': EfficientNet-B0 transfer learning
        """

        def __init__(self, model_type='resnet', img_size=128, device=None):
            """
            Args:
                model_type: 'cnn', 'resnet', or 'efficientnet'
                img_size: Input image size
                device: 'cuda', 'mps', 'cpu', or None (auto-detect)
            """
            self.model_type = model_type
            self.img_size = img_size
            self.model = None
            self.is_fitted = False

            # Auto-detect device
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
            """Create the neural network model."""
            if self.model_type == 'cnn':
                return LightCNN(img_size=self.img_size, dropout=0.3)
            elif self.model_type == 'resnet':
                return ResNetLight(pretrained=True, dropout=0.3)
            elif self.model_type == 'efficientnet':
                return EfficientNetLight(pretrained=True, dropout=0.3)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        def load_data(self, training_folder):
            """
            Load training data from folder structure.

            Returns:
                img_paths, light_directions, groups
            """
            all_img_paths, all_light_dirs, all_groups = [], [], []

            object_folders = sorted(glob.glob(os.path.join(training_folder, '*PNG')))
            print(f"Found {len(object_folders)} objects for training:")

            for gid, folder in enumerate(object_folders):
                name = os.path.basename(folder)

                # Load light directions
                light_file = os.path.join(folder, 'light_directions.txt')
                if not os.path.exists(light_file):
                    print(f"  Skipping {name}: no light_directions.txt")
                    continue

                lights = np.loadtxt(light_file)

                # Find all PNG images
                imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
                imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]

                n = min(len(imgs), len(lights))
                imgs, lights = imgs[:n], lights[:n]

                print(f"  {name}: {n} images")

                all_img_paths.extend(imgs)
                all_light_dirs.extend(lights)
                all_groups.extend([gid] * n)

            return all_img_paths, np.array(all_light_dirs, dtype=np.float32), np.array(all_groups)

        def train(self, img_paths, light_directions, groups=None,
                  epochs=100, batch_size=32, lr=1e-4, val_split=0.15,
                  use_augmentation=True, verbose=True):
            """
            Train the deep learning model.

            Args:
                img_paths: List of image file paths
                light_directions: Numpy array (N, 3)
                groups: Optional group labels for stratified split
                epochs: Number of training epochs
                batch_size: Batch size
                lr: Learning rate
                val_split: Validation split ratio
                use_augmentation: Whether to use data augmentation
                verbose: Print training progress

            Returns:
                Training history (dict)
            """
            # Create model
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

            # Loss and optimizer
            criterion = CombinedLoss(cos_weight=0.7, mse_weight=0.3)
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            early_stopping = EarlyStopping(patience=20, min_delta=0.001)

            # Training history
            history = {'train_loss': [], 'val_loss': [], 'train_error': [], 'val_error': []}

            if verbose:
                print(f"\nTraining {self.model_type.upper()} model...")
                print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
                print(f"Batch size: {batch_size}, Learning rate: {lr}")
                print("-" * 60)

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_losses = []
                train_preds, train_targets = [], []

                for images, targets in train_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_targets.extend(targets.cpu().numpy())

                # Validation phase
                self.model.eval()
                val_losses = []
                val_preds, val_targets = [], []

                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(images)
                        loss = criterion(outputs, targets)

                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())

                # Compute metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_error = compute_angular_error(np.array(train_preds), np.array(train_targets)).mean()
                val_error = compute_angular_error(np.array(val_preds), np.array(val_targets)).mean()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_error'].append(train_error)
                history['val_error'].append(val_error)

                # Learning rate scheduling
                scheduler.step()

                # Early stopping
                early_stopping(val_loss, self.model)

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Train Error: {train_error:.2f}° | Val Error: {val_error:.2f}°")

                if early_stopping.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # Restore best model
            early_stopping.restore(self.model)
            self.model.to(self.device)
            self.is_fitted = True

            if verbose:
                print("-" * 60)
                print(f"Best validation error: {min(history['val_error']):.2f}°")

            return history

        def cross_validate(self, img_paths, light_directions, groups,
                          epochs=50, batch_size=32, lr=1e-4, verbose=True):
            """
            Leave-One-Object-Out cross-validation.

            Returns:
                errors: Angular errors for all samples
                per_object_errors: Dict of per-object mean errors
            """
            unique_groups = np.unique(groups)
            all_errors = np.zeros(len(img_paths))
            per_object_errors = {}

            if verbose:
                print(f"\nLeave-One-Object-Out Cross-validation ({self.model_type.upper()})")
                print("=" * 60)

            for test_group in unique_groups:
                # Split data
                test_mask = groups == test_group
                train_mask = ~test_mask

                train_paths = [img_paths[i] for i in range(len(img_paths)) if train_mask[i]]
                train_lights = light_directions[train_mask]
                test_paths = [img_paths[i] for i in range(len(img_paths)) if test_mask[i]]
                test_lights = light_directions[test_mask]

                if verbose:
                    print(f"\nTesting on object {test_group} ({len(test_paths)} images)...")

                # Train model (without validation split for CV)
                self.model = self._create_model().to(self.device)

                # Create datasets
                train_transform = get_train_transforms(self.img_size)
                test_transform = get_val_transforms(self.img_size)

                train_dataset = LightDirectionDataset(train_paths, train_lights, train_transform, self.img_size)
                test_dataset = LightDirectionDataset(test_paths, test_lights, test_transform, self.img_size)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                # Training
                criterion = CombinedLoss()
                optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

                self.model.train()
                for epoch in range(epochs):
                    for images, targets in train_loader:
                        images = images.to(self.device)
                        targets = targets.to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                # Evaluate on test set
                self.model.eval()
                test_preds = []
                with torch.no_grad():
                    for images, _ in test_loader:
                        images = images.to(self.device)
                        outputs = self.model(images)
                        test_preds.extend(outputs.cpu().numpy())

                # Compute errors
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
            """
            Predict light directions for new images.

            Args:
                img_paths: List of image file paths

            Returns:
                Light directions (N, 3), normalized
            """
            if not self.is_fitted:
                raise ValueError("Model not trained. Call train() first.")

            # Create dataset
            dummy_lights = np.zeros((len(img_paths), 3), dtype=np.float32)
            transform = get_val_transforms(self.img_size)
            dataset = LightDirectionDataset(img_paths, dummy_lights, transform, self.img_size)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            # Predict
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    predictions.extend(outputs.cpu().numpy())

            return np.array(predictions)

        def predict_from_folder(self, folder, output_file=None):
            """Predict light directions for all images in a folder."""
            imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
            imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]

            print(f"Predicting light directions for {len(imgs)} images...")
            lights = self.predict(imgs)

            if output_file:
                np.savetxt(output_file, lights, fmt='%.4f')
                print(f"Saved to: {output_file}")

            return lights

        def save(self, path):
            """Save the trained model."""
            if not self.is_fitted:
                raise ValueError("Model not trained.")

            torch.save({
                'model_type': self.model_type,
                'img_size': self.img_size,
                'model_state_dict': self.model.state_dict(),
            }, path)
            print(f"Model saved to: {path}")

        def load(self, path):
            """Load a trained model."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model_type = checkpoint['model_type']
            self.img_size = checkpoint['img_size']
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.is_fitted = True
            print(f"Model loaded from: {path}")


# ==================== COMPARISON FUNCTION ====================

def compare_dl_models(training_folder, epochs=50, batch_size=32):
    """
    Compare all deep learning models.

    Returns:
        results: Dict of model_type -> mean_error
    """
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

        predictor = DeepLightPredictor(model_type=model_type, img_size=128)
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

        # Compare all DL models
        results = compare_dl_models(TRAINING_FOLDER, epochs=50, batch_size=32)
