"""
Light Direction Predictor for Robust Photometric Stereo
========================================================
Train ML/DL models to predict light directions from images only.

Available Models:
-----------------
Classical ML:
  - ridge: Ridge Regression (fast baseline)
  - rf: Random Forest (good balance)
  - gbr: Gradient Boosting Regressor
  - mlp: Multi-Layer Perceptron (sklearn)

Deep Learning (requires PyTorch):
  - cnn: Custom Lightweight CNN
  - resnet: ResNet18 Transfer Learning
  - efficientnet: EfficientNet-B0 Transfer Learning

Improvements:
1. Better physics-based features (shading-invariant)
2. Image normalization to reduce object-specific bias
3. Multi-scale features
4. Deep learning with transfer learning
5. Data augmentation for small datasets
"""

import numpy as np
import os
import glob
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
import pickle
import warnings

warnings.filterwarnings('ignore')

# Check for deep learning support
try:
    from .dl_models import DeepLightPredictor, TORCH_AVAILABLE
except ImportError:
    try:
        from dl_models import DeepLightPredictor, TORCH_AVAILABLE
    except ImportError:
        TORCH_AVAILABLE = False
        DeepLightPredictor = None

# Define which models are deep learning based
DL_MODELS = ['cnn', 'resnet', 'efficientnet']
ML_MODELS = ['ridge', 'rf', 'gbr', 'mlp']
ALL_MODELS = ML_MODELS + DL_MODELS


class LightDirectionPredictor:
    """
    Predicts light direction vectors from images using ML or DL.

    Supports both classical ML models (ridge, rf, gbr, mlp) and
    deep learning models (cnn, resnet, efficientnet).
    """

    def __init__(self, img_size=64, n_pca=64, normalize_images=True):
        """
        Args:
            img_size: Resize images to (img_size x img_size) for ML models
                      DL models use 128x128 by default
            n_pca: Number of PCA components for image features (ML only)
            normalize_images: Whether to normalize images (ML only)
        """
        self.img_size = img_size
        self.n_pca = n_pca
        self.normalize_images = normalize_images
        self.pca = None
        self.scaler = None
        self.model = None
        self.dl_predictor = None  # For deep learning models
        self.model_type = None
        self.is_fitted = False

    # ==================== FEATURE EXTRACTION ====================

    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess with optional normalization."""
        img = Image.open(img_path).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

        # Normalize to reduce object-specific intensity patterns
        if self.normalize_images:
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        else:
            arr = arr / 255.0

        return arr

    def _extract_enhanced_physics_features(self, img_arr):
        """
        Enhanced physics features designed for light direction estimation.
        Focus on RELATIVE patterns that generalize across objects.
        """
        h, w = img_arr.shape
        features = []

        # === 1. GRADIENT-BASED FEATURES (most important for lighting) ===
        # Sobel gradients
        grad_x = ndimage.sobel(img_arr, axis=1)
        grad_y = ndimage.sobel(img_arr, axis=0)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Gradient direction histogram (8 bins, 0-360°)
        grad_angle = np.arctan2(grad_y, grad_x)  # -3.14 to 3.14
        angle_bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(grad_angle, bins=angle_bins, weights=grad_mag)
        hist = hist / (hist.sum() + 1e-8)  # Normalize
        features.extend(hist)  # 8 features

        # Dominant gradient direction (weighted by magnitude)
        weighted_cos = (np.cos(grad_angle) * grad_mag).sum()
        weighted_sin = (np.sin(grad_angle) * grad_mag).sum()
        dominant_angle = np.arctan2(weighted_sin, weighted_cos)
        features.extend([np.cos(dominant_angle), np.sin(dominant_angle)])  # 2 features

        # === 2. SPATIAL DISTRIBUTION OF BRIGHTNESS ===
        # Divide image into 4x4 grid, compute relative brightness
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        grid_values = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = img_arr[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                grid_values.append(cell.mean())
        grid_values = np.array(grid_values)
        grid_values = grid_values / (grid_values.sum() + 1e-8)  # Relative
        features.extend(grid_values)  # 16 features

        # === 3. BRIGHTEST REGION ANALYSIS ===
        # Find brightest 10% pixels
        threshold = np.percentile(img_arr, 90)
        bright_mask = img_arr >= threshold
        if bright_mask.sum() > 0:
            y_coords, x_coords = np.where(bright_mask)
            # Centroid (normalized to [-1, 1])
            cx = (np.mean(x_coords) / w) * 2 - 1
            cy = (np.mean(y_coords) / h) * 2 - 1
            # Spread
            spread_x = np.std(x_coords) / w if len(x_coords) > 1 else 0
            spread_y = np.std(y_coords) / h if len(y_coords) > 1 else 0
        else:
            cx, cy, spread_x, spread_y = 0, 0, 0, 0
        features.extend([cx, cy, spread_x, spread_y])  # 4 features

        # === 4. SHADOW REGION ANALYSIS ===
        # Find darkest 10% pixels
        threshold_dark = np.percentile(img_arr, 10)
        dark_mask = img_arr <= threshold_dark
        if dark_mask.sum() > 0:
            y_coords, x_coords = np.where(dark_mask)
            sx = (np.mean(x_coords) / w) * 2 - 1
            sy = (np.mean(y_coords) / h) * 2 - 1
        else:
            sx, sy = 0, 0
        features.extend([sx, sy])  # 2 features

        # Light-shadow vector (points from shadow to bright)
        light_shadow_vec = [cx - sx, cy - sy]
        features.extend(light_shadow_vec)  # 2 features

        # === 5. ASYMMETRY FEATURES ===
        # Left-right asymmetry
        left_half = img_arr[:, :w // 2].mean()
        right_half = img_arr[:, w // 2:].mean()
        lr_asym = (right_half - left_half) / (left_half + right_half + 1e-8)

        # Top-bottom asymmetry
        top_half = img_arr[:h // 2, :].mean()
        bottom_half = img_arr[h // 2:, :].mean()
        tb_asym = (bottom_half - top_half) / (top_half + bottom_half + 1e-8)

        features.extend([lr_asym, tb_asym])  # 2 features

        # === 6. MULTI-SCALE FEATURES ===
        # Blur at different scales and compute gradient directions
        for sigma in [2, 4, 8]:
            blurred = ndimage.gaussian_filter(img_arr, sigma=sigma)
            gx = ndimage.sobel(blurred, axis=1)
            gy = ndimage.sobel(blurred, axis=0)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            w_cos = (np.cos(np.arctan2(gy, gx)) * mag).sum()
            w_sin = (np.sin(np.arctan2(gy, gx)) * mag).sum()
            dom_ang = np.arctan2(w_sin, w_cos)
            features.extend([np.cos(dom_ang), np.sin(dom_ang)])  # 2 features per scale

        # Total: 8 + 2 + 16 + 4 + 2 + 2 + 2 + 6 = 42 physics features
        return np.array(features, dtype=np.float32)

    def _extract_features_single(self, img_arr):
        """Extract all features from a single image."""
        pixel_features = img_arr.flatten()
        physics_features = self._extract_enhanced_physics_features(img_arr)
        return pixel_features, physics_features

    def extract_features(self, img_paths, fit_pca=False):
        """
        Extract features from multiple images.

        Args:
            img_paths: List of image file paths
            fit_pca: Whether to fit PCA (True for training)

        Returns:
            Feature matrix (n_images x n_features)
        """
        pixel_list, physics_list = [], []

        for path in img_paths:
            img_arr = self._load_and_preprocess_image(path)
            pix, phys = self._extract_features_single(img_arr)
            pixel_list.append(pix)
            physics_list.append(phys)

        X_pixels = np.array(pixel_list)
        X_physics = np.array(physics_list)

        # Apply PCA to pixel features
        if fit_pca:
            n_comp = min(self.n_pca, X_pixels.shape[0], X_pixels.shape[1])
            self.pca = PCA(n_components=n_comp)
            X_pca = self.pca.fit_transform(X_pixels)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit_pca=True first.")
            X_pca = self.pca.transform(X_pixels)

        # Combine features
        X = np.hstack([X_pca, X_physics])

        # Scale
        if fit_pca:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    # ==================== DATA LOADING ====================

    def load_training_data(self, training_folder):
        """
        Load all training data from folder structure.

        Expected structure:
            data/training/
                ballPNG/
                    001.png, 002.png, ..., 096.png
                    light_directions.txt
                bearPNG/
                    ...

        Returns:
            X: Feature matrix
            Y: Light direction matrix (n_samples x 3)
            groups: Object group labels (for cross-validation)
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

            lights = np.loadtxt(light_file) # Shape: (96, 3)

            # Find all PNG images (sorted)
            imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
            # Exclude mask.png and Normal_gt.png
            imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]

            if len(imgs) != len(lights):
                print(f"  Warning {name}: {len(imgs)} images vs {len(lights)} lights")

            n = min(len(imgs), len(lights))
            imgs, lights = imgs[:n], lights[:n]

            print(f"  {name}: {n} images")

            all_img_paths.extend(imgs)
            all_light_dirs.extend(lights)
            all_groups.extend([gid] * n)

        # Convert to arrays
        Y = np.array(all_light_dirs, dtype=np.float32)
        groups = np.array(all_groups)

        print(f"\nExtracting features from {len(all_img_paths)} images...")
        X = self.extract_features(all_img_paths, fit_pca=True)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Light directions shape: {Y.shape}")

        return X, Y, groups, object_folders

    # ==================== TRAINING ====================

    def train(self, X, Y, model_type='mlp', img_paths=None, groups=None, **dl_kwargs):
        """
        Train the light direction prediction model.

        Args:
            X: Feature matrix (for ML models) or None (for DL models)
            Y: Light direction targets (n_samples x 3)
            model_type: Model type - ML: 'ridge', 'rf', 'gbr', 'mlp'
                                     DL: 'cnn', 'resnet', 'efficientnet'
            img_paths: List of image paths (required for DL models)
            groups: Group labels (optional, for DL validation)
            **dl_kwargs: Additional kwargs for DL training (epochs, batch_size, lr)

        Returns:
            angular_errors: Training angular errors
        """
        self.model_type = model_type

        # Check if this is a deep learning model
        if model_type in DL_MODELS:
            if not TORCH_AVAILABLE:
                raise ImportError(
                    f"PyTorch is required for {model_type} model. "
                    "Install with: pip install torch torchvision"
                )
            if img_paths is None:
                raise ValueError("img_paths required for deep learning models")

            return self._train_dl(img_paths, Y, model_type, **dl_kwargs)
        else:
            return self._train_ml(X, Y, model_type)

    def _train_ml(self, X, Y, model_type):
        """Train classical ML model."""
        print(f"\nTraining {model_type.upper()} model (Classical ML)...")

        if model_type == 'ridge':
            self.model = Ridge(alpha=10.0)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=15,
                min_samples_leaf=5, n_jobs=-1, random_state=42
            )
        elif model_type == 'gbr':
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100, max_depth=5,
                    learning_rate=0.1, random_state=42
                )
            )
        elif model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                alpha=0.01,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown ML model: {model_type}")

        self.model.fit(X, Y)
        self.is_fitted = True

        # Evaluate
        Y_pred = self.model.predict(X)
        Y_pred = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)

        cos_sim = np.clip(np.sum(Y * Y_pred, axis=1), -1, 1)
        angular_errors = np.arccos(cos_sim) * 180 / np.pi
        print(f"Training angular error: {angular_errors.mean():.2f}° ± {angular_errors.std():.2f}°")

        return angular_errors

    def _train_dl(self, img_paths, Y, model_type, epochs=100, batch_size=32, lr=1e-4, **kwargs):
        """Train deep learning model."""
        print(f"\nTraining {model_type.upper()} model (Deep Learning)...")

        # Create DL predictor
        self.dl_predictor = DeepLightPredictor(model_type=model_type, img_size=128)

        # Train
        history = self.dl_predictor.train(
            img_paths, Y,
            epochs=epochs, batch_size=batch_size, lr=lr,
            use_augmentation=True, verbose=True
        )

        self.is_fitted = True

        # Return final training errors
        return np.array([history['train_error'][-1]])

    def cross_validate(self, X, Y, groups, model_type='mlp', img_paths=None, **dl_kwargs):
        """
        Leave-one-object-out cross-validation.
        This tests how well the model generalizes to unseen objects.

        Args:
            X: Feature matrix (for ML models)
            Y: Light direction targets
            groups: Object group labels
            model_type: Model type (ML or DL)
            img_paths: List of image paths (required for DL models)
            **dl_kwargs: Additional kwargs for DL training

        Returns:
            errors: Angular errors for all samples
        """
        # Check if this is a deep learning model
        if model_type in DL_MODELS:
            if not TORCH_AVAILABLE:
                raise ImportError(
                    f"PyTorch is required for {model_type} model. "
                    "Install with: pip install torch torchvision"
                )
            if img_paths is None:
                raise ValueError("img_paths required for deep learning models")

            return self._cross_validate_dl(img_paths, Y, groups, model_type, **dl_kwargs)
        else:
            return self._cross_validate_ml(X, Y, groups, model_type)

    def _cross_validate_ml(self, X, Y, groups, model_type):
        """Cross-validate classical ML model."""
        print(f"\nLeave-One-Object-Out Cross-validating {model_type.upper()} (Classical ML)...")

        if model_type == 'ridge':
            model = Ridge(alpha=10.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1)
        elif model_type == 'gbr':
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5))
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(512, 256, 128), alpha=0.01, max_iter=1000)
        else:
            raise ValueError(f"Unknown ML model: {model_type}")

        logo = LeaveOneGroupOut()
        Y_pred = cross_val_predict(model, X, Y, groups=groups, cv=logo)
        Y_pred = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)

        cos_sim = np.clip(np.sum(Y * Y_pred, axis=1), -1, 1)
        errors = np.arccos(cos_sim) * 180 / np.pi

        # Per-object errors
        print("\nPer-object angular errors:")
        for g in np.unique(groups):
            mask = groups == g
            print(f"  Object {g}: {errors[mask].mean():.2f}°")

        print(f"\nOverall CV angular error: {errors.mean():.2f}° ± {errors.std():.2f}°")
        return errors

    def _cross_validate_dl(self, img_paths, Y, groups, model_type, epochs=50, batch_size=32, lr=1e-4):
        """Cross-validate deep learning model."""
        print(f"\nLeave-One-Object-Out Cross-validating {model_type.upper()} (Deep Learning)...")

        dl_predictor = DeepLightPredictor(model_type=model_type, img_size=128)
        errors, per_object = dl_predictor.cross_validate(
            img_paths, Y, groups,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=True
        )
        return errors

    # ==================== INFERENCE ====================

    def predict(self, img_paths):
        """
        Predict light directions for new images.

        Args:
            img_paths: List of image file paths

        Returns:
            Light directions matrix (n_images x 3), normalized
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        # Use DL predictor if model is deep learning based
        if self.model_type in DL_MODELS:
            return self.dl_predictor.predict(img_paths)
        else:
            X = self.extract_features(img_paths, fit_pca=False)
            Y_pred = self.model.predict(X)
            return Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)

    def predict_from_folder(self, folder, output_file=None):
        """
        Predict light directions for all images in a folder.

        Args:
            folder: Folder containing PNG images
            output_file: Optional path to save light_directions.txt

        Returns:
            Light directions matrix (96 x 3)
        """
        imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
        imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]

        print(f"Predicting light directions for {len(imgs)} images...")
        lights = self.predict(imgs)

        if output_file:
            np.savetxt(output_file, lights, fmt='%.4f')
            print(f"Saved to: {output_file}")

        return lights

    # ==================== SAVE/LOAD ====================

    def save(self, path):
        """Save the trained model and preprocessing objects."""
        if not self.is_fitted:
            raise ValueError("Model not trained.")

        # Handle DL models differently
        if self.model_type in DL_MODELS:
            # Save DL model with .pt extension
            dl_path = path.replace('.pkl', '_dl.pt') if path.endswith('.pkl') else path + '_dl.pt'
            self.dl_predictor.save(dl_path)

            # Save metadata
            with open(path, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'is_dl': True,
                    'dl_path': dl_path
                }, f)
        else:
            # Save ML model
            with open(path, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'is_dl': False,
                    'img_size': self.img_size,
                    'n_pca': self.n_pca,
                    'normalize': self.normalize_images,
                    'pca': self.pca,
                    'scaler': self.scaler,
                    'model': self.model
                }, f)
        print(f"Model saved to: {path}")

    def load(self, path):
        """Load model (ML or DL)."""
        with open(path, 'rb') as f:
            d = pickle.load(f)

        self.model_type = d.get('model_type', 'rf')  # Default to rf for old models

        if d.get('is_dl', False):
            # Load DL model
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required to load this model")

            self.dl_predictor = DeepLightPredictor(model_type=self.model_type)
            self.dl_predictor.load(d['dl_path'])
        else:
            # Load ML model
            self.img_size = d['img_size']
            self.n_pca = d['n_pca']
            self.normalize_images = d['normalize']
            self.pca = d['pca']
            self.scaler = d['scaler']
            self.model = d['model']

        self.is_fitted = True
        print(f"Model loaded from: {path}")


def compare_all_models(training_folder, include_dl=True, dl_epochs=50):
    """
    Compare all model types (ML and optionally DL).

    Args:
        training_folder: Path to training data folder
        include_dl: Whether to include deep learning models
        dl_epochs: Number of epochs for DL model cross-validation

    Returns:
        best_model: Name of best performing model
        results: Dict of model_type -> mean_error
    """
    print("=" * 70)
    print("COMPARING ALL MODELS (ML + DL)")
    print("=" * 70)

    predictor = LightDirectionPredictor(img_size=64, n_pca=64)
    X, Y, groups, object_folders = predictor.load_training_data(training_folder)

    # Get image paths for DL models
    img_paths = []
    for gid, folder in enumerate(object_folders):
        imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
        imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
        light_file = os.path.join(folder, 'light_directions.txt')
        if os.path.exists(light_file):
            lights = np.loadtxt(light_file)
            n = min(len(imgs), len(lights))
            img_paths.extend(imgs[:n])

    results = {}

    # ==================== ML MODELS ====================
    print("\n" + "=" * 70)
    print("CLASSICAL MACHINE LEARNING MODELS")
    print("=" * 70)

    for model_type in ML_MODELS:
        print(f"\n{'=' * 50}")
        print(f"Model: {model_type.upper()}")
        print("=" * 50)
        errors = predictor.cross_validate(X, Y, groups, model_type=model_type)
        results[model_type] = errors.mean()

    # ==================== DL MODELS ====================
    if include_dl and TORCH_AVAILABLE:
        print("\n" + "=" * 70)
        print("DEEP LEARNING MODELS")
        print("=" * 70)

        for model_type in DL_MODELS:
            print(f"\n{'=' * 50}")
            print(f"Model: {model_type.upper()}")
            print("=" * 50)
            errors = predictor.cross_validate(
                X, Y, groups, model_type=model_type,
                img_paths=img_paths, epochs=dl_epochs
            )
            results[model_type] = errors.mean()
    elif include_dl and not TORCH_AVAILABLE:
        print("\n[WARNING] PyTorch not available. Skipping DL models.")
        print("Install with: pip install torch torchvision")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY - ALL MODELS COMPARISON")
    print("=" * 70)

    # Separate ML and DL results
    ml_results = {k: v for k, v in results.items() if k in ML_MODELS}
    dl_results = {k: v for k, v in results.items() if k in DL_MODELS}

    print("\nClassical ML Models:")
    for m, err in sorted(ml_results.items(), key=lambda x: x[1]):
        print(f"  {m:15s}: {err:.2f}°")

    if dl_results:
        print("\nDeep Learning Models:")
        for m, err in sorted(dl_results.items(), key=lambda x: x[1]):
            print(f"  {m:15s}: {err:.2f}°")

    print("\n" + "-" * 70)
    print("Overall Ranking:")
    for rank, (m, err) in enumerate(sorted(results.items(), key=lambda x: x[1]), 1):
        model_cat = "DL" if m in DL_MODELS else "ML"
        print(f"  {rank}. {m:15s} ({model_cat}): {err:.2f}°")

    best = min(results, key=results.get)
    best_cat = "Deep Learning" if best in DL_MODELS else "Classical ML"
    print(f"\nBest model: {best.upper()} ({best_cat}) with {results[best]:.2f}° error")

    return best, results


def compare_ml_models(training_folder):
    """Compare only classical ML models (no PyTorch required)."""
    return compare_all_models(training_folder, include_dl=False)


def compare_dl_models(training_folder, epochs=50):
    """Compare only deep learning models."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch torchvision")

    print("=" * 70)
    print("COMPARING DEEP LEARNING MODELS")
    print("=" * 70)

    predictor = LightDirectionPredictor(img_size=64, n_pca=64)
    _, Y, groups, object_folders = predictor.load_training_data(training_folder)

    # Get image paths
    img_paths = []
    for gid, folder in enumerate(object_folders):
        imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
        imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
        light_file = os.path.join(folder, 'light_directions.txt')
        if os.path.exists(light_file):
            lights = np.loadtxt(light_file)
            n = min(len(imgs), len(lights))
            img_paths.extend(imgs[:n])

    results = {}
    for model_type in DL_MODELS:
        print(f"\n{'=' * 50}")
        print(f"Model: {model_type.upper()}")
        print("=" * 50)
        errors = predictor.cross_validate(
            None, Y, groups, model_type=model_type,
            img_paths=img_paths, epochs=epochs
        )
        results[model_type] = errors.mean()

    print("\n" + "=" * 70)
    print("DEEP LEARNING MODELS SUMMARY")
    print("=" * 70)
    for m, err in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {m:15s}: {err:.2f}°")

    best = min(results, key=results.get)
    print(f"\nBest DL model: {best.upper()}")
    return best, results


# ==================== TRAIN BEST MODELS ====================

def train_best_ml(training_folder, output_path='./light_predictor_ml_v3.pkl'):
    """
    Compare all ML models, train the best one, and save it.

    Workflow:
    1. Cross-validate all ML models (ridge, rf, gbr, mlp)
    2. Select the best performing model
    3. Train it on full dataset
    4. Save to output_path

    Args:
        training_folder: Path to training data
        output_path: Path to save the best ML model

    Returns:
        predictor: Trained predictor
        best_model: Name of the best model
        results: All model comparison results
    """
    print("=" * 70)
    print("ML PIPELINE: Compare → Train Best → Save")
    print("=" * 70)

    # Step 1: Compare all ML models
    print("\n" + "=" * 70)
    print("STEP 1: COMPARING ALL ML MODELS")
    print("=" * 70)

    best_model, results = compare_ml_models(training_folder)

    # Step 2: Train the best model on full data
    print("\n" + "=" * 70)
    print(f"STEP 2: TRAINING BEST ML MODEL ({best_model.upper()}) ON FULL DATA")
    print("=" * 70)

    predictor = LightDirectionPredictor(img_size=64, n_pca=64)
    X, Y, groups, _ = predictor.load_training_data(training_folder)
    predictor.train(X, Y, model_type=best_model)

    # Step 3: Save the model
    predictor.save(output_path)

    # Summary
    print("\n" + "=" * 70)
    print("ML PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best ML model: {best_model.upper()}")
    print(f"CV Error: {results[best_model]:.2f}°")
    print(f"Saved to: {output_path}")

    print("\nAll ML Results:")
    for m, err in sorted(results.items(), key=lambda x: x[1]):
        marker = "→ BEST" if m == best_model else ""
        print(f"  {m:10s}: {err:.2f}° {marker}")

    return predictor, best_model, results


def train_best_dl(training_folder, output_path='./light_predictor_dl_v3.pkl', epochs=100):
    """
    Compare all DL models, train the best one, and save it.

    Workflow:
    1. Cross-validate all DL models (cnn, resnet, efficientnet)
    2. Select the best performing model
    3. Train it on full dataset
    4. Save to output_path

    Args:
        training_folder: Path to training data
        output_path: Path to save the best DL model
        epochs: Number of training epochs

    Returns:
        predictor: Trained predictor
        best_model: Name of the best model
        results: All model comparison results
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch torchvision")

    print("=" * 70)
    print("DL PIPELINE: Compare → Train Best → Save")
    print("=" * 70)

    # Step 1: Compare all DL models
    print("\n" + "=" * 70)
    print("STEP 1: COMPARING ALL DL MODELS")
    print("=" * 70)

    best_model, results = compare_dl_models(training_folder, epochs=epochs)

    # Step 2: Train the best model on full data
    print("\n" + "=" * 70)
    print(f"STEP 2: TRAINING BEST DL MODEL ({best_model.upper()}) ON FULL DATA")
    print("=" * 70)

    predictor = LightDirectionPredictor(img_size=64, n_pca=64)
    X, Y, groups, object_folders = predictor.load_training_data(training_folder)

    # Get image paths for DL training
    img_paths = []
    for folder in object_folders:
        imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
        imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
        light_file = os.path.join(folder, 'light_directions.txt')
        if os.path.exists(light_file):
            lights = np.loadtxt(light_file)
            n = min(len(imgs), len(lights))
            img_paths.extend(imgs[:n])

    predictor.train(X, Y, model_type=best_model, img_paths=img_paths, epochs=epochs)

    # Step 3: Save the model
    predictor.save(output_path)

    # Summary
    print("\n" + "=" * 70)
    print("DL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best DL model: {best_model.upper()}")
    print(f"CV Error: {results[best_model]:.2f}°")
    print(f"Saved to: {output_path}")

    print("\nAll DL Results:")
    for m, err in sorted(results.items(), key=lambda x: x[1]):
        marker = "→ BEST" if m == best_model else ""
        print(f"  {m:15s}: {err:.2f}° {marker}")

    return predictor, best_model, results


# ==================== MAIN SCRIPT ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Light Direction Predictor - Train and Compare ML/DL Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare and train best ML model
  python light_direction_predictor.py --mode train_best_ml

  # Compare and train best DL model
  python light_direction_predictor.py --mode train_best_dl --epochs 100

  # Compare all models without training
  python light_direction_predictor.py --mode compare_all

  # Train a specific model
  python light_direction_predictor.py --mode train --model resnet --epochs 100
        """
    )
    parser.add_argument('--mode', type=str, default='train_best_ml',
                        choices=['train_best_ml', 'train_best_dl', 'compare_all', 'compare_ml', 'compare_dl', 'train'],
                        help='Mode: train_best_ml, train_best_dl, compare_all, compare_ml, compare_dl, or train')
    parser.add_argument('--model', type=str, default='rf',
                        choices=ALL_MODELS,
                        help='Model type (only for --mode train)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for DL models (default: 100)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path (default: auto-generated based on mode)')
    parser.add_argument('--data', type=str, default='./data/training/',
                        help='Training data folder')

    args = parser.parse_args()

    TRAINING_FOLDER = args.data

    if args.mode == 'train_best_ml':
        # RECOMMENDED: Compare all ML models, train the best, save it
        output_path = args.output or './light_predictor_ml_v3.pkl'
        predictor, best_model, results = train_best_ml(TRAINING_FOLDER, output_path)

    elif args.mode == 'train_best_dl':
        # RECOMMENDED: Compare all DL models, train the best, save it
        output_path = args.output or './light_predictor_dl_v3.pkl'
        predictor, best_model, results = train_best_dl(TRAINING_FOLDER, output_path, epochs=args.epochs)

    elif args.mode == 'compare_all':
        # Compare all models (ML + DL) without training
        best_model, results = compare_all_models(TRAINING_FOLDER, include_dl=True, dl_epochs=args.epochs)

    elif args.mode == 'compare_ml':
        # Compare only ML models without training
        best_model, results = compare_ml_models(TRAINING_FOLDER)

    elif args.mode == 'compare_dl':
        # Compare only DL models without training
        best_model, results = compare_dl_models(TRAINING_FOLDER, epochs=args.epochs)

    elif args.mode == 'train':
        # Train specific model
        print(f"\n{'=' * 70}")
        print(f"Training {args.model.upper()} model")
        print("=" * 70)

        predictor = LightDirectionPredictor(img_size=64, n_pca=64)
        X, Y, groups, object_folders = predictor.load_training_data(TRAINING_FOLDER)

        # Get image paths for DL models
        if args.model in DL_MODELS:
            img_paths = []
            for folder in object_folders:
                imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
                imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
                light_file = os.path.join(folder, 'light_directions.txt')
                if os.path.exists(light_file):
                    lights = np.loadtxt(light_file)
                    n = min(len(imgs), len(lights))
                    img_paths.extend(imgs[:n])

            predictor.train(X, Y, model_type=args.model, img_paths=img_paths, epochs=args.epochs)
        else:
            predictor.train(X, Y, model_type=args.model)

        predictor.save(args.output)
        print(f"\nModel saved to: {args.output}")