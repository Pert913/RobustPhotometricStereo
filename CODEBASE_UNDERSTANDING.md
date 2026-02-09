# Comprehensive Codebase Analysis: RobustPhotometricStereo

A state-of-the-art robust photometric stereo implementation with multiple solver algorithms and an ML/DL-based light direction predictor.

**Total Lines of Code:** ~2,300+ lines (excluding dependencies)

**Last Updated:** February 2025

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Components & Responsibilities](#2-core-components--responsibilities)
3. [Data Flow](#3-data-flow)
4. [Key Algorithms](#4-key-algorithms)
5. [ML/DL Models: Light Direction Predictor](#5-mldl-models-light-direction-predictor)
6. [Deep Learning Architecture Details](#6-deep-learning-architecture-details)
7. [Configuration & Parameters](#7-configuration--parameters)
8. [Entry Points & Usage](#8-entry-points--usage)
9. [Dependencies & Frameworks](#9-dependencies--frameworks)
10. [Testing Structure](#10-testing-structure)
11. [Data Handling & Formats](#11-data-handling--formats)
12. [Workflow Examples](#12-workflow-examples)
13. [Key Innovations & Robust Features](#13-key-innovations--robust-features)

---

## 1. Project Structure

```
RobustPhotometricStereo/
├── Core RPS Implementation (root level)
│   ├── rps.py                    # Main RPS class (302 lines)
│   ├── rpsnumerics.py            # Numerical algorithms (176 lines)
│   ├── psutil.py                 # Utility functions (163 lines)
│   ├── demo.py                   # Demonstration script (54 lines)
│   ├── test.py                   # Test script (30 lines)
│   ├── pngToNpy.py               # Image conversion utility (56 lines)
│   └── README.md                 # Project documentation
│
├── light_direction_predictor/    # ML/DL module
│   ├── light_direction_predictor.py   # Main predictor class (600+ lines)
│   ├── dl_models.py                   # Deep learning models (580+ lines) [NEW]
│   ├── rps_integration.py             # Integration pipeline (493 lines)
│   ├── __init__.py                    # Package init with exports
│   ├── light_predictor_v2.pkl         # Trained ML model (Random Forest)
│   ├── README.md                      # ML/DL module documentation
│   └── data/
│       ├── training/             # Training data (10 objects × 96 images)
│       │   ├── ballPNG/
│       │   ├── bearPNG/
│       │   ├── buddhaPNG/
│       │   ├── catPNG/
│       │   ├── cowPNG/
│       │   ├── gobletPNG/
│       │   ├── harvestPNG/
│       │   ├── pot1PNG/
│       │   ├── pot2PNG/
│       │   └── readingPNG/
│       └── testing/              # Test data (Pikachu dataset)
│           ├── batteryPNG/
│           ├── bootaoPNG/
│           ├── bootao2PNG/
│           ├── bootao3PNG/
│           └── dimooPNG/
│
├── data/                         # Original datasets
│   ├── buddha/
│   ├── bunny/
│   └── cat/
│
├── CODEBASE_UNDERSTANDING.md     # This documentation
└── Output visualizations (*.png, *.npy)
```

---

## 2. Core Components & Responsibilities

### A. RPS (Robust Photometric Stereo) Class (`rps.py`)

**Purpose:** Main API for photometric stereo computation

**Key Data Members:**
| Member | Description |
|--------|-------------|
| `M` | Measurement matrix (p × f) - pixels × images |
| `L` | Light matrix (3 × f) - light direction vectors |
| `N` | Normal matrix (p × 3) - surface normals |
| `foreground_ind` | Indices of foreground pixels |
| `background_ind` | Indices of background pixels |

**Solver Methods (6 variants):**

| Solver | Description |
|--------|-------------|
| `L2_SOLVER` | Conventional least-squares (Woodham 1980) |
| `L1_SOLVER` | L1 residual minimization (single-threaded) |
| `L1_SOLVER_MULTICORE` | L1 with multiprocessing |
| `SBL_SOLVER` | Sparse Bayesian Learning (single-threaded) |
| `SBL_SOLVER_MULTICORE` | SBL with multiprocessing |
| `RPCA_SOLVER` | Robust PCA decomposition |

**Key Methods:**
- `load_lighttxt()` / `load_lightnpy()` - Load lighting information
- `load_images()` / `load_npyimages()` - Load image data
- `load_mask()` - Load binary foreground mask
- `solve()` - Execute stereo algorithm
- `save_normalmap()` - Save computed normals
- `disp_normalmap()` - Visualize results

### B. Numerical Algorithms (`rpsnumerics.py`)

**Purpose:** Core mathematical operations for robust stereo

**Key Algorithms:**

| Algorithm | Function | Description |
|-----------|----------|-------------|
| L1 Residual Minimization | `L1_residual_min` | Iteratively reweighted least squares (IRLS) |
| Sparse Bayesian Learning | `sparse_bayesian_learning` | Sparse regression via Bayesian inference |
| Robust PCA | `rpca_inexact_alm` | Inexact Augmented Lagrangian Multiplier method |

**Utility Functions:**
- `pos()`, `neg()` - Extract positive/negative elements
- `shrinkage()` - Soft thresholding operation

### C. Utility Functions (`psutil.py`)

**Purpose:** I/O, preprocessing, visualization, evaluation

| Category | Functions |
|----------|-----------|
| Data Loading | `load_lighttxt()`, `load_lightnpy()`, `load_image()`, `load_images()`, `load_npyimages()` |
| Visualization | `disp_normalmap()`, `save_normalmap_as_npy()`, `load_normalmap_from_npy()` |
| Evaluation | `evaluate_angular_error()` |

### D. Light Direction Predictor (`light_direction_predictor.py`)

**Purpose:** ML/DL-based prediction of light directions from images

**Key Components:**
- `LightDirectionPredictor` - Main class supporting 7 model types
- `compare_all_models()` - Compare ML + DL models
- `compare_ml_models()` - Compare only classical ML models
- `compare_dl_models()` - Compare only deep learning models

### E. Deep Learning Models (`dl_models.py`) [NEW]

**Purpose:** PyTorch-based CNN models for light direction prediction

**Key Components:**
- `LightCNN` - Custom 5-layer CNN architecture
- `ResNetLight` - ResNet18 with transfer learning
- `EfficientNetLight` - EfficientNet-B0 with transfer learning
- `DeepLightPredictor` - Training and inference wrapper
- `LightDirectionDataset` - PyTorch Dataset class
- Loss functions, data augmentation, early stopping utilities

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. DATA PREPARATION                         │
├─────────────────────────────────────────────────────────────────┤
│  PNG Images (612×512)                                           │
│         ↓                                                       │
│  pngToNpy.py (convert to grayscale)                            │
│         ↓                                                       │
│  NPY Arrays (612×512 × uint8)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      2. DATA LOADING                            │
├─────────────────────────────────────────────────────────────────┤
│  rps.load_images() or rps.load_npyimages()                     │
│         ↓                                                       │
│  Measurement matrix M (p × f)                                   │
│  where p = height × width, f = number of images                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              3. LIGHT DIRECTION (Known or Predicted)            │
├─────────────────────────────────────────────────────────────────┤
│  Option A: rps.load_lighttxt("light_directions.txt")           │
│                                                                 │
│  Option B: ML/DL Prediction (NEW)                              │
│    - LightDirectionPredictor.predict_from_folder()             │
│    - Supports: ridge, rf, gbr, mlp, cnn, resnet, efficientnet  │
│         ↓                                                       │
│  Light matrix L (3 × f)                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. MASKING (Optional)                        │
├─────────────────────────────────────────────────────────────────┤
│  rps.load_mask("mask.png")                                     │
│         ↓                                                       │
│  foreground_ind, background_ind                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      5. COMPUTATION                             │
├─────────────────────────────────────────────────────────────────┤
│  rps.solve(method)                                             │
│      ├─ L2: N = lstsq(Lᵀ, Mᵀ).T                               │
│      ├─ L1: Per-pixel L1 minimization                          │
│      ├─ SBL: Per-pixel Bayesian regression                     │
│      └─ RPCA: Matrix decomposition                             │
│         ↓                                                       │
│  Normal matrix N (p × 3)                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        6. OUTPUT                                │
├─────────────────────────────────────────────────────────────────┤
│  rps.save_normalmap("est_normal")                              │
│         ↓                                                       │
│  est_normal.npy (height × width × 3)                           │
│  Displayed as RGB image: N = (N + 1) / 2                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Algorithms

### A. Woodham's L2 (Least-Squares)

```
Problem: Find N such that M = L·N (least squares fit)

Algorithm:
  N = (Lᵀ·L)⁻¹·Lᵀ·M  [via lstsq]
  N ← N / ||N||       [normalize]

Complexity: O(f³ + p·f²)

Properties:
  - Fast
  - Sensitive to outliers
  - Good for clean Lambertian data
```

### B. L1 Residual Minimization (IRLS)

```
Problem: minimize ||L·n - m||₁

Algorithm:
  1. Initialize: W = I
  2. Iterate:
     - Solve weighted LS: n = (WL)ᵀ\(Wm)
     - r = m - L·n
     - W = diag(1 / max(|r|, ε))
  3. Stop when ||n - n_old|| < tol

Complexity: O(iter × f³) per pixel

Properties:
  - Robust to sparse outliers
  - Effective against specularities
  - More computationally expensive
```

### C. Sparse Bayesian Learning (SBL)

```
Problem: Sparse regression with Bayesian inference

Model:
  m = L·n + e, where e ~ N(0, Σ)
  γ = diag precision, λ₁ = L2 prior strength

Algorithm:
  1. Initialize: γ = 1
  2. Iterate:
     - W = diag(1/γ)
     - Solve ridge regression: n = (λ₂I + LᵀWL)⁻¹LᵀWm
     - Update precision: γ = (m - Ln)² + diag(LᵀWL)⁻¹ + λ₁I
  3. Stop when ||n - n_old|| < tol

Complexity: O(iter × f³) per pixel

Properties:
  - Probabilistic interpretation
  - Learns uncertainty per component
  - Better generalization than L1
```

### D. Robust PCA (RPCA)

```
Problem: Decompose D = A + E where A low-rank, E sparse

Minimize: ||A||* + λ||E||₁  subject to D = A + E

Method: Inexact Augmented Lagrangian Multiplier

Algorithm:
  1. Initialize: A=0, E=0, Y=D, μ=1.25/σ_max(D)
  2. Iterate:
     - Soft-threshold E: E = shrink(D-A+Y/μ, λ/μ)
     - SVD of (D-E+Y/μ), truncate at 1/μ: A = U·diag(σ-1/μ)·Vᵀ
     - Dual update: Y ← Y + μ(D - A - E)
     - μ ← min(μ·ρ, μ_bar)
  3. Stop when ||D-A-E||_F / ||D||_F < tol

Complexity: O(iter × min(m,n)²·n)

Properties:
  - Exploits low-rank structure (Lambertian component)
  - Separates shadows/specularities (sparse outliers)
  - Globally optimal solution
  - Most robust method
```

### Algorithm Comparison

| Solver | Method | Robustness | Speed | Best For |
|--------|--------|------------|-------|----------|
| `L2_SOLVER` | Least-squares | Low | Fast | Clean Lambertian data |
| `L1_SOLVER` | IRLS | Medium | Slow | Sparse outliers |
| `L1_SOLVER_MULTICORE` | Parallel L1 | Medium | Medium | Sparse outliers (faster) |
| `SBL_SOLVER` | Sparse Bayesian | High | Slow | Uncertainty estimation |
| `SBL_SOLVER_MULTICORE` | Parallel SBL | High | Medium | Uncertainty (faster) |
| `RPCA_SOLVER` | Robust PCA | Highest | Medium | Shadows & specularities |

---

## 5. ML/DL Models: Light Direction Predictor

### Purpose

Remove dependency on known light directions when collecting own data. Predict 3D light direction vectors from images alone.

### Available Models (7 Total)

#### Classical ML Models (sklearn-based)

| Model | Code | Description | Speed | Typical Error |
|-------|------|-------------|-------|---------------|
| Ridge Regression | `ridge` | Linear regression with L2 regularization | Fast | ~15-20° |
| Random Forest | `rf` | Ensemble of 200 decision trees | Medium | ~10-15° |
| Gradient Boosting | `gbr` | Sequential ensemble learning | Slow | ~8-12° |
| MLP | `mlp` | Multi-layer perceptron (512-256-128) | Medium | ~10-15° |

#### Deep Learning Models (PyTorch-based) [NEW]

| Model | Code | Description | Speed | Expected Error |
|-------|------|-------------|-------|----------------|
| Custom CNN | `cnn` | 5-layer CNN designed for light estimation | Fast | ~8-12° |
| ResNet18 | `resnet` | Transfer learning from ImageNet | Medium | ~5-10° |
| EfficientNet-B0 | `efficientnet` | Efficient architecture with transfer learning | Medium | ~5-10° |

### Feature Extraction Pipeline (ML Models)

```
┌─────────────────────────────────────────────────────────────────┐
│                  1. IMAGE PREPROCESSING                         │
├─────────────────────────────────────────────────────────────────┤
│  - Resize to 64×64                                             │
│  - Optional normalization: (I - mean) / (std + 1e-8)           │
│  - Flatten to 4096-dim vectors                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. PIXEL FEATURES                            │
├─────────────────────────────────────────────────────────────────┤
│  - PCA dimensionality reduction                                │
│  - Configurable: n_pca = 64 components (default)               │
│  - Output: 64-dimensional vector                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              3. PHYSICS-BASED FEATURES (42 dims)                │
├─────────────────────────────────────────────────────────────────┤
│  - Gradient Features (8 dims): 8-bin histogram                 │
│  - Dominant Gradient (2 dims): Weighted angle (cos, sin)       │
│  - Spatial Distribution (16 dims): 4×4 grid brightness ratios  │
│  - Brightest Region (4 dims): Centroid (cx, cy), spread        │
│  - Shadow Region (2 dims): Dark pixels centroid                │
│  - Light-Shadow Vector (2 dims): Direction dark→bright         │
│  - Asymmetry (2 dims): Left-right, top-bottom ratios           │
│  - Multi-scale (6 dims): Gradients at σ=2,4,8 scales           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   4. COMBINED FEATURES                          │
├─────────────────────────────────────────────────────────────────┤
│  - Concatenate: [PCA_features, Physics_features]               │
│  - Standardize with StandardScaler                             │
│  - Total: 64 + 42 = 106 dimensions                             │
└─────────────────────────────────────────────────────────────────┘
```

### Training Data Structure

```
data/training/
├── ballPNG/        (96 images + light_directions.txt)
├── bearPNG/        (96 images + light_directions.txt)
├── buddhaPNG/      (96 images + light_directions.txt)
├── catPNG/         (96 images + light_directions.txt)
├── cowPNG/         (96 images + light_directions.txt)
├── gobletPNG/      (96 images + light_directions.txt)
├── harvestPNG/     (96 images + light_directions.txt)
├── pot1PNG/        (96 images + light_directions.txt)
├── pot2PNG/        (96 images + light_directions.txt)
└── readingPNG/     (96 images + light_directions.txt)
```

**Total training samples:** 10 objects × 96 images = 960 images

### Evaluation

**Metric:** Angular error (degrees) between predicted and ground truth light vectors

```python
cos_similarity = dot(y_pred, y_gt)
error = arccos(clip(cos_sim, -1, 1)) × 180/π
```

**Validation Strategy:** Leave-One-Object-Out Cross-Validation
- Tests generalization to completely unseen object categories
- Per-object error reporting
- Enables unbiased model comparison

---

## 6. Deep Learning Architecture Details [NEW]

### Custom CNN (`cnn`) - LightCNN

```
Input: (1, 128, 128) grayscale image
         ↓
Conv Block 1: Conv(1→32, 3×3) + BN + ReLU + Conv(32→32) + BN + ReLU + MaxPool(2) + Dropout(0.3)
         ↓  Output: (32, 64, 64)
Conv Block 2: Conv(32→64, 3×3) + BN + ReLU + Conv(64→64) + BN + ReLU + MaxPool(2) + Dropout(0.3)
         ↓  Output: (64, 32, 32)
Conv Block 3: Conv(64→128, 3×3) + BN + ReLU + Conv(128→128) + BN + ReLU + MaxPool(2) + Dropout(0.3)
         ↓  Output: (128, 16, 16)
Conv Block 4: Conv(128→256, 3×3) + BN + ReLU + Conv(256→256) + BN + ReLU + MaxPool(2) + Dropout(0.3)
         ↓  Output: (256, 8, 8)
Conv Block 5: Conv(256→512, 3×3) + BN + ReLU + Conv(512→512) + BN + ReLU + MaxPool(2) + Dropout(0.3)
         ↓  Output: (512, 4, 4)
Global Average Pooling
         ↓  Output: (512,)
FC(512→256) + ReLU + Dropout(0.3)
         ↓
FC(256→128) + ReLU + Dropout(0.3)
         ↓
FC(128→3)
         ↓
L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### ResNet18 Transfer Learning (`resnet`) - ResNetLight

```
Pretrained ResNet18 (ImageNet weights)
         ↓
Modified Conv1: Conv(1→64, 7×7) for grayscale input
  - Initialized with mean of RGB pretrained weights
         ↓
ResNet18 Feature Extractor (frozen or fine-tuned)
         ↓
Global Average Pooling → 512 features
         ↓
Dropout(0.3)
         ↓
FC(512→256) + ReLU + Dropout(0.3)
         ↓
FC(256→3)
         ↓
L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### EfficientNet-B0 (`efficientnet`) - EfficientNetLight

```
Pretrained EfficientNet-B0 (ImageNet weights)
         ↓
Modified First Conv: Conv(1→32, 3×3) for grayscale
         ↓
EfficientNet-B0 Feature Extractor
         ↓
Global Average Pooling → 1280 features
         ↓
Dropout(0.3)
         ↓
FC(1280→256) + ReLU + Dropout(0.3)
         ↓
FC(256→3)
         ↓
L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### Data Augmentation Pipeline

For small datasets (960 images), augmentation is critical:

```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.3),
])
```

### Loss Functions

**CombinedLoss** (default):
```
Loss = 0.7 × CosineSimilarityLoss + 0.3 × MSELoss
```

**CosineSimilarityLoss:**
```
Loss = 1 - cos(predicted, target)
```

### Training Features

| Feature | Configuration |
|---------|--------------|
| **Optimizer** | AdamW with weight_decay=1e-4 |
| **Learning Rate** | 1e-4 (default), cosine annealing to 1e-6 |
| **Batch Size** | 32 |
| **Early Stopping** | Patience=20 epochs, min_delta=0.001 |
| **Validation Split** | 15% |
| **Max Epochs** | 100 |

### Device Support

- **CUDA** (NVIDIA GPUs) - Auto-detected
- **MPS** (Apple Silicon) - Auto-detected
- **CPU** - Fallback

---

## 7. Configuration & Parameters

### RPS Solver Parameters

**L1 Minimization:**
```python
max_ite = 1000      # Maximum iterations
tol = 1.0e-8        # Convergence tolerance
eps = 1.0e-8        # Numerical stability
```

**Sparse Bayesian Learning:**
```python
GAMMA_THR = 1e-8    # Precision matrix threshold
lambda1 = 1.0       # Coefficient regularizer
lambda2 = 1.0e-6    # Ridge regularization
max_ite = 1000
tol = 1.0e-8
```

**RPCA:**
```python
lambda_ = 1.0 / np.sqrt(max(m, n))
mu = 1.25 / sigma_max
mu_bar = mu * 1e7
rho = 1.5           # Scaling factor
sv = 10             # SVD truncation
max_ite = 1000
tol = 1.0e-6
```

### Light Direction Predictor Configuration

**ML Models:**
```python
img_size = 64           # Image resize dimension
n_pca = 64              # PCA components
normalize_images = True # Image normalization
model_type = 'rf'       # Options: 'ridge', 'rf', 'gbr', 'mlp'
```

**DL Models:** [NEW]
```python
img_size = 128          # Image resize dimension (larger for DL)
model_type = 'resnet'   # Options: 'cnn', 'resnet', 'efficientnet'
epochs = 100            # Training epochs
batch_size = 32         # Batch size
lr = 1e-4               # Learning rate
use_augmentation = True # Data augmentation
```

---

## 8. Entry Points & Usage

### Command Line Interface [NEW]

```bash
# Compare all models (ML + DL)
python light_direction_predictor.py --mode compare_all --epochs 50

# Compare only ML models (no PyTorch required)
python light_direction_predictor.py --mode compare_ml

# Compare only DL models
python light_direction_predictor.py --mode compare_dl --epochs 50

# Train a specific model
python light_direction_predictor.py --mode train --model resnet --epochs 100 --output ./model.pkl

# Available model options: ridge, rf, gbr, mlp, cnn, resnet, efficientnet
```

### Main Script: `demo.py`

```python
from rps import RPS

# Basic workflow
rps = RPS()
rps.load_mask(filename='./data/buddha/mask.png')
rps.load_lighttxt(filename='./data/buddha/light_directions.txt')
rps.load_npyimages(foldername='./data/buddha/buddhaPNG_npy/')
rps.solve(RPS.L2_SOLVER)
rps.save_normalmap(filename="./est_normal")
```

### ML Model Training

```python
from light_direction_predictor import LightDirectionPredictor

# Train ML model (Random Forest)
predictor = LightDirectionPredictor(img_size=64, n_pca=64)
X, Y, groups, folders = predictor.load_training_data('./data/training/')
predictor.train(X, Y, model_type='rf')
predictor.save('./light_predictor_rf.pkl')
```

### DL Model Training [NEW]

```python
from light_direction_predictor import LightDirectionPredictor
import glob
import os

predictor = LightDirectionPredictor()
X, Y, groups, folders = predictor.load_training_data('./data/training/')

# Collect image paths for DL models
img_paths = []
for folder in folders:
    imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
    imgs = [f for f in imgs if 'mask' not in f.lower() and 'normal' not in f.lower()]
    img_paths.extend(imgs[:96])  # Match with light directions

# Train ResNet18 model
predictor.train(X, Y, model_type='resnet', img_paths=img_paths, epochs=100)
predictor.save('./light_predictor_resnet.pkl')
```

### Model Comparison [NEW]

```python
from light_direction_predictor import compare_all_models, compare_ml_models, compare_dl_models

# Compare all 7 models
best_model, results = compare_all_models('./data/training/', include_dl=True, dl_epochs=50)

# Compare only ML models
best_ml, ml_results = compare_ml_models('./data/training/')

# Compare only DL models
best_dl, dl_results = compare_dl_models('./data/training/', epochs=50)
```

### Inference

```python
from light_direction_predictor import LightDirectionPredictor

# Load trained model (works for both ML and DL)
predictor = LightDirectionPredictor()
predictor.load('./light_predictor_resnet.pkl')

# Predict light directions
lights = predictor.predict_from_folder('./my_images/', output_file='predicted_lights.txt')

# Use with RPS
from rps import RPS
rps = RPS()
rps.load_lighttxt('predicted_lights.txt')
rps.load_images('./my_images/')
rps.solve(RPS.RPCA_SOLVER)
```

---

## 9. Dependencies & Frameworks

### Core Dependencies

| Library | Purpose | Required |
|---------|---------|----------|
| `numpy` | Matrix operations, linear algebra | Yes |
| `scipy` | ndimage filters, numerical operations | Yes |
| `opencv-cv2` | Image I/O, visualization | Yes |
| `scikit-learn` | ML models, preprocessing (PCA, StandardScaler) | Yes |
| `Pillow` | Image loading and resizing | Yes |
| `pickle` | Model serialization (built-in) | Yes |

### Deep Learning Dependencies [NEW]

| Library | Purpose | Required |
|---------|---------|----------|
| `torch` | PyTorch deep learning framework | For DL models |
| `torchvision` | Pretrained models, transforms | For DL models |

### Installation

```bash
# Basic installation (ML models only)
pip install numpy scipy opencv-python scikit-learn pillow

# Full installation (ML + DL models)
pip install numpy scipy opencv-python scikit-learn pillow torch torchvision
```

### Key Functions Used

**NumPy/SciPy:**
- `np.linalg.lstsq()` - Least squares solver
- `np.linalg.svd()` - Singular value decomposition
- `scipy.ndimage.sobel()` - Gradient computation
- `scipy.ndimage.gaussian_filter()` - Smoothing

**PyTorch:** [NEW]
- `torch.nn.Conv2d` - Convolutional layers
- `torch.nn.BatchNorm2d` - Batch normalization
- `torchvision.models.resnet18` - Pretrained ResNet
- `torchvision.models.efficientnet_b0` - Pretrained EfficientNet
- `torch.optim.AdamW` - Optimizer with weight decay

---

## 10. Testing Structure

### Test Files

| File | Lines | Purpose |
|------|-------|---------|
| `test.py` | 30 | Basic functionality smoke tests |
| `rps_integration.py` | 493 | Integration tests and pipeline validation |

### Cross-Validation

**Leave-One-Object-Out (LOGO):**
- Tests generalization to unseen objects
- Evaluates on 10 objects independently
- Reports per-object and overall errors

**Usage:**
```python
errors = predictor.cross_validate(X, Y, groups, model_type='resnet', img_paths=img_paths)
```

### Datasets for Validation

**Training (DiLiGenT-like):**
- 10 objects × 96 images = 960 samples
- Known light directions for each

**Testing (Pikachu dataset):**
- 5 objects: battery, bootao, bootao2, bootao3, dimoo
- Used for real-world validation

---

## 11. Data Handling & Formats

### Input Formats

**Images:**
| Format | Description |
|--------|-------------|
| PNG | Standard 8-bit grayscale or RGB (612×512 typical) |
| NPY | NumPy arrays (preprocessed, faster loading) |

**Light Directions (text file):**
```
-0.0635 -0.4317 0.8998    # Image 1
-0.0629 -0.3178 0.9461    # Image 2
...                        # 96 rows (one per image)
```
Format: f × 3 matrix where f = number of images

**Mask Image:**
- PNG image where non-zero = foreground, zero = background

### Output Formats

**Normal Map:**
- **NPY:** (height × width × 3) array
- **Range:** [-1, 1] (unit normals)
- **Display:** Rescaled to [0, 1] as RGB: `(N + 1) / 2`

**Trained Models:**
- **ML Models:** `.pkl` (pickle format)
- **DL Models:** `.pkl` (metadata) + `.pt` (PyTorch state dict)

### Data Structures

| Matrix | Shape | Description |
|--------|-------|-------------|
| **M** (Measurement) | p × f | p = total pixels, f = number of images |
| **L** (Light) | 3 × f | 3D direction vectors for each image |
| **N** (Normal) | p × 3 | 3D surface normal for each pixel |

---

## 12. Workflow Examples

### Example 1: Standard RPS on Known Data

```python
from rps import RPS

rps = RPS()
rps.load_lighttxt('./data/buddha/light_directions.txt')
rps.load_npyimages('./data/buddha/buddhaPNG_npy/')
rps.load_mask('./data/buddha/mask.png')
rps.solve(RPS.RPCA_SOLVER)  # Most robust
rps.save_normalmap('./output')
```

### Example 2: ML-Based Light Prediction + RPS

```python
from light_direction_predictor import LightDirectionPredictor
from rps_integration import run_rps_pipeline

# Load pre-trained ML model
predictor = LightDirectionPredictor()
predictor.load('./light_predictor_v2.pkl')

# Predict on new images
lights = predictor.predict_from_folder('./my_images/', output_file='lights.txt')

# Run RPS
rps = run_rps_pipeline('./my_images/', 'lights.txt', 'mask.png')
```

### Example 3: DL-Based Light Prediction + RPS [NEW]

```python
from light_direction_predictor import LightDirectionPredictor
import glob

# Train ResNet model
predictor = LightDirectionPredictor()
X, Y, groups, folders = predictor.load_training_data('./data/training/')

img_paths = []
for folder in folders:
    imgs = sorted(glob.glob(f"{folder}/*.png"))
    imgs = [f for f in imgs if 'mask' not in f.lower()]
    img_paths.extend(imgs[:96])

predictor.train(X, Y, model_type='resnet', img_paths=img_paths, epochs=100)
predictor.save('./model_resnet.pkl')

# Inference on new data
lights = predictor.predict_from_folder('./new_images/', output_file='predicted_lights.txt')
```

### Example 4: Compare All Models [NEW]

```python
from light_direction_predictor import compare_all_models

# Run comprehensive comparison
best_model, results = compare_all_models('./data/training/', include_dl=True, dl_epochs=50)

print(f"Best model: {best_model}")
print("All results:")
for model, error in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {model}: {error:.2f}°")
```

---

## 13. Key Innovations & Robust Features

### Robustness Mechanisms (RPS)

| Mechanism | Description |
|-----------|-------------|
| **L1 Norm** | Mitigates effect of sparse outliers (shadows, specularities) |
| **Sparse Bayesian Learning** | Probabilistic outlier handling with learned uncertainty |
| **RPCA** | Globally separates diffuse (low-rank) from non-diffuse (sparse) components |
| **Masking** | Excludes background/invalid regions from computation |

### ML/DL Integration

| Feature | Description |
|---------|-------------|
| **Physics-informed features** | 42 features designed for lighting, not object-specific |
| **Multi-scale analysis** | Captures features at different resolutions |
| **Image normalization** | Reduces object-specific intensity bias |
| **Transfer learning** [NEW] | Leverages ImageNet pretrained weights |
| **Data augmentation** [NEW] | Effectively increases dataset size |
| **Ensemble methods** | Random forests leverage non-linear patterns |

### Deep Learning Advantages [NEW]

| Advantage | Description |
|-----------|-------------|
| **End-to-end learning** | Learns optimal features automatically |
| **Transfer learning** | Uses knowledge from 1M+ ImageNet images |
| **Better generalization** | Expected 30-50% error reduction vs ML |
| **GPU acceleration** | Fast training on CUDA/MPS devices |

### Performance Optimizations

| Optimization | Description |
|--------------|-------------|
| **Multicore processing** | L1 and SBL solvers use multiprocessing |
| **NumPy vectorization** | Efficient matrix operations throughout |
| **NPY format** | Faster image I/O compared to PNG reading |
| **GPU support** [NEW] | DL models leverage CUDA/MPS acceleration |
| **Early stopping** [NEW] | Prevents overfitting in DL training |

---

## Project Status

**Current Branch:** `thanhn/adding_dl_model`

**Recent Updates (February 2025):**
- Added deep learning models (CNN, ResNet18, EfficientNet)
- Integrated PyTorch-based training pipeline
- Added data augmentation for small datasets
- Updated CLI with model comparison tools
- Unified API for ML and DL models

**Available Models:**
- ML: `ridge`, `rf`, `gbr`, `mlp`
- DL: `cnn`, `resnet`, `efficientnet`

**Key Contributions:**
- Original RPS implementation by Yasuyuki Matsushita (Osaka University)
- ML light direction prediction module
- Deep learning integration with transfer learning [NEW]
- End-to-end pipeline for calibration-free photometric stereo

---

## Summary

This codebase implements state-of-the-art robust photometric stereo with an innovative ML/DL-based extension to predict light directions without manual calibration. The system now supports:

1. **6 RPS solver algorithms** for robust normal estimation
2. **4 classical ML models** for light direction prediction
3. **3 deep learning models** [NEW] with transfer learning for improved accuracy
4. **Unified API** for seamless model switching and comparison
5. **Comprehensive evaluation** with leave-one-object-out cross-validation

The modular design enables researchers to compare multiple algorithms while maintaining clean, maintainable code.
