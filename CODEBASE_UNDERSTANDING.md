# Comprehensive Codebase Analysis: RobustPhotometricStereo

A state-of-the-art robust photometric stereo implementation with multiple solver algorithms and an ML/DL-based light direction predictor.

**Total Lines of Code:** ~2,500+ lines (excluding dependencies)

**Last Updated:** March 2025

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Components & Responsibilities](#2-core-components--responsibilities)
3. [Data Flow](#3-data-flow)
4. [Key Algorithms](#4-key-algorithms)
5. [ML/DL Models: Light Direction Predictor](#5-mldl-models-light-direction-predictor)
6. [Deep Learning Architecture Details (v3)](#6-deep-learning-architecture-details-v3)
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
│   ├── ps_utils.py               # Utility functions (163 lines) [renamed from psutil.py]
│   ├── demo.py                   # Demonstration script (54 lines)
│   ├── test.py                   # Test script (30 lines)
│   ├── pngToNpy.py               # Image conversion utility (56 lines)
│   └── README.md                 # Project documentation
│
├── light_direction_predictor/    # ML/DL module
│   ├── light_direction_predictor.py   # Main predictor class (960+ lines)
│   ├── dl_models.py                   # Deep learning models v3 (600+ lines)
│   ├── rps_integration.py             # Integration pipeline (493 lines)
│   ├── __init__.py                    # Package init with exports
│   ├── light_predictor_v2.pkl         # Trained ML model (Random Forest)
│   ├── light_predictor_ml_v3.pkl      # Best ML model (from train_best_ml)
│   ├── light_predictor_dl_v3.pkl      # Best DL model (from train_best_dl)
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

### C. Utility Functions (`ps_utils.py`)

**Purpose:** I/O, preprocessing, visualization, evaluation

> **Note:** Renamed from `psutil.py` to `ps_utils.py` to avoid naming conflict with the system `psutil` package (used by sklearn/joblib for CPU detection).

| Category | Functions |
|----------|-----------|
| Data Loading | `load_lighttxt()`, `load_lightnpy()`, `load_image()`, `load_images()`, `load_npyimages()` |
| Visualization | `disp_normalmap()`, `save_normalmap_as_npy()`, `load_normalmap_from_npy()` |
| Evaluation | `evaluate_angular_error()` |

### D. Light Direction Predictor (`light_direction_predictor.py`)

**Purpose:** ML/DL-based prediction of light directions from images

**Key Components:**
- `LightDirectionPredictor` - Main class supporting 7 model types
- `train_best_ml()` - Pipeline: compare ML models → train best → save `.pkl`
- `train_best_dl()` - Pipeline: compare DL models → train best → save `.pkl` + `.pt`
- `compare_all_models()` - Compare ML + DL models
- `compare_ml_models()` - Compare only classical ML models
- `compare_dl_models()` - Compare only deep learning models

### E. Deep Learning Models (`dl_models.py`) — v3 Gradient Input

**Purpose:** PyTorch-based CNN models with 3-channel gradient input for object-invariant light direction prediction

**Key Components:**
- `LightCNN` - Custom 4-block CNN (3-channel gradient input)
- `ResNetLight` - ResNet18 transfer learning (no conv1 modification needed)
- `EfficientNetLight` - EfficientNet-B0 transfer learning (no conv1 modification needed)
- `DeepLightPredictor` - Training/inference wrapper with 2-phase training + mixup
- `LightDirectionDataset` - PyTorch Dataset with gradient computation + augmentation
- `AngularLoss` / `CombinedLoss` - Loss functions
- `EarlyStopping` - Early stopping with best model restoration
- `mixup_data()` - Mixup augmentation function

**v3 Key Innovation:** 3-channel input `[normalized_img, sobel_x, sobel_y]` makes predictions object-invariant by focusing on shading patterns rather than object identity.

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
│  Option B: ML/DL Prediction                                    │
│    - LightDirectionPredictor.predict_from_folder()             │
│    - ML: ridge, rf, gbr, mlp (uses PCA + physics features)    │
│    - DL: cnn, resnet, efficientnet (uses gradient input)       │
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

#### Deep Learning Models (PyTorch-based) — v3 Gradient Input

| Model | Code | Description | Input | Target Error |
|-------|------|-------------|-------|-------------|
| Custom CNN | `cnn` | 4-block CNN with gradient input | 3ch × 224 | ~12-18° |
| ResNet18 | `resnet` | Transfer learning, 3-channel gradient | 3ch × 224 | ~10-15° |
| EfficientNet-B0 | `efficientnet` | Efficient architecture, gradient input | 3ch × 224 | ~10-15° |

**v3 Improvement:** DL models now use 3-channel gradient input `[img, sobel_x, sobel_y]` instead of raw grayscale pixels. This makes predictions object-invariant and should significantly reduce CV error from ~24° (v1) toward ~15° or better.

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

### DL Input Pipeline (v3 Gradient Input)

```
┌─────────────────────────────────────────────────────────────────┐
│                  1. IMAGE LOADING                                │
├─────────────────────────────────────────────────────────────────┤
│  - Load as grayscale                                            │
│  - Resize to 224×224 (or 256 for random crop)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            2. AUGMENTATION (training only)                       │
├─────────────────────────────────────────────────────────────────┤
│  - Random crop: 256 → 224                                      │
│  - Horizontal flip + label correction (negate light.x)         │
│  - Brightness/contrast jitter (×0.8-1.2)                       │
│  - Random erasing (p=0.3)                                      │
│  - Mixup: blend images + labels (α=0.4, p=0.5 per batch)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            3. GRADIENT COMPUTATION                               │
├─────────────────────────────────────────────────────────────────┤
│  - Per-image normalization: (img - mean) / std                 │
│  - Sobel X gradient (horizontal edges)                         │
│  - Sobel Y gradient (vertical edges)                           │
│  - Normalize gradients to [-1, 1]                              │
│  - Clip image to [-1, 1]                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                4. 3-CHANNEL TENSOR                               │
├─────────────────────────────────────────────────────────────────┤
│  Stack: [normalized_img, sobel_x, sobel_y]                     │
│  Shape: (3, 224, 224) — matches RGB for pretrained models      │
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

## 6. Deep Learning Architecture Details (v3)

### Core Innovation: 3-Channel Gradient Input

**Problem (v1):** DL models with raw grayscale input learned object identity instead of shading patterns, achieving ~24° CV error — worse than ML models (~10-15°).

**Solution (v3):** Feed `[normalized_img, sobel_x, sobel_y]` as 3 channels:
- **Object-invariant**: Gradients encode shading, not texture
- **Physics-motivated**: For Lambertian shading I=N·L, gradients of I relate directly to L
- **Natural 3-channel**: Pretrained models expect 3 channels — no conv1 modification hack needed
- **Same insight as ML**: The best ML features are gradient-based

### Custom CNN (`cnn`) - LightCNN

```
Input: (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
         ↓
Conv Block 1: Conv(3→32, 3×3) + BN + ReLU + Conv(32→32) + BN + ReLU + MaxPool(2) + Dropout2d(0.25)
         ↓  Output: (32, 112, 112)
Conv Block 2: Conv(32→64, 3×3) + BN + ReLU + Conv(64→64) + BN + ReLU + MaxPool(2) + Dropout2d(0.25)
         ↓  Output: (64, 56, 56)
Conv Block 3: Conv(64→128, 3×3) + BN + ReLU + Conv(128→128) + BN + ReLU + MaxPool(2) + Dropout2d(0.375)
         ↓  Output: (128, 28, 28)
Conv Block 4: Conv(128→256, 3×3) + BN + ReLU + Conv(256→256) + BN + ReLU + MaxPool(2) + Dropout2d(0.5)
         ↓  Output: (256, 14, 14)
Global Average Pooling
         ↓  Output: (256,)
FC(256→128) + BatchNorm1d + ReLU + Dropout(0.5)
         ↓
FC(128→3)
         ↓
L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### ResNet18 Transfer Learning (`resnet`) - ResNetLight

```
Input: (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
         ↓
Pretrained ResNet18 (ImageNet weights)
  - conv1(3→64, 7×7): Uses pretrained 3-channel weights directly!
  - NO conv1 modification needed (unlike v1 grayscale hack)
         ↓
ResNet18 Feature Extractor
  - Phase 1: Freeze layer1-4, train conv1+bn1+head (adapt to gradient input)
  - Phase 2: Unfreeze all, fine-tune with differential LR
         ↓
Global Average Pooling → 512 features
         ↓
BatchNorm1d(512) + Dropout(0.5)
         ↓
FC(512→128) + BatchNorm1d + ReLU + Dropout(0.25)
         ↓
FC(128→3) → L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### EfficientNet-B0 (`efficientnet`) - EfficientNetLight

```
Input: (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
         ↓
Pretrained EfficientNet-B0 (ImageNet weights)
  - First conv(3→32, 3×3): Uses pretrained weights directly!
  - Phase 1: Freeze features[1:], train features[0]+classifier
  - Phase 2: Unfreeze all with differential LR
         ↓
Global Average Pooling → 1280 features
         ↓
BatchNorm1d(1280) + Dropout(0.5)
         ↓
FC(1280→128) + BatchNorm1d + ReLU + Dropout(0.25)
         ↓
FC(128→3) → L2 Normalize
         ↓
Output: 3D Light Direction (unit vector)
```

### 2-Phase Transfer Learning Strategy

| Phase | Frozen | Trainable | Learning Rate | Purpose |
|-------|--------|-----------|---------------|---------|
| Phase 1 | layer1-4 (ResNet) / features[1:] (EfficientNet) | conv1/bn1 + head | lr × 10 | Adapt first conv to gradient input, train head |
| Phase 2 | Nothing | All | backbone: lr×0.1, head: lr | Fine-tune full network |

### Data Augmentation Pipeline (v3)

```python
# In LightDirectionDataset.__getitem__():

# 1. Random crop (256 → 224)
img = resize(img, 256)
img = random_crop(img, 224)

# 2. Horizontal flip WITH label correction
if random() > 0.5:
    img = mirror(img)
    light[0] = -light[0]  # Correct x-component!

# 3. Brightness/contrast jitter
img = adjust_brightness_contrast(img, 0.8-1.2)

# 4. Compute gradients (AFTER spatial augmentation)
img_norm = (img - mean) / std
grad_x = sobel(img_norm, axis=1)
grad_y = sobel(img_norm, axis=0)
tensor = stack([img_norm, grad_x, grad_y])

# 5. Random erasing (on tensor)
if random() > 0.7:
    tensor[:, ry:ry+rh, rx:rx+rw] = 0

# 6. Mixup (in training loop, per batch)
if random() > 0.5:
    lam = Beta(0.4, 0.4)
    mixed_x = lam * x + (1-lam) * x[perm]
    mixed_y = normalize(lam * y + (1-lam) * y[perm])
```

### Loss Functions (v3)

**CombinedLoss** (default):
```
Loss = 0.7 × AngularLoss + 0.3 × MSELoss

AngularLoss = mean(arccos(clamp(cos_sim(pred, target), -1+ε, 1-ε)))
```

**Why Angular Loss over Cosine Loss:**
- Cosine loss = `1 - cos(θ)` → insensitive near θ=0
- Angular loss = `arccos(cos(θ))` = θ → linear in the actual metric we evaluate

### Training Configuration (v3)

| Feature | Value |
|---------|-------|
| **Image size** | 224×224 |
| **Input channels** | 3: [img, grad_x, grad_y] |
| **Optimizer** | AdamW, weight_decay=5e-4 |
| **Learning Rate** | 1e-4, cosine annealing to 1e-6 |
| **Gradient Clipping** | max_norm=1.0 |
| **Early Stopping** | Patience 8-25 epochs (varies by context) |
| **Dropout** | 0.5 (head), progressive in conv blocks |
| **Batch Size** | 32 |
| **Mixup** | α=0.4, applied 50% of batches |
| **Validation Split** | 15% (training), 12% (CV folds) |

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

**DL Models (v3):**
```python
img_size = 224           # Image resize dimension (for pretrained models)
model_type = 'resnet'    # Options: 'cnn', 'resnet', 'efficientnet'
epochs = 150             # Training epochs
batch_size = 32          # Batch size
lr = 1e-4                # Learning rate
dropout = 0.5            # Dropout rate
weight_decay = 5e-4      # L2 regularization
use_augmentation = True  # Data augmentation + mixup
# Input: 3 channels [normalized_img, sobel_x, sobel_y]
```

---

## 8. Entry Points & Usage

### Command Line Interface

```bash
# ==================== RECOMMENDED PIPELINES ====================

# ML Pipeline: Compare all ML → train best → save light_predictor_ml_v3.pkl
python light_direction_predictor.py --mode train_best_ml

# DL Pipeline: Compare all DL → train best → save light_predictor_dl_v3.pkl
python light_direction_predictor.py --mode train_best_dl --epochs 150

# ==================== COMPARISON MODES ====================

# Compare all models (ML + DL)
python light_direction_predictor.py --mode compare_all --epochs 50

# Compare only ML models (no PyTorch required)
python light_direction_predictor.py --mode compare_ml

# Compare only DL models
python light_direction_predictor.py --mode compare_dl --epochs 50

# ==================== TRAIN SPECIFIC MODEL ====================

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

### ML Pipeline

```python
from light_direction_predictor import train_best_ml

# Compare all ML models, train best, save
predictor, best_model, results = train_best_ml('./data/training/', './light_predictor_ml_v3.pkl')
```

### DL Pipeline

```python
from light_direction_predictor import train_best_dl

# Compare all DL models, train best, save
predictor, best_model, results = train_best_dl('./data/training/', './light_predictor_dl_v3.pkl', epochs=150)
```

### Model Comparison

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
predictor.load('./light_predictor_dl_v3.pkl')

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
| `scipy` | ndimage filters (Sobel), numerical operations | Yes |
| `opencv-cv2` | Image I/O, visualization | Yes |
| `scikit-learn` | ML models, preprocessing (PCA, StandardScaler) | Yes |
| `Pillow` | Image loading and resizing | Yes |
| `pickle` | Model serialization (built-in) | Yes |

### Deep Learning Dependencies

| Library | Purpose | Required |
|---------|---------|----------|
| `torch` | PyTorch deep learning framework | For DL models |
| `torchvision` | Pretrained models (ResNet18, EfficientNet-B0) | For DL models |

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
- `scipy.ndimage.sobel()` - Gradient computation (used in both ML features AND DL input)
- `scipy.ndimage.gaussian_filter()` - Smoothing

**PyTorch:**
- `torch.nn.Conv2d` - Convolutional layers
- `torch.nn.BatchNorm2d` / `BatchNorm1d` - Batch normalization
- `torchvision.models.resnet18` - Pretrained ResNet (3-channel input)
- `torchvision.models.efficientnet_b0` - Pretrained EfficientNet (3-channel input)
- `torch.optim.AdamW` - Optimizer with weight decay
- `torch.nn.utils.clip_grad_norm_` - Gradient clipping

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
- DL v3: Each fold uses 2-phase training + early stopping + mixup

**Usage:**
```python
# ML
errors = predictor.cross_validate(X, Y, groups, model_type='rf')

# DL
errors = predictor.cross_validate(X, Y, groups, model_type='resnet', img_paths=img_paths, epochs=150)
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

### Example 2: ML Pipeline → RPS

```bash
# Step 1: Train best ML model
python light_direction_predictor.py --mode train_best_ml
```

```python
# Step 2: Use trained model for prediction + RPS
from light_direction_predictor import LightDirectionPredictor
from rps import RPS

predictor = LightDirectionPredictor()
predictor.load('./light_predictor_ml_v3.pkl')
lights = predictor.predict_from_folder('./my_images/', output_file='lights.txt')

rps = RPS()
rps.load_lighttxt('lights.txt')
rps.load_images('./my_images/')
rps.solve(RPS.RPCA_SOLVER)
```

### Example 3: DL Pipeline → RPS

```bash
# Step 1: Train best DL model (v3 gradient input)
python light_direction_predictor.py --mode train_best_dl --epochs 150
```

```python
# Step 2: Use trained model for prediction + RPS
from light_direction_predictor import LightDirectionPredictor

predictor = LightDirectionPredictor()
predictor.load('./light_predictor_dl_v3.pkl')
lights = predictor.predict_from_folder('./new_images/', output_file='predicted_lights.txt')
```

### Example 4: Compare All Models

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
| **Physics-informed features** | 42 features designed for lighting, not object-specific (ML) |
| **3-channel gradient input** | Object-invariant DL input: [img, sobel_x, sobel_y] (DL v3) |
| **Mixup augmentation** | Blends images from different objects to prevent memorization (DL v3) |
| **Horizontal flip + label correction** | Doubles data with correct light direction adjustment (DL v3) |
| **Transfer learning** | Leverages ImageNet pretrained weights (no conv1 hack in v3) |
| **2-phase training** | Freeze backbone → fine-tune all with differential LR (DL v3) |
| **Multi-scale analysis** | Captures features at different resolutions (ML) |
| **Image normalization** | Reduces object-specific intensity bias (both ML and DL) |
| **Ensemble methods** | Random forests leverage non-linear patterns (ML) |

### Deep Learning v3 Advantages

| Advantage | v1 (Grayscale) | v3 (Gradient Input) |
|-----------|----------------|---------------------|
| **Input** | 1ch grayscale | 3ch [img, grad_x, grad_y] |
| **Object invariance** | Low (learns texture) | High (learns shading) |
| **Conv1 modification** | Required (1ch hack) | Not needed (3ch native) |
| **Augmentation** | Basic (flip, rotate) | Physics-aware (flip+correction, mixup) |
| **Loss function** | Cosine similarity | Angular loss (direct metric) |
| **CV fold training** | No early stopping | 2-phase + early stopping + mixup |
| **Expected CV error** | ~24° | Target ~15° |

### Performance Optimizations

| Optimization | Description |
|--------------|-------------|
| **Multicore processing** | L1 and SBL solvers use multiprocessing |
| **NumPy vectorization** | Efficient matrix operations throughout |
| **NPY format** | Faster image I/O compared to PNG reading |
| **GPU support** | DL models leverage CUDA/MPS acceleration |
| **Early stopping** | Prevents overfitting in DL training |
| **Gradient clipping** | Prevents training instability |

---

## Project Status

**Current Branch:** `thanhn/continue_fine_tuning`

**Recent Updates (March 2025):**
- **DL v3**: 3-channel gradient input [img, sobel_x, sobel_y] for object-invariant prediction
- **Mixup augmentation** to prevent memorizing object appearances
- **Horizontal flip with label correction** (negates light x-component)
- **2-phase transfer learning** with early stopping in CV folds
- **Angular loss** replacing cosine similarity loss
- **Gradient clipping** and higher regularization (dropout=0.5, weight_decay=5e-4)
- Added `train_best_ml()` and `train_best_dl()` pipeline functions
- Renamed `psutil.py` → `ps_utils.py` to avoid system package conflict

**Previous Updates (February 2025):**
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
- Deep learning integration with transfer learning
- v3 gradient-based input for object-invariant DL prediction
- End-to-end pipeline for calibration-free photometric stereo

---

## Summary

This codebase implements state-of-the-art robust photometric stereo with an innovative ML/DL-based extension to predict light directions without manual calibration. The system now supports:

1. **6 RPS solver algorithms** for robust normal estimation
2. **4 classical ML models** for light direction prediction (~10-15° CV error)
3. **3 deep learning models (v3)** with gradient input for object-invariant prediction
4. **Unified API** for seamless model switching and comparison
5. **Two recommended pipelines**: `train_best_ml` and `train_best_dl`
6. **Comprehensive evaluation** with leave-one-object-out cross-validation

The modular design enables researchers to compare multiple algorithms while maintaining clean, maintainable code. The v3 DL approach addresses the fundamental limitation of previous DL versions (learning object identity instead of shading patterns) through gradient-based input, mixup augmentation, and physics-aware training.
