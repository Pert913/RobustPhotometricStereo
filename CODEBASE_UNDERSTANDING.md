# Comprehensive Codebase Analysis: RobustPhotometricStereo

A state-of-the-art robust photometric stereo implementation with multiple solver algorithms and an ML-based light direction predictor.

**Total Lines of Code:** ~1,654 lines (excluding dependencies)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Components & Responsibilities](#2-core-components--responsibilities)
3. [Data Flow](#3-data-flow)
4. [Key Algorithms](#4-key-algorithms)
5. [ML/DL Models: Light Direction Predictor](#5-mldl-models-light-direction-predictor)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Entry Points & Usage](#7-entry-points--usage)
8. [Dependencies & Frameworks](#8-dependencies--frameworks)
9. [Testing Structure](#9-testing-structure)
10. [Data Handling & Formats](#10-data-handling--formats)
11. [Workflow Examples](#11-workflow-examples)
12. [Key Innovations & Robust Features](#12-key-innovations--robust-features)

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
├── light_direction_predictor/    # ML/DL module (new feature)
│   ├── light_direction_predictor.py   # Main ML model (483 lines)
│   ├── rps_integration.py             # Integration pipeline (382 lines)
│   ├── __init__.py                    # Package init (8 lines)
│   ├── data/
│   │   ├── training/             # Training data (10 objects)
│   │   └── testing/              # Test data (Pikachu dataset)
│   └── README.md                 # ML module documentation
├── data/                         # Dataset directory
│   ├── buddha/                   # Buddha statue dataset
│   │   ├── buddhaPNG/            # Raw PNG images (96 images)
│   │   ├── buddhaPNG_npy/        # Preprocessed NPY images
│   │   └── light_directions.txt  # Known light directions
│   ├── bunny/                    # Bunny dataset
│   │   ├── bunny_lambert/        # Lambertian (with shadow)
│   │   ├── bunny_lambert_noshadow/  # Lambertian (no shadow)
│   │   └── bunny_specular/       # Specular reflectance
│   └── cat/                      # Cat dataset
└── Output visualizations
    ├── L1_specular_normal.png
    ├── L2_specular_normal.png
    ├── RPCA_specular_normal.png
    ├── SBL_specular_normal.png
    └── est_normal.npy            # Estimated normal map
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
│                      3. LIGHT LOADING                           │
├─────────────────────────────────────────────────────────────────┤
│  rps.load_lighttxt("light_directions.txt")                     │
│         ↓                                                       │
│  Light matrix L (3 × f)                                        │
│  [light_x light_y light_z]ᵀ for each image                     │
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
│                    6. POST-PROCESSING                           │
├─────────────────────────────────────────────────────────────────┤
│  - L2 normalization of normals                                 │
│  - Apply mask (set background to zero)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        7. OUTPUT                                │
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

Parameters:
  - λ = 1/√(max(m,n))  [control sparsity]
  - ρ = 1.5            [dual step scaling]
  - μ_bar = μ × 1e7    [maximum penalty]
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

### Architecture

**LightDirectionPredictor Class** (`light_direction_predictor.py`, 483 lines)

### Feature Extraction Pipeline

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

### Machine Learning Models

| Model | Configuration | Typical Error | Notes |
|-------|--------------|---------------|-------|
| **Ridge Regression** | α=10.0 | ~15-20° | Fast baseline |
| **Random Forest** | 200 trees, max_depth=15 | ~10-15° | **Current best** |
| **Gradient Boosting** | 100 estimators, max_depth=5 | ~8-12° | Best generalization |
| **MLP Neural Network** | 512-256-128 hidden layers | ~10-15° | Deep learning approach |

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

### Integration Pipeline (`rps_integration.py`)

| Function | Purpose |
|----------|---------|
| `train_light_predictor()` | Train from scratch or load existing |
| `predict_light_directions()` | Predict for new images |
| `run_rps_pipeline()` | Execute RPS with predicted lights |
| `evaluate_results()` | Compare with ground truth |

---

## 6. Configuration & Parameters

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

```python
img_size = 64           # Image resize dimension
n_pca = 64              # PCA components
normalize_images = True # Image normalization

# Model parameters
model_type = 'rf'       # Options: 'ridge', 'rf', 'gbr', 'mlp'
```

---

## 7. Entry Points & Usage

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

### ML Integration: `rps_integration.py`

```python
from light_direction_predictor import LightDirectionPredictor

# Train predictor
predictor = LightDirectionPredictor(img_size=64, n_pca=64)
X, Y, groups = predictor.load_training_data('./data/training/')
predictor.train(X, Y, model_type='rf')
predictor.save('./light_predictor_v2.pkl')

# Predict and run RPS
lights = predictor.predict_from_folder('./test_images/')
rps = run_rps_pipeline(images_folder, lights_path, mask_path)
```

### Data Conversion: `pngToNpy.py`

Convert PNG images to NumPy arrays with proper naming for faster loading.

---

## 8. Dependencies & Frameworks

### External Libraries

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations, linear algebra |
| `scipy` | ndimage filters, numerical operations |
| `opencv-cv2` | Image I/O, visualization |
| `scikit-learn` | ML models, preprocessing (PCA, StandardScaler) |
| `scikit-image` | Image processing utilities |
| `Pillow` | Image loading and resizing |
| `pickle` | Model serialization (built-in) |

### Key NumPy/SciPy Functions Used

- `np.linalg.lstsq()` - Least squares solver
- `np.linalg.svd()` - Singular value decomposition
- `np.linalg.solve()` - Linear system solver
- `scipy.ndimage.sobel()` - Gradient computation
- `scipy.ndimage.gaussian_filter()` - Smoothing

---

## 9. Testing Structure

### Test Files

| File | Lines | Purpose |
|------|-------|---------|
| `test.py` | 30 | Basic functionality smoke tests |

### Cross-Validation in Light Predictor

- Leave-One-Object-Out (LOGO) strategy
- Evaluates on 10 objects independently
- Outputs per-object and overall errors

### Datasets for Validation

**Training (DiLiGenT-like):**
- 10 objects × 96 images = 960 samples
- Known light directions for each

**Testing:**
- Pikachu dataset (custom data)
- Prepared for light direction prediction evaluation

---

## 10. Data Handling & Formats

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
- Used to ignore pixels (background, invalid regions)

### Output Formats

**Normal Map:**
- **NPY:** (height × width × 3) array
- **Range:** [-1, 1] (unit normals)
- **Display:** Rescaled to [0, 1] as RGB: `(N + 1) / 2`

### Data Structures

| Matrix | Shape | Description |
|--------|-------|-------------|
| **M** (Measurement) | p × f | p = total pixels, f = number of images |
| **L** (Light) | 3 × f | 3D direction vectors for each image |
| **N** (Normal) | p × 3 | 3D surface normal for each pixel |

---

## 11. Workflow Examples

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

### Example 2: Light Direction Prediction + RPS

```python
from light_direction_predictor import LightDirectionPredictor
from rps_integration import run_rps_pipeline

# Load pre-trained model
predictor = LightDirectionPredictor()
predictor.load('./light_predictor_v2.pkl')

# Predict on new images
lights = predictor.predict_from_folder('./my_images/')

# Run RPS
rps = run_rps_pipeline('./my_images/', lights_path, mask_path)
```

### Example 3: Train Custom Model

```python
from light_direction_predictor import LightDirectionPredictor

predictor = LightDirectionPredictor(img_size=64, n_pca=64)
X, Y, groups = predictor.load_training_data('./data/training/')
predictor.cross_validate(X, Y, groups, model_type='gbr')
predictor.train(X, Y, model_type='gbr')
predictor.save('./my_model.pkl')
```

---

## 12. Key Innovations & Robust Features

### Robustness Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **L1 Norm** | Mitigates effect of sparse outliers (shadows, specularities) |
| **Sparse Bayesian Learning** | Probabilistic outlier handling with learned uncertainty |
| **RPCA** | Globally separates diffuse (low-rank) from non-diffuse (sparse) components |
| **Masking** | Excludes background/invalid regions from computation |

### ML/DL Integration

| Feature | Description |
|---------|-------------|
| **Physics-informed features** | Designed for lighting, not object-specific |
| **Multi-scale analysis** | Captures features at different resolutions |
| **Image normalization** | Reduces object-specific intensity bias |
| **Ensemble methods** | Random forests leverage non-linear patterns |

### Performance Optimizations

| Optimization | Description |
|--------------|-------------|
| **Multicore processing** | L1 and SBL solvers use multiprocessing |
| **NumPy vectorization** | Efficient matrix operations throughout |
| **NPY format** | Faster image I/O compared to PNG reading |
| **Flexible solvers** | Choose method based on robustness/speed tradeoff |

---

## Project Status

**Current Branch:** `thanhnh/improve_ml_dl_models`

**Recent Activity:**
- Merge: ML Light Detection feature branch
- Adding README for Pikachu test data
- Bug fixes and testing data organization

**Key Contributions:**
- Original RPS implementation by Yasuyuki Matsushita (Osaka University)
- ML light direction prediction module (recent addition)
- Integration pipeline for end-to-end workflows

---

## Summary

This codebase implements state-of-the-art robust photometric stereo with an innovative ML-based extension to predict light directions without manual calibration. The modular design enables researchers to compare multiple algorithms while maintaining clean, maintainable code.
