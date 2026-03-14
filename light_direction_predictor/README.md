# ML/DL Project: Light Direction Prediction

## Goal

Remove the dependency on known light directions when collecting your own data.

---

## Quick Start

### Installation

```bash
# Basic ML models (sklearn)
pip install numpy scipy scikit-learn pillow

# Deep Learning models (optional but recommended)
pip install torch torchvision
```

### Recommended Pipelines

```bash
# Pipeline 1: Compare all ML models → train best → save
python light_direction_predictor.py --mode train_best_ml

# Pipeline 2: Compare all DL models → train best → save
python light_direction_predictor.py --mode train_best_dl --epochs 150
```

### Compare Models (without saving)

```bash
# Compare all ML + DL models
python light_direction_predictor.py --mode compare_all --epochs 50

# Compare only ML models (no PyTorch required)
python light_direction_predictor.py --mode compare_ml

# Compare only DL models
python light_direction_predictor.py --mode compare_dl --epochs 50
```

### Train a Specific Model

```bash
# Train Random Forest (ML)
python light_direction_predictor.py --mode train --model rf --output ./model_rf.pkl

# Train ResNet18 (DL)
python light_direction_predictor.py --mode train --model resnet --epochs 100 --output ./model_resnet.pkl

# Train Custom CNN (DL)
python light_direction_predictor.py --mode train --model cnn --epochs 100 --output ./model_cnn.pkl
```

---

## Available Models

### Classical ML Models (sklearn)

| Model | Code | Description | Speed | Accuracy |
|-------|------|-------------|-------|----------|
| Ridge Regression | `ridge` | Linear regression with L2 regularization | Fast | Baseline |
| Random Forest | `rf` | Ensemble of decision trees | Medium | Good |
| Gradient Boosting | `gbr` | Sequential ensemble learning | Slow | Good |
| MLP | `mlp` | Multi-layer perceptron (sklearn) | Medium | Good |

### Deep Learning Models (PyTorch) — v3 Gradient Input

| Model | Code | Description | Speed | Accuracy |
|-------|------|-------------|-------|----------|
| Custom CNN | `cnn` | 4-block CNN with gradient input | Fast | Good |
| ResNet18 | `resnet` | Transfer learning, 3-channel gradient | Medium | Best |
| EfficientNet-B0 | `efficientnet` | Efficient architecture, gradient input | Medium | Best |

**Note:** DL models require PyTorch. Install with `pip install torch torchvision`

---

## Deep Learning v3: Key Innovation — 3-Channel Gradient Input

### The Problem (v1)

DL models fed with raw grayscale pixels achieved ~24° CV error — worse than ML models (~10-15°). The reason: **models learned object identity** (texture, shape) instead of lighting patterns. They couldn't generalize to unseen objects.

### The Solution (v3)

Instead of raw grayscale pixels, DL models receive **3-channel gradient-based input**:

```
Channel 0: Normalized image     (zero mean, unit std)
Channel 1: Sobel X gradient     (horizontal edges)
Channel 2: Sobel Y gradient     (vertical edges)
```

**Why this works:**
- **Object-invariant**: Gradients encode shading patterns, not object identity
- **Physics-motivated**: For Lambertian surfaces, image gradients directly relate to light direction
- **Natural 3-channel**: Pretrained models (ResNet, EfficientNet) expect 3 channels — no conv1 hack needed
- **Same insight as ML models**: The ML features that work best are gradient-based (gradient histograms, dominant direction)

---

## Architecture Details

### Custom CNN (`cnn`)

```
Input (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
    ↓
Conv Block 1: Conv(3→32) + BN + ReLU + Conv + BN + ReLU + MaxPool + Dropout(0.25)
    ↓
Conv Block 2: Conv(32→64) + BN + ReLU + Conv + BN + ReLU + MaxPool + Dropout(0.25)
    ↓
Conv Block 3: Conv(64→128) + BN + ReLU + Conv + BN + ReLU + MaxPool + Dropout(0.375)
    ↓
Conv Block 4: Conv(128→256) + BN + ReLU + Conv + BN + ReLU + MaxPool + Dropout(0.5)
    ↓
Global Average Pooling → (256,)
    ↓
FC(256→128) + BatchNorm1d + ReLU + Dropout(0.5)
    ↓
FC(128→3) → L2 Normalize
    ↓
Output: 3D Light Direction (unit vector)
```

### ResNet18 Transfer Learning (`resnet`)

```
Input (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
    ↓
Pretrained ResNet18 (ImageNet weights)
  - conv1 uses pretrained 3-channel weights (NO modification needed!)
  - Phase 1: Freeze layers 1-4, train conv1 + bn1 + head
  - Phase 2: Unfreeze all, fine-tune with differential LR
    ↓
Global Average Pooling → (512,)
    ↓
BatchNorm1d(512) + Dropout(0.5) + FC(512→128) + BatchNorm1d + ReLU + Dropout(0.25)
    ↓
FC(128→3) → L2 Normalize
    ↓
Output: 3D Light Direction (unit vector)
```

### EfficientNet-B0 (`efficientnet`)

```
Input (3, 224, 224) — [normalized_img, sobel_x, sobel_y]
    ↓
Pretrained EfficientNet-B0 (ImageNet weights)
  - First conv uses pretrained 3-channel weights (NO modification needed!)
  - Phase 1: Freeze features[1:], train features[0] + classifier
  - Phase 2: Unfreeze all with differential LR
    ↓
Global Average Pooling → (1280,)
    ↓
BatchNorm1d(1280) + Dropout(0.5) + FC(1280→128) + BatchNorm1d + ReLU + Dropout(0.25)
    ↓
FC(128→3) → L2 Normalize
    ↓
Output: 3D Light Direction (unit vector)
```

---

## Data Augmentation (DL Models v3)

For small datasets (960 images), augmentation is critical. v3 uses **physics-aware** augmentations:

| Augmentation | Details | Why |
|-------------|---------|-----|
| **Random Crop** | Resize to 256, crop to 224 | Position invariance |
| **Horizontal Flip** | With label correction: negate `light[0]` | Doubles data correctly |
| **Brightness/Contrast Jitter** | ×0.8-1.2 | Removed by per-image normalization |
| **Random Erasing** | Zero out random rectangle (p=0.3) | Prevents learning object shape |
| **Mixup** | Blend images + labels (α=0.4) | Prevents memorizing objects |

**Important**: No random rotation (would require complex 3D light direction correction).

**Label correction for horizontal flip:**
```python
if flip:
    image = mirror(image)
    light[0] = -light[0]  # Negate x-component
```

---

## Training Features

### Loss Function

Combined Angular + MSE loss:
```
Loss = 0.7 × AngularLoss + 0.3 × MSELoss

AngularLoss = mean(arccos(cos_sim(pred, target)))  # in radians
```

### 2-Phase Transfer Learning (ResNet, EfficientNet)

| Phase | Trainable | LR | Epochs | Purpose |
|-------|-----------|-----|--------|---------|
| Phase 1 | Head + conv1/bn1 | lr × 10 | epochs/3 | Adapt to gradient input |
| Phase 2 | All layers | backbone: lr×0.1, head: lr | remaining | Fine-tune features |

### Optimization

| Feature | Value |
|---------|-------|
| **Optimizer** | AdamW, weight_decay=5e-4 |
| **LR Schedule** | Cosine annealing to 1e-6 |
| **Gradient Clipping** | max_norm=1.0 |
| **Early Stopping** | Patience=15-25 epochs |
| **Dropout** | 0.5 (higher than v1 for regularization) |
| **Mixup** | α=0.4, applied 50% of batches |

### Cross-Validation

Leave-One-Object-Out (LOGO) with **proper training per fold**:
- Each fold has internal val split (12%) for early stopping
- 2-phase training applied within each fold
- Mixup augmentation within each fold
- Reports per-object and overall error

---

### Training Data (from DiLiGenT-like set)

For each object:
- **Input:** images + known `light_directions` (3×96)
- **Objective:** Train a model that learns the mapping between image appearance and lighting direction

### Inference (your own captured data)

- You only have 96 images of a new object under **unknown** lights
- You want a model that, given only those images, predicts the associated 96×3 light direction matrix (`light_directions.txt`)
- Then feed `[your images + predicted light_directions]` into the existing RPS pipeline to get normals, angular error, etc.

### Formal Problem Statement

Learn a function:

$$F: \{I_1, \ldots, I_{96}\} \rightarrow \{\ell_1, \ldots, \ell_{96}\}$$

where each $\ell_i \in \mathbb{R}^3$ is a unit light direction vector.

> **Preference:** Start with something simple and NumPy-friendly (classical ML / light DL) before going into huge CNNs.

---

## High-Level Plan

### 1. Decide on the Learning Granularity

#### Option A: Per-Image Regression

Learn a model $f(I) \approx \ell$ for a single image.

- **Training samples:** Each image across all objects is one training point
- **Input:** Image (maybe downsampled, or simple features from it)
- **Output:** 3D light direction (normalized)
- **At inference:** Run $f$ on each of the 96 images to get 96 vectors

#### Option B: Set-Level Regression

Learn a model $F(I_1, \ldots, I_{96}) \approx (\ell_1, \ldots, \ell_{96})$ all at once.

- More complex (needs permutation-aware architecture or consistent ordering)
- Probably overkill for now

> **Recommendation:** Start with **per-image regression**. You already have a natural one-to-one mapping between image index and light index in `filenames.txt` / `light_directions.txt`. This makes the pipeline much simpler.

---

### 2. Build Numeric Features

To keep it "simple, mostly NumPy":

#### Option A: Downsample & Flatten (Recommended)

1. Convert each 612×512 grayscale image to something like 64×64 or 32×32 using OpenCV/PIL
2. Flatten to a vector: 64×64 = 4096 dims
3. Reduce via PCA (e.g., 128 components)
4. Use those as features for a regression model

#### Option B: Global Intensity Statistics

For each image, compute:
- Mean intensity
- Higher-order moments
- Low-frequency DCT coefficients

> This is more "toy", but very fast to prototype.

**Starting point:** Downsample + flatten + PCA as a good balance.

#### Option C: Physics-Informed Features (Recommended Addition)

Extract features that are more directly related to lighting:
- **Brightest region centroid** (x, y coordinates of brightest 5% pixels)
- **Intensity gradient direction** (dominant gradient angle)
- **Intensity histogram** (distribution of pixel values)
- **Quadrant intensity ratios** (compare mean intensity of image quadrants)

> **Recommended approach:** Combine Options A + C for richer features.

---

### 3. Choose a Learner

#### First Pass
- **Linear regression** or **Ridge regression** on the PCA features to predict the 3D direction

#### Then Try
- Random Forest Regressor
- Small feed-forward neural net (PyTorch, Keras) on the PCA features

#### Handling Unit Vector Output

Since the output is a unit vector:
1. Train to predict a 3D vector $v$
2. Normalize at inference: $\ell = v / \|v\|$

#### Loss Functions

- **MSE** between predicted and GT direction, or
- **Angular loss:** $\arccos(\cos(\theta))$ in radians (preferred for v3)
- **Combined:** 0.7 × Angular + 0.3 × MSE

---

### 4. Training Dataset Organization

Write a small loader using your existing structure:

```
For each object in data/training/*PNG:
    1. Read all filenames from filenames.txt (or glob '*.png' in order)
    2. For each index i:
        - Load image Iᵢ
        - Fetch the i-th row from light_directions.txt as target ℓᵢ
    3. Build arrays:
        - X shape: (num_images_total, feature_dim)
        - Y shape: (num_images_total, 3)
```

**Train/Val/Test Split:** Split across objects (e.g., train on `ball`, `bear`, `cow`, ... and test on `buddha` or `cat`).

---

### 5. Using the Model on New Data

#### Inference Pipeline

1. **Capture** your own 96 images of a new object (under unknown lights)
2. **Preprocess** exactly as in training (grayscale, resize, normalization, PCA transform)
3. **Predict:** For each image $j$, feed to $f$ → get predicted $\ell_j$
4. **Stack** predictions into 96×3 matrix and save as `light_directions.txt` (or `.npy`), ensuring each row is normalized
5. **Run RPS pipeline:**
   ```python
   rps.load_lighttxt("predicted_light_directions.txt")
   rps.load_images()  # or load_npyimages for your images
   ```
6. **Evaluate:** Visualize estimated normals & compute angular error (if GT available)
