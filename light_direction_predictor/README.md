# ML/DL Project: Light Direction Prediction

## Goal

Remove the dependency on known light directions when collecting your own data.

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
- **Cosine loss:** $1 - \cos(\theta) = 1 - \text{dot product}$ (preferred)

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