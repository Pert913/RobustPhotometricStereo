# TransUNetPS — Uncalibrated Photometric Stereo with TransUNet

Predict surface normal maps from multiple images under **unknown lighting** — no light direction input needed.

Based on the **TransUNet** architecture, closely following the original implementation:
> Chen et al., "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers", *Medical Image Analysis*, 2024.
> https://doi.org/10.1016/j.media.2024.103280
> Source: https://github.com/Beckschen/TransUNet

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision numpy scipy pillow tifffile h5py

# 2. Prepare data
cd deep_photometric_stereo
python prepare_data.py --data_root ./data/training
python prepare_data.py --data_root ./data/testing

# 3. Train (SGD recommended)
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type transunet --optimizer sgd --lr 0.005 \
    --epochs 150 --batch_size 4 --patches_per_epoch 2000

# 4. Test
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

# 5. Predict normal map from new images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/dimooPNG --output ./output/dimoo
```

---

## Model Architectures

Two architectures available via `--model_type`. Both share the same pipeline for photometric stereo (SharedEncoder -> Fusion -> Transformer -> Decoder -> NormalHead), but differ in internal component design and capacity.

---

### Architecture 1: `transunet` (default) — ~11M params

Closely follows the **TransUNet Encoder-only** configuration from the paper. All Transformer components (Attention, MLP, Block) are custom implementations matching the TransUNet source code exactly.

#### Alignment with TransUNet Source Code

| Component | TransUNet (`vit_seg_modeling.py`) | Our `TransUNetPS` (`model.py`) |
|---|---|---|
| **Attention** | `Attention` (L50-94): Q/K/V linear projections, scaled dot-product, multi-head | Identical implementation |
| **MLP** | `Mlp` (L97-119): Linear(d, mlp_dim) -> GELU -> Dropout -> Linear(mlp_dim, d) -> Dropout | Identical, Xavier uniform init |
| **Block** | `Block` (L168-224): LayerNorm -> MSA -> residual -> LayerNorm -> MLP -> residual (pre-norm) | Identical pre-norm design |
| **Encoder** | `Encoder` (L227-244): N x Block + final LayerNorm | Identical stack |
| **Embeddings** | `Embeddings` (L122-165): Conv2d patch projection + learned 1D position embeddings | Identical, with interpolation for variable spatial sizes |
| **DecoderBlock** | `DecoderBlock` (L284-315): BilinearUpsample -> Concat(skip) -> Conv2dReLU x 2 | Identical |
| **DecoderCup** | `DecoderCup` (L326-367): Conv2dReLU head (conv_more) + cascaded DecoderBlocks | Identical, configurable n_skip |
| **Conv2dReLU** | `Conv2dReLU` (L259-281): Conv2d -> BatchNorm2d -> ReLU | Identical |

#### Full Architecture Diagram

```
Input: N grayscale images [B, N, 1, H, W]
                    |
    =====================================
    |  Part I: CNN Encoder (SharedEncoder)  |
    |  (shared weights across all N images) |
    =====================================
    |
    |  enc1: Conv2dReLU(1,64) + Conv2dReLU(64,64)       -> [B*N, 64, H, W]
    |  enc2: MaxPool2d(2) + Conv2dReLU(64,128) x2        -> [B*N, 128, H/2, W/2]
    |  enc3: MaxPool2d(2) + Conv2dReLU(128,256) x2       -> [B*N, 256, H/4, W/4]
    |  enc4: MaxPool2d(2) + Conv2dReLU(256,512) x2       -> [B*N, 512, H/8, W/8]
    |  bottleneck: MaxPool2d(2)                           -> [B*N, 512, H/16, W/16]
    |
    =========================================
    |  Multi-Scale Fusion (our contribution)   |
    |  Max-pool across N images at each level  |
    |  Handles variable N via masking          |
    =========================================
    |
    |  fused_enc1: [B, 64, H, W]         (skip connection 1)
    |  fused_enc2: [B, 128, H/2, W/2]    (skip connection 2)
    |  fused_enc3: [B, 256, H/4, W/4]    (skip connection 3)
    |  fused_enc4: [B, 512, H/8, W/8]    (skip connection 4)
    |  fused_bottleneck: [B, 512, H/16, W/16]
    |                    |
    ==========================================
    |  Part III: Transformer Encoder            |
    |  (following TransUNet Eq.2 and Eq.3)      |
    ==========================================
    |
    |  Patch Embedding:
    |    Conv2d(512, 256, kernel=1)  -- project to hidden_size
    |    + Learned 1D position embeddings (n_patches tokens)
    |    + Dropout(0.1)
    |    -> [B, H/16 * W/16, 256]   (sequence of tokens)
    |
    |  Transformer Block x2:
    |    z' = Attention(LayerNorm(z)) + z    -- Multi-Head Self-Attention + residual
    |    z  = MLP(LayerNorm(z')) + z'        -- Feed-Forward Network + residual
    |
    |    Attention: 4 heads, head_dim=64, scaled dot-product
    |      Q = Linear(256, 256), K = Linear(256, 256), V = Linear(256, 256)
    |      attn = softmax(Q @ K^T / sqrt(64)) @ V
    |      out = Linear(256, 256)
    |
    |    MLP: Linear(256, 512) -> GELU -> Dropout -> Linear(512, 256) -> Dropout
    |
    |  Final LayerNorm(256)
    |  -> [B, H/16 * W/16, 256]
    |
    |  Reshape back to spatial: [B, 256, H/16, W/16]
    |                    |
    ==========================================
    |  Part II: CNN Decoder (DecoderCup)        |
    |  Cascaded upsampler with skip connections  |
    ==========================================
    |
    |  conv_more: Conv2dReLU(256, 512)           -> [B, 512, H/16, W/16]
    |  block1: Upsample(2x) + Cat(fused_enc4)
    |          + Conv2dReLU(1024, 256) x2        -> [B, 256, H/8, W/8]
    |  block2: Upsample(2x) + Cat(fused_enc3)
    |          + Conv2dReLU(512, 128) x2         -> [B, 128, H/4, W/4]
    |  block3: Upsample(2x) + Cat(fused_enc2)
    |          + Conv2dReLU(256, 64) x2          -> [B, 64, H/2, W/2]
    |  block4: Upsample(2x) + Cat(fused_enc1)
    |          + Conv2dReLU(128, 16) x2          -> [B, 16, H, W]
    |                    |
    ==========================================
    |  NormalHead                                |
    ==========================================
    |
    |  Conv2d(16, 3, kernel=3, padding=1)
    |  L2 Normalize along channel dim
    |  -> [B, 3, H, W]  (unit normal vectors)

Total parameters: ~10,963,000
```

---

### Architecture 2: `lightweight` — ~4.6M params

Our custom lighter architecture designed for faster training and lower memory. Uses PyTorch's built-in Transformer instead of custom blocks, with smaller channels throughout.

#### Key Differences from TransUNet

| Design choice | `transunet` | `lightweight` |
|---|---|---|
| Transformer implementation | Custom Attention + Mlp + Block classes | PyTorch `nn.TransformerEncoderLayer` |
| Attention mechanism | Manual Q/K/V projections + scaled dot-product | PyTorch built-in multi-head attention |
| MLP activation | GELU (from TransUNet) | GELU (PyTorch default for TransformerEncoderLayer) |
| Pre-norm | Custom LayerNorm before each sub-layer | `norm_first=True` in PyTorch |
| Positional encoding | Learned 1D embeddings (flat sequence) | Learned 2D embeddings (separate row + column) |
| Patch projection | Conv2d(512, 256) + learned 1D pos | Conv2d(256, 256) + 2D row/col Embedding |
| Encoder architecture | `EncoderBlock` (Conv2dReLU x2) | `_LWConvBlock` (Conv-BN-ReLU x2) |
| Decoder architecture | `DecoderCup` (conv_more head + DecoderBlocks) | Simple `_LWUpBlock` (Upsample + Concat + Conv x2) |
| Residual around Transformer | No | Yes (input + output) |
| Output channels | [64, 128, 256, 512] -> 256 hidden -> [512, 256, 128, 64, 16] | [32, 64, 128, 256] -> 256 hidden -> [128, 64, 32, 32] |

#### Full Architecture Diagram

```
Input: N grayscale images [B, N, 1, H, W]
                    |
    =========================================
    |  CNN Encoder (shared across N images)    |
    =========================================
    |
    |  enc1: Conv(1,32)-BN-ReLU + Conv(32,32)-BN-ReLU     -> [B*N, 32, H, W]
    |  enc2: MaxPool(2) + Conv(32,64)-BN-ReLU x2           -> [B*N, 64, H/2, W/2]
    |  enc3: MaxPool(2) + Conv(64,128)-BN-ReLU x2          -> [B*N, 128, H/4, W/4]
    |  enc4: MaxPool(2) + Conv(128,256)-BN-ReLU x2         -> [B*N, 256, H/8, W/8]
    |  bottleneck: MaxPool(2) + Conv(256,256)-BN-ReLU x2   -> [B*N, 256, H/16, W/16]
    |
    =========================================
    |  Multi-Scale Fusion                      |
    |  (same as transunet — max-pool across N) |
    =========================================
    |
    |  fused_enc1 through fused_enc4 + fused_bottleneck
    |                    |
    =========================================
    |  Transformer Bottleneck                   |
    |  (PyTorch built-in + residual connection) |
    =========================================
    |
    |  patch_proj: Conv2d(256, 256, kernel=1)
    |  + Learned 2D position encoding:
    |      row_embed: Embedding(64, 128)  -- 128 = hidden_dim / 2
    |      col_embed: Embedding(64, 128)
    |      pos = concat(row_embed[y], col_embed[x])  -> (256, H/16, W/16)
    |
    |  Flatten to sequence: [B, H/16 * W/16, 256]
    |
    |  nn.TransformerEncoder(
    |      nn.TransformerEncoderLayer(
    |          d_model=256, nhead=4, dim_feedforward=512,
    |          dropout=0.1, norm_first=True   -- pre-norm
    |      ),
    |      num_layers=2
    |  )
    |  + LayerNorm(256)
    |
    |  Reshape: [B, 256, H/16, W/16]
    |  proj_back: Conv2d(256, 256, kernel=1)
    |  + RESIDUAL connection (input + output)  -- key difference from transunet
    |                    |
    =========================================
    |  U-Net Decoder (simple UpBlocks)          |
    =========================================
    |
    |  up4: Upsample(2x) + Cat(fused_enc4)
    |       + Conv(512,128)-BN-ReLU x2          -> [B, 128, H/8, W/8]
    |  up3: Upsample(2x) + Cat(fused_enc3)
    |       + Conv(256,64)-BN-ReLU x2           -> [B, 64, H/4, W/4]
    |  up2: Upsample(2x) + Cat(fused_enc2)
    |       + Conv(128,32)-BN-ReLU x2           -> [B, 32, H/2, W/2]
    |  up1: Upsample(2x) + Cat(fused_enc1)
    |       + Conv(64,32)-BN-ReLU x2            -> [B, 32, H, W]
    |                    |
    =========================================
    |  Normal Output                            |
    =========================================
    |
    |  Conv2d(32, 3, kernel=1)
    |  L2 Normalize along channel dim
    |  -> [B, 3, H, W]  (unit normal vectors)

Total parameters: ~4,552,000
```

---

### Side-by-Side Summary

| | `transunet` | `lightweight` |
|---|---|---|
| **Total params** | ~11.0M | ~4.6M |
| **Encoder params** | ~7.1M (4 levels, 64->512 channels) | ~1.9M (4 levels, 32->256 channels) |
| **Transformer params** | ~0.9M (custom Attention+MLP x2) | ~0.9M (PyTorch TransformerEncoderLayer x2) |
| **Decoder params** | ~2.9M (DecoderCup, 512->16 channels) | ~1.7M (UpBlocks, 128->32 channels) |
| **Head params** | ~0.05M | ~0.01M |
| **Bottleneck spatial** | H/16 x W/16 (e.g., 8x8 for 128x128 input) | Same |
| **Hidden dim** | 256 | 256 |
| **Attention heads** | 4 | 4 |
| **FF dimension** | 512 | 512 |
| **Transformer layers** | 2 | 2 |

### Our Additions for Photometric Stereo (shared by both architectures)

| Addition | Purpose |
|---|---|
| **SharedEncoder** | Weight-shared CNN encoder processes each of N images independently — same weights for all N images in a batch |
| **MultiScaleFusion** | Max-pool across N images at EVERY encoder scale — captures strongest shading features regardless of lighting direction |
| **Variable-N support** | Masking allows batches with different numbers of input images (8 to 96) |
| **NormalHead** | L2-normalized 3-channel output ensures predicted normals are unit vectors |
| **Dual mode** | Same architecture supports `normal` (photometric stereo) and `segmentation` (Synapse) output heads |
| **Auto model detection** | Checkpoint file stores model type — test.py and predict.py auto-detect, no manual flag needed |

---

## Comparison with Prior Work

| | TransUNet (original) | CNN-PS | SDM-UniPS | **TransUNetPS (ours)** |
|---|---|---|---|---|
| Task | Medical segmentation | Calibrated PS | Uncalibrated PS | **Uncalibrated PS** |
| Light dirs | N/A | Required | Not required | **Not required** |
| Architecture | CNN + ViT + U-Net | DenseNet | ConvNeXt+UPerNet+PMA | **SharedCNN + ViT + U-Net** |
| Params | ~41.4M (Enc+Dec) | ~80M | ~300M | **~11M / ~4.6M** |
| Transformer | 12-layer ViT, d=768 | None | Cross-image | **2-layer, d=256** |

---

## Training

### Photometric Stereo

**SGD is recommended** for photometric stereo training from scratch. SGD's noisy gradients act as a regularizer, which is critical for patch-based training to generalize to full-image inference. AdamW achieves lower training loss but can overfit to patches.

| Parameter | TransUNet (paper) | Ours (photometric stereo) | Ours (Synapse) |
|---|---|---|---|
| Optimizer | SGD(momentum=0.9) | SGD(momentum=0.9) | SGD(momentum=0.9) |
| Learning rate | 0.01 | 0.005 (recommended) | 0.01 |
| LR schedule | Poly decay (0.9) | Poly decay (0.9) | Poly decay (0.9) |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Epochs | 150 | 150 | 150 |
| Grad clipping | N/A | max_norm=1.0 | N/A |

```bash
# Photometric stereo — SGD (recommended)
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type transunet --optimizer sgd --lr 0.005 \
    --epochs 150 --batch_size 4 --patches_per_epoch 2000

# Photometric stereo — AdamW (alternative, may overfit to patches)
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type transunet --optimizer adamw --lr 1e-3 \
    --epochs 150 --batch_size 4 --patches_per_epoch 2000
```

### Synapse Segmentation

```bash
python train.py --mode synapse --model_type transunet --epochs 150
```

### Loss Functions

| Mode | Loss | Formula |
|---|---|---|
| Photometric Stereo | Angular + Cosine + L1 | `0.5 * Angular + 0.3 * Cosine + 0.2 * L1` (masked) |
| Segmentation | CE + Dice (same as TransUNet) | `0.5 * CrossEntropy + 0.5 * Dice` |

---

## Supported Datasets

| Dataset | Format | Description |
|---|---|---|
| **DiLiGenT** | `.mat`, `.png` | 10 training + 5 testing objects, 96 images each, 512x612 |
| **PRPS** | `.tif`, `gt_normal.tif`, `inboundary.png` | Synthetic objects, specular/metallic variants |
| **Synapse** | `.npz`, `.h5` | 2211 train slices + 12 test volumes, 9-class organ segmentation |

---

## Testing / Evaluation

Model type is **auto-detected** from the checkpoint.

### Photometric Stereo

```bash
# Test on all training objects
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

# Test on all testing objects
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing

# Test + save output images
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing --save_output ./output/testing

# Single object
python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG

# LOGO folds
python test.py --mode logo_eval --checkpoint_dir checkpoints/
```

### Synapse Segmentation

```bash
python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt
python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt --save_output ./output/synapse
```

### Evaluation Metrics

**Photometric stereo** (standard metrics from DiLiGenT benchmark):

| Metric | Description |
|---|---|
| Mean Angular Error (MAE) | Average angle between predicted and GT normals (degrees) |
| Median Angular Error | Median angle, robust to outliers |
| % pixels < 10° / 20° / 30° | Percentage of pixels within error thresholds |
| Max Error | Worst-case pixel error |

**Synapse segmentation** (standard metrics from TransUNet paper):

| Metric | Description |
|---|---|
| Per-organ Dice score | Overlap metric per organ class (8 organs) |
| Average Dice | Mean Dice across all organs |

---

## Prediction

Model type is **auto-detected** from the checkpoint.

### Photometric Stereo

Uses **Sliding Window with Hanning Blending** to eliminate tile boundary artifacts: generous padding (tile_size//2 on all sides), 50% overlap, Hanning window weighting for smooth blending.

```bash
# Predict from a folder of images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/dimooPNG --output ./output/dimoo

# Predict with a mask
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --mask ./my_images/mask.png --output ./output/

# Predict from specific files
python predict.py --checkpoint checkpoints/train/best.pt --images img1.png img2.png img3.png --output ./output/
```

**Output files:**
- `predicted_normal.npy` — raw normal map (H, W, 3) float32
- `predicted_normal.png` — RGB visualization
- `predicted_nx.png`, `predicted_ny.png`, `predicted_nz.png` — per-component

### Synapse Segmentation

```bash
# Predict on .npz slices
python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt \
    --images case0005_slice050.npz --output ./output/synapse_pred

# Predict on a folder
python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt \
    --input_dir ./my_ct_slices/ --output ./output/synapse_pred
```

**Output files (per input):**
- `*_pred.npy` — label map (H, W) uint8, class IDs 0-8
- `*_pred_color.png` — color-coded organ visualization
- `*_pred_label.png` — grayscale label visualization

---

## REST API (serve.py + client.py)

The trained model can be served as a REST API for integration with web apps.

### Start the Server

```bash
pip install fastapi uvicorn python-multipart
python serve.py --checkpoint checkpoints/train/best.pt --port 8000
```

API docs auto-generated at: `http://localhost:8000/docs`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/info` | GET | Model info (type, params, device) |
| `/predict` | POST | Predict normal map — returns PNG, NPY, or ZIP |
| `/predict/json` | POST | Predict and return stats as JSON |

### Query Parameters for `/predict`

| Parameter | Default | Description |
|---|---|---|
| `format` | `png` | Output format: `png` (normal map image), `npy` (raw numpy), `zip` (all outputs) |
| `tile_size` | `128` | Tile size for inference |
| `max_images` | `96` | Max number of images to use |

### Test with curl

```bash
# Get normal map as PNG
curl -X POST http://localhost:8000/predict \
    -F "images=@img1.png" -F "images=@img2.png" -F "images=@img3.png" \
    --output predicted_normal.png

# Get all outputs as ZIP (normal map + nx/ny/nz components + npy)
curl -X POST "http://localhost:8000/predict?format=zip" \
    -F "images=@img1.png" -F "images=@img2.png" -F "images=@img3.png" \
    --output result.zip
```

### Python Client

```python
from client import TransUNetPSClient

client = TransUNetPSClient("http://localhost:8000")

# Predict from a folder
normal_map = client.predict_from_folder("./data/testing/dimooPNG")
normal_map.save("output.png")

# Download all outputs as zip
client.predict_zip(["img1.png", "img2.png"], output_dir="./output")

# Or from command line
# python client.py "./data/testing/dimooPNG"
```

### Building a React Frontend

The API is designed to work with a React app. Here's how to integrate:

#### 1. React Component: Upload Images and Get Normal Map

```jsx
import React, { useState } from 'react';

function NormalMapPredictor() {
  const [files, setFiles] = useState([]);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [format, setFormat] = useState('png'); // 'png' or 'zip'

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length < 2) {
      alert('Please upload at least 2 images');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    files.forEach((file) => formData.append('images', file));

    try {
      const response = await fetch(
        `http://localhost:8000/predict?format=${format}`,
        { method: 'POST', body: formData }
      );

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Prediction failed');
      }

      const blob = await response.blob();

      if (format === 'png') {
        // Display the normal map image
        const url = URL.createObjectURL(blob);
        setResultUrl(url);
      } else {
        // Download as zip
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction_result.zip';
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Normal Map Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={(e) => setFiles(Array.from(e.target.files))}
        />
        <p>{files.length} images selected</p>

        <label>
          Output format:
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="png">PNG (normal map image)</option>
            <option value="zip">ZIP (all outputs)</option>
          </select>
        </label>

        <button type="submit" disabled={loading || files.length < 2}>
          {loading ? 'Predicting...' : 'Predict Normal Map'}
        </button>
      </form>

      {resultUrl && (
        <div>
          <h2>Predicted Normal Map</h2>
          <img src={resultUrl} alt="Predicted normal map" style={{ maxWidth: '100%' }} />
        </div>
      )}
    </div>
  );
}

export default NormalMapPredictor;
```

#### 2. Enable CORS on the API Server

If the React app runs on a different port (e.g., `localhost:3000`), add CORS to `serve.py`:

```python
# In serve.py, after creating the app:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3. API Request/Response Summary for Frontend

```
POST /predict?format=png
  Request:  multipart/form-data with field "images" (multiple files)
  Response: image/png (binary) — the RGB normal map

POST /predict?format=zip
  Request:  multipart/form-data with field "images" (multiple files)
  Response: application/zip containing:
    - predicted_normal.png  (RGB normal map)
    - predicted_normal.npy  (raw float32 array)
    - predicted_nx.png      (X component)
    - predicted_ny.png      (Y component)
    - predicted_nz.png      (Z component)

POST /predict/json
  Request:  multipart/form-data with field "images" (multiple files)
  Response: application/json
    {
      "shape": [H, W, 3],
      "num_images_used": N,
      "normal_mean": [x, y, z],
      "normal_std": [x, y, z],
      "norm_mean": 1.0,
      "norm_min": 0.99,
      "norm_max": 1.0
    }

GET /health
  Response: { "status": "ok", "model_loaded": true, "device": "cuda" }

GET /info
  Response: { "model_type": "TransUNetPS", "parameters": 10962995, "device": "cuda" }
```

---

## Project Structure

```
deep_photometric_stereo/
├── config.py           # Configuration (aligned with TransUNet)
├── model.py            # TransUNetPS (~11M) + LightweightUNetPS (~4.6M)
├── losses.py           # Normal losses (angular/cosine/L1) + segmentation (CE/Dice)
├── dataset.py          # DiLiGenT / PRPS dataset loaders
├── train.py            # Training: photometric stereo (train/logo_cv) + synapse
├── test.py             # Evaluation: eval/eval_all/logo_eval/synapse
├── predict.py          # Inference: normal map (sliding window) + synapse segmentation
├── serve.py            # REST API server (FastAPI + uvicorn)
├── client.py           # Python client for the API
├── prepare_data.py     # Data preparation (mat/tif/npz -> npy)
├── utils.py            # Metrics, checkpointing, visualization, model type detection
├── intruction/         # Step-by-step workflow guides
│   ├── full_workflow_photometric.md
│   └── full_worklfow_synapse.md
├── data/               # Training & testing data (gitignored)
│   ├── training/       # DiLiGenT: 10 objects x 96 images
│   ├── testing/        # DiLiGenT: 5 test objects
│   └── Synapse/        # 2211 train slices + 12 test volumes (symlink)
└── README.md
```

---

## References

- **TransUNet**: Chen, J. et al. "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers." *Medical Image Analysis* 97 (2024): 103280.
- **CNN-PS**: Ikehata, S. "CNN-PS: CNN-based photometric stereo for general non-convex surfaces." *ECCV* 2018.
- **SDM-UniPS**: Ikehata, S. "SDM-UniPS: Scalable, Detailed, and Mask-free Universal Photometric Stereo." *CVPR* 2023.
- **DiLiGenT**: Shi, B. et al. "A benchmark dataset and evaluation for non-Lambertian and uncalibrated photometric stereo." *IEEE TPAMI* 2019.
