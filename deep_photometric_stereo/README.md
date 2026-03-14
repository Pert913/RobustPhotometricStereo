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
pip install torch torchvision numpy scipy pillow tifffile

# 2. Prepare data
cd deep_photometric_stereo
python prepare_data.py

# 3. Train
python train.py --mode logo_cv --epochs 150

# 4. Predict normal map from new images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --output ./output/
```

---

## Architecture: TransUNetPS

We closely follow the **TransUNet Encoder-only** configuration (Parts I + III + II), adapting it for uncalibrated photometric stereo.

### Alignment with TransUNet Source Code

| Component | TransUNet (`vit_seg_modeling.py`) | Our Implementation (`model.py`) |
|---|---|---|
| **Attention** | `Attention` class (L50-94): Q/K/V projections, scaled dot-product | `Attention` class: identical design |
| **MLP** | `Mlp` class (L97-119): fc1 → GELU → dropout → fc2 → dropout | `Mlp` class: identical, Xavier init |
| **Block** | `Block` class (L168-224): pre-LayerNorm, MSA + residual, MLP + residual | `Block` class: identical pre-norm design |
| **Encoder** | `Encoder` class (L227-244): N × Block + final LayerNorm | `TransformerEncoder`: identical stack |
| **Embeddings** | `Embeddings` (L122-165): Conv patch embed + learned 1D pos embed | `Embeddings`: identical, with pos interpolation |
| **DecoderBlock** | `DecoderBlock` (L284-315): Upsample → Cat(skip) → Conv2dReLU × 2 | `DecoderBlock`: identical design |
| **DecoderCup** | `DecoderCup` (L326-367): conv_more head + cascaded decoder blocks | `DecoderCup`: identical, configurable n_skip |
| **Conv2dReLU** | `Conv2dReLU` (L259-281): Conv + BN + ReLU | `Conv2dReLU`: identical |
| **Training** | SGD(lr=0.01, momentum=0.9, wd=1e-4), poly LR decay (power=0.9) | Identical optimizer and scheduler |

### Our Additions for Photometric Stereo

| Addition | Purpose |
|---|---|
| **SharedEncoder** | Weight-shared CNN encoder processes each of N images independently |
| **MultiScaleFusion** | Max-pool across N images at every encoder scale (variable-N support) |
| **NormalHead** | L2-normalized 3-channel output for surface normals |
| **Dual mode** | Supports both `normal` (photometric stereo) and `segmentation` (Synapse) modes |

```
Input: N grayscale images [B, N, 1, H, W]

Part I: CNN Encoder (shared weights across N images)
  enc1: 2×Conv2dReLU(1→64)   → [B*N, 64, H, W]
  enc2: Pool + 2×Conv(64→128) → [B*N, 128, H/2, W/2]
  enc3: Pool + 2×Conv(128→256)→ [B*N, 256, H/4, W/4]
  enc4: Pool + 2×Conv(256→512)→ [B*N, 512, H/8, W/8]
  bottleneck: Pool            → [B*N, 512, H/16, W/16]

Multi-Scale Fusion (our contribution):
  Max-pool across N at each level → [B, C, h, w]

Part III: Transformer Encoder
  Patch Embedding: Conv 1x1 (512→256) + learned 1D pos embed
  2× TransformerBlock:
    z' = MSA(LN(z)) + z   (Eq.2 from TransUNet)
    z  = MLP(LN(z')) + z'  (Eq.3 from TransUNet)
  hidden=256, heads=4, mlp=512

Part II: CNN Decoder (DecoderCup)
  conv_more: Conv2dReLU(256→512)
  block1: Up + Cat(enc4) + 2×Conv → 256
  block2: Up + Cat(enc3) + 2×Conv → 128
  block3: Up + Cat(enc2) + 2×Conv → 64
  block4: Up + Cat(enc1) + 2×Conv → 16

NormalHead: Conv(16→3) → L2 normalize → [B, 3, H, W]
~11M parameters
```

---

## Comparison with Prior Work

| | TransUNet (original) | CNN-PS | SDM-UniPS | **TransUNetPS (ours)** |
|---|---|---|---|---|
| Task | Medical segmentation | Calibrated PS | Uncalibrated PS | **Uncalibrated PS** |
| Light dirs | N/A | Required | Not required | **Not required** |
| Architecture | CNN + ViT + U-Net | DenseNet | ConvNeXt+UPerNet+PMA | **SharedCNN + ViT + U-Net** |
| Params | ~41.4M (Enc+Dec) | ~80M | ~300M | **~11M** |
| Transformer | 12-layer ViT, d=768 | None | Cross-image | **2-layer, d=256** |

---

## Training

### Hyperparameters (aligned with TransUNet)

| Parameter | TransUNet | Ours |
|---|---|---|
| Optimizer | SGD(momentum=0.9) | SGD(momentum=0.9) |
| Learning rate | 0.01 | 0.01 |
| LR schedule | Poly decay (power=0.9) | Poly decay (power=0.9) |
| Weight decay | 1e-4 | 1e-4 |
| Epochs | 150 | 150 |
| Grad clipping | N/A | max_norm=1.0 |

### Loss Functions

**Photometric Stereo mode:** `0.5 × Angular + 0.3 × Cosine + 0.2 × L1` (masked)

**Segmentation mode:** `0.5 × CrossEntropy + 0.5 × Dice` (same as TransUNet)

---

## Supported Datasets

| Dataset | Format | Description |
|---|---|---|
| **DiLiGenT** | `.mat`, `.png` | 10 objects × 96 images, 512×612 |
| **PRPS** | `.tif`, `gt_normal.tif` | Synthetic objects, specular/metallic variants |
| **Synapse** | `.npz`, `.h5` | 30 CT volumes, 9-class organ segmentation |

---

## Prediction

```bash
# Predict normal map from images
python predict.py \
    --checkpoint checkpoints/train/best.pt \
    --input_dir /path/to/images/ \
    --mask /path/to/mask.png \
    --output ./output/

# Output files:
#   output/predicted_normal.npy   (H, W, 3) float32
#   output/predicted_normal.png   RGB visualization
#   output/predicted_nx.png       X-component
#   output/predicted_ny.png       Y-component
#   output/predicted_nz.png       Z-component
```

---

## Project Structure

```
deep_photometric_stereo/
├── config.py           # Configuration (aligned with TransUNet)
├── model.py            # TransUNetPS (Attention, Block, Encoder, DecoderCup — all from TransUNet)
├── losses.py           # Masked angular + cosine + L1 losses
├── dataset.py          # DiLiGenT/PRPS/Synapse data loaders
├── train.py            # Training with SGD + poly LR (TransUNet style)
├── test.py             # Evaluation
├── predict.py          # Inference on new images → normal map
├── prepare_data.py     # Data preparation (mat/tif/npz → npy)
├── utils.py            # Metrics, checkpointing, visualization
├── data/               # Training data (gitignored)
│   ├── training/       # DiLiGenT: 10 objects × 96 images
│   ├── testing/        # DiLiGenT: 5 test objects
│   └── Synapse/        # Synapse: 2211 train slices + 12 test volumes (symlink)
└── README.md
```

---

## References

- **TransUNet**: Chen, J. et al. "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers." *Medical Image Analysis* 97 (2024): 103280.
- **CNN-PS**: Ikehata, S. "CNN-PS: CNN-based photometric stereo for general non-convex surfaces." *ECCV* 2018.
- **SDM-UniPS**: Ikehata, S. "SDM-UniPS: Scalable, Detailed, and Mask-free Universal Photometric Stereo." *CVPR* 2023.
- **DiLiGenT**: Shi, B. et al. "A benchmark dataset and evaluation for non-Lambertian and uncalibrated photometric stereo." *IEEE TPAMI* 2019.
