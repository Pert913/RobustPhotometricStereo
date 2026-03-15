# Full Workflow: Synapse Segmentation (Organ Segmentation)

All commands run from: `cd deep_photometric_stereo`

**Prerequisite:** Synapse data must be at `./data/Synapse/` (symlinked or copied), with `train_npz/`, `test_vol_h5/`, `train.txt`, and `test_vol.txt`.

---

## Step 1: Training

### Option A: TransUNet model (~11M params — recommended)

```bash
python train.py --mode synapse --model_type transunet --epochs 150
```

### Option B: Lower batch size (if out-of-memory)

```bash
python train.py --mode synapse --model_type transunet --epochs 150 --batch_size 2
```

### Option C: Lightweight model (~4.6M params — faster)

```bash
python train.py --mode synapse --model_type lightweight --epochs 150
```

**Output:** `checkpoints/synapse/best.pt` + periodic checkpoints every 50 epochs

> **Note:** For a quick test, use `--epochs 5`. Full 150 epochs takes ~3-4 hours on GPU.

---

## Step 2: Testing (all 12 test volumes)

Model type is auto-detected from the checkpoint.

```bash
# Evaluate on all 12 test volumes — prints per-organ Dice scores
python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt

# Evaluate + save prediction volumes as .npy
python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt --save_output ./output/synapse
```

**Output:** Per-organ Dice scores for 8 organs (Aorta, Gallbladder, Kidney L/R, Liver, Pancreas, Spleen, Stomach) + average Dice.

---

## Step 3: Prediction (on new CT slices)

```bash
# Predict on individual .npz slices
python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt \
    --images ./data/Synapse/train_npz/case0005_slice050.npz \
    --output ./output/synapse_pred

# Predict on multiple slices
python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt \
    --images case0005_slice050.npz case0005_slice060.npz case0005_slice070.npz \
    --output ./output/synapse_pred

# Predict on a folder of PNG/JPG/NPZ images
python predict.py --mode synapse --checkpoint checkpoints/synapse/best.pt \
    --input_dir ./my_ct_slices/ \
    --output ./output/synapse_pred
```

**Output files (per input):**
- `*_pred.npy` — raw label map (H, W) uint8, class IDs 0-8
- `*_pred_color.png` — color-coded organ visualization
- `*_pred_label.png` — grayscale label visualization

**Color legend:**
| Color | Organ |
|-------|-------|
| Red | Aorta |
| Green | Gallbladder |
| Blue | Kidney (L) |
| Yellow | Kidney (R) |
| Orange | Liver |
| Purple | Pancreas |
| Cyan | Spleen |
| Magenta | Stomach |
| Black | Background |
