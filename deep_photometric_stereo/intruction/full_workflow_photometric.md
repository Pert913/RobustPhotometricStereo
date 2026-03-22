# Full Workflow: Photometric Stereo (Normal Map Prediction)

All commands run from: `cd deep_photometric_stereo`

---

## Step 0: Prepare Data (run once)

```bash
python prepare_data.py --data_root ./data/training
python prepare_data.py --data_root ./data/testing
```

---

## Step 1: Training

### Option A: SGD optimizer (recommended — best generalization)

#### Note: lr is 0.005 , if you want to make the model better , then reduce the lr but if you want to be faster than lr=0.01 is okay.

```bash
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type transunet \
    --optimizer sgd \
    --lr 0.005 \
    --epochs 150 \
    --batch_size 4 \
    --patches_per_epoch 2000
```

### Option B: Lightweight model + SGD (faster, ~4.6M params)

```bash
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type lightweight \
    --optimizer sgd \
    --lr 0.005 \
    --epochs 150 \
    --batch_size 4 \
    --patches_per_epoch 2000
```

### Option C: AdamW optimizer (lower training loss, but may overfit to patches)

```bash
python train.py --mode train \
    --train_objects ballPNG bearPNG buddhaPNG catPNG cowPNG gobletPNG harvestPNG pot1PNG pot2PNG readingPNG \
    --test_object ballPNG \
    --model_type transunet \
    --optimizer adamw \
    --lr 1e-3 \
    --epochs 150 \
    --batch_size 4 \
    --patches_per_epoch 2000
```

**Output:** `checkpoints/train/best.pt`

> **Important:** Use `--optimizer sgd` (recommended). SGD's noisy gradients act as a regularizer, which is critical for patch-based training to generalize to full-image inference. AdamW achieves lower training loss but can overfit to patches.
>
> If OOM, lower `--batch_size 2`. For a quick test: `--epochs 5 --patches_per_epoch 100`.

---

## Step 2: Testing

Model type is auto-detected from the checkpoint.

```bash
# Test on all training objects (10 objects)
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

# Test on all testing objects (5 objects)
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing

# Test + save output images (predicted normals, GT normals, error maps)
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training --save_output ./output/training
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing --save_output ./output/testing

# Test a single object
python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object batteryPNG --data_root ./data/testing
```

---

## Step 3: Prediction (on new images)

Prediction uses **center-crop tiling** — each tile is processed with extra context margin, only the center is kept. No grid/square artifacts.

```bash
# Predict from a folder of images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/dimooPNG --output ./output/dimoo

# Predict from a folder of images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/batteryPNG --output ./output/battery

# Predict with a mask
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --mask ./my_images/mask.png --output ./output/

# Predict from specific files
python predict.py --checkpoint checkpoints/train/best.pt --images img1.png img2.png img3.png --output ./output/

# Use larger tile size for better quality (slower, needs more memory)
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --tile_size 256 --output ./output/
```

**Output files:**
- `predicted_normal.npy` — raw normal map (H, W, 3) float32
- `predicted_normal.png` — RGB visualization
- `predicted_nx.png`, `predicted_ny.png`, `predicted_nz.png` — per-component
