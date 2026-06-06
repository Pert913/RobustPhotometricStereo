Photometric Stereo — Session Handoff (2026-05-29 → 2026-06-02)
Core change: 24-LED ring-mixing training strategy
New formula (per sample):

I_c_train = α·I_c_env + (1-α)·Σ_{i=0..7} b_{c,i}·I_c_point_i for c ∈ {R, G, B}
I_c_lights_off = α·I_c_env + N(0, dark_noise_std²)
I_c_sub = I_c_train - I_c_lights_off ← model input
b_{c,i} ~ U(0, 1) raw (no normalization), α ~ U(0.4, 0.7)
Canonical ring geometry (in _canonical_led_directions):

24 LEDs on a circle around optical axis at 30° tilt (ring_tilt_deg=30.0)
LED 0 at bottom, CCW around optical axis as viewed from camera
Frame: +x right, +y up, +z from object→camera (so +z = cos(30°))
LEDs 0-7 = R arc, 8-15 = G arc, 16-23 = B arc
Files changed
config.py — added to DataConfig:

use_ring_mixing=True, ring_tilt_deg=30.0
alpha_min=0.4, alpha_max=0.7
dark_noise_std=0.01 (we trained with 0 to isolate effects)
gamma_min=0.7, gamma_max=1.4
dataset.py — module-level helpers added:

_canonical_led_directions(tilt_deg) — 24×3 unit vectors
_match_canonical(avail_dirs, canonical_dirs) — nearest-neighbor cosine similarity matching
_load_prps_light_dirs_cam(cam_folder) — parses point_lights.config (parent dir, format name x y z intensity) + cams.config (format name 3-floats 16-floats(4x4 cam-to-world)); transforms positions to cam frame; auto-flips z sign if mean is negative
_ring_mix_channels(point_imgs, env_rgb, ring_indices, alpha, b_weights, dark_noise_std) — composes I_sub
Both DiLiGentDataset and SyntheticDataset constructors gained use_ring_mixing, ring_tilt_deg, alpha_min, alpha_max, dark_noise_std, gamma_min, gamma_max kwargs.

Gamma applied in raw-intensity space BEFORE z-score, gated on augment and gamma_min < gamma_max. Negatives clipped to 0.

Per-dataset behavior:

DiLiGenT (data/training/<obj>/light_directions.txt): 96 lights → matched to 24 canonical at _scan_object time (one-time per object)
PRPS (data/synthetic/render-complex-v1/<material>/cam_NNNNN): parses configs at material parent dir, transforms to cam frame, cached in _dirs_cache: dict. 20 lights → 24 canonical (some duplicates)
Formatted_Dataset_Final (D:\Formatted_Dataset_Final\<obj>): no configs → random 24-of-32 fallback (no arc structure but still feeds the mixer)
train.py — wired all DataConfig kwargs through 3 dataset constructor sites; added CLI args:

--use_ring_mixing / --no_ring_mixing
--ring_tilt_deg <float>, --alpha_min, --alpha_max
--dark_noise_std, --gamma_min, --gamma_max
Bug fixed: patches_per_epoch // 2 in joint_888 pipeline silently halved DiLiGenT's patches/epoch. Removed.

predict.py — added:

_ring_aggregate_images(images, light_dirs, tilt_deg) — sums 8 lights per arc into 3 channels at inference (uniform b=1, z-score handles scale)
--ring_aggregate / --no_ring_aggregate (default ON; auto-triggers when light_directions.txt present)
--ring_tilt_deg <float>
--use_model_mask — uses model's seg-head sigmoid > 0.5 instead of Otsu/dark-ref
Two bugs fixed in predict.py:

(N,H,W) grayscale stacks with N≠3 and W≥3 were being misclassified as (H,W,C) single images. Tightened the elif guard: now requires shape[2] in (3,4) and shape[2] < shape[0].
The model-mask branch lives at the top of the mask-decision block, alongside dark-ref and Otsu — all three set bright_mask, bg_floor, brightness_threshold, mask_source then share the morphology pipeline.
Trained checkpoint
checkpoints/joint_888_run/best_joint.pt

Epoch 155, val_loss=9.2540
Model: transunet, 10,967,300 params
Trained with: --mode joint_888 --epochs 200 --batch_size 32 --patch_size 256 --num_workers 8 --lr 1e-3 --patches_per_epoch 4000 --dark_noise_std 0 (other defaults from config)
Inference findings on bootao2PNG
data/testing/bootao2PNG: 24 unique images × 4 duplicates = 96 files. Has predicted_light_directions.txt (algorithmic estimates) but no ground-truth light_directions.txt.
Tested with 3 explicit images (003, 011, 019) + --use_model_mask.
Model mask is unreliable: outputs 98.3% foreground on this input. Either the seg branch still has the issue flagged for best_trans.pt, the 0.5 threshold is too low, or the OOD input (3 single LEDs, model trained on arc-summed) confuses it. Need to inspect output visually.
Open items / known limitations
Test datasets (DiLiGentTestDataset, SyntheticTestDataset, DiLiGentMVTestDataset) still use the old 3-random sampler. Eval numbers (val_loss=9.25) are on the OLD distribution, not on ring-mixed inputs that match training. Need to mirror the new path into these for honest eval.
No --light_dirs <path> CLI flag yet in predict.py. Would let us point at predicted_light_directions.txt without renaming. User interrupted me before adding this.
Model mask reliability for the new checkpoint needs investigation. Possible causes: seg head still under-supervised, threshold tuning, or OOD inputs at inference.
predict.py distribution mismatch. When N=96 lights without light_directions.txt, the legacy 3-of-N selection ([n//6, n//2, 5n//6]) is used. Model was trained on arc-summed channels, not single-LED stacks → OOD.
Env light cancels mathematically in noiseless case (I_sub = (1-α)·Σ b·I_point). The env contribution only matters when dark_noise_std > 0. Doc'd in _ring_mix_channels.
Quick-reference commands
Retrain (clean baseline of new strategy):


python train.py --mode joint_888 --model_type transunet --json_path train_split.json --val_json val_split.json --data_root data/training --epochs 200 --batch_size 32 --patch_size 256 --num_workers 8 --lr 1e-3 --patches_per_epoch 4000 --dark_noise_std 0
Predict (legacy 3-of-N path):


python predict.py --checkpoint checkpoints/joint_888_run/best_joint.pt --input_dir data/testing/bootao2PNG --output ./output/<name> --no_ring_aggregate
Predict (ring-aggregated, requires light_directions.txt in input_dir):


python predict.py --checkpoint checkpoints/joint_888_run/best_joint.pt --input_dir <dir_with_light_directions.txt> --output ./output/<name>
Predict (explicit 3 images + model mask):


python predict.py --checkpoint checkpoints/joint_888_run/best_joint.pt --images <img1> <img2> <img3> --output ./output/<name> --use_model_mask
Style/process notes (carry forward)
Vietnamese-language comments occasionally appear from prior sessions; keep English in new code.
No unnecessary refactoring/cleanup.
Default to no code comments unless WHY is non-obvious.
Honest assessment; don't oversell results.
Spell-check IDE hints on numpy/Otsu/linalg/keepdims/loadtxt/newaxis are Information-level only — ignore.
Paste this into the new device's first prompt alongside the existing CLAUDE.md and you'll have full context.