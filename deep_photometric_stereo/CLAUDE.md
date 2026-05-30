1. Model head — model.py:188-216 (MultiTaskHead)

Decoupled head: one branch produces the 3 normal channels, the other the single mask logit. They share the decoder but have independent final convolutions to prevent gradient interference. Output is a 4-channel tensor [Nx, Ny, Nz, mask_logit].

All three model variants (TransUNetPS, LightweightUNetPS, SwinUNetPS) use the same MultiTaskHead (model.py:261, model.py:340, model.py:410).

2. Training loss — losses.py:212-232 (MultiTaskLoss)


pred_normal = pred_4c[:, :3, :, :]
pred_mask_logit = pred_4c[:, 3:, :, :]
loss_normal = self.normal_loss(pred_normal, target_normal, target_mask)
loss_seg = self.seg_loss(pred_mask_logit, target_mask.float())   # BCEWithLogitsLoss
total_loss = loss_normal + (self.seg_weight * loss_seg)
Wired up in train.py:219 and train.py:387 with seg_weight=0.5. The dataset's mask_tensor (returned as the third element of __getitem__) is the ground-truth supervision target.

The SegmentationLoss / DiceLoss classes at losses.py:149-209 are for Synapse multi-organ segmentation only (a separate mode, not used in the PS training).

3. Inference — predict.py:176-328

This is where it gets important to understand the project state. The CLAUDE.md note says:

best_trans.pt has legacy coupled mask head — predictions unreliable, do NOT use as primary mask.

So at inference, predict.py bypasses the model's mask head entirely and builds the mask from photometric heuristics:

predict.py:289-294: picks Otsu's method on luminance (or dark-frame subtraction max(lit - dark, 0) when a dark frame is supplied) — the model's seg branch output is not used.
predict.py:300-326: threshold floor at max(0.08, 5th_pct * 1.5), then binary_closing + binary_fill_holes + connected-component size filter + Gaussian blur for a soft mask.
predict.py:327: the soft mask multiplies into pred_normal to zero-out background.
The seg branch is essentially being trained but ignored at inference. The --no_pred_mask flag mentioned in CLAUDE.md is the explicit knob that documents this.

4. Dataset supervision target — dataset.py:249, dataset.py:567, dataset.py:846

Each __getitem__ returns (img, normal, mask) where mask is (1, H, W) float32. That mask is what BCEWithLogitsLoss trains the model's seg head against. The masks come from the on-disk ground truth (mask.npy for DiLiGenT, binary_mask.exr for Synthetic).

TL;DR: the seg head exists, is supervised with BCE during training, but is overridden at inference by the Otsu / dark-ref pipeline because the legacy checkpoint's seg branch was unreliable. If you re-train from scratch (which you're about to), the new checkpoint's seg head should actually be usable — at that point you could drop the Otsu fallback and trust the model's mask. Worth re-evaluating after the new run finishes.