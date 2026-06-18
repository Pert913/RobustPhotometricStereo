import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Foreground IoU (%) as stronger geometric priors are folded into the mask pipeline.
labels = [
    "Raw seg head\n(threshold)",
    "+ brightness\n(Otsu) seed",
    "+ seed-and-grow\n(hysteresis)",
    "Single-pass +\nconn. component",
]
iou = [13.0, 57.6, 71.2, 95.0]
colors = ["#bdbdbd", "#fdae61", "#74add1", "#1a9850"]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
bars = ax.bar(labels, iou, color=colors, edgecolor="black", linewidth=0.6, width=0.65)
for b, v in zip(bars, iou):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.set_ylabel(r"Foreground IoU (\%)" if matplotlib.rcParams["text.usetex"] else "Foreground IoU (%)")
ax.set_ylim(0, 105)
ax.set_title("Foreground-masking strategy (DiLiGenT-MV, TransUNetPS)")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("mask_ablation.png", dpi=160, bbox_inches="tight")
print("saved mask_ablation.png")
