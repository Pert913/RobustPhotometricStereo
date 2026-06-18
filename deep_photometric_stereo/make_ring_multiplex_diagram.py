"""Schematic of the 24-LED ring -> RGB-arc multiplexing -> 3-channel input tensor."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, Rectangle, Circle

R_RED, R_GRN, R_BLU = "#d62728", "#2ca02c", "#1f77b4"

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
ax.set_aspect("equal")

# ---- Left: the LED ring -----------------------------------------------------
cx, cy, r = 2.5, 2.5, 1.7
# faint arc bands (each spans 8 LEDs = 120 deg). LED i at angle -90 + i*15.
for (t1, t2, col) in [(-97.5, 22.5, R_RED), (22.5, 142.5, R_GRN), (142.5, 262.5, R_BLU)]:
    ax.add_patch(Wedge((cx, cy), r + 0.32, t1, t2, width=0.64, facecolor=col, alpha=0.18, edgecolor="none"))

# 24 LEDs, coloured by arc (0-7 R, 8-15 G, 16-23 B)
for i in range(24):
    ang = np.deg2rad(-90 + i * 15)
    x, y = cx + r * np.cos(ang), cy + r * np.sin(ang)
    col = R_RED if i < 8 else (R_GRN if i < 16 else R_BLU)
    ax.add_patch(Circle((x, y), 0.13, facecolor=col, edgecolor="black", lw=0.5, zorder=3))

# object/camera at centre
ax.add_patch(Circle((cx, cy), 0.5, facecolor="#dddddd", edgecolor="black", lw=0.8, zorder=2))
ax.text(cx, cy, "object", ha="center", va="center", fontsize=8)
ax.text(cx, cy - r - 0.75, "24-LED ring (3 colour arcs)", ha="center", va="center", fontsize=10, fontweight="bold")
# arc labels
ax.text(cx, cy - r - 0.30, "R arc (8)", ha="center", va="center", fontsize=8, color=R_RED)
ax.text(cx - r - 0.30, cy + 0.6, "B arc (8)", ha="center", va="center", fontsize=8, color=R_BLU, rotation=90)
ax.text(cx + r - 0.1, cy + 0.95, "G arc (8)", ha="center", va="center", fontsize=8, color=R_GRN, rotation=-60)

# ---- Right: the 3-channel input tensor (stacked planes) ---------------------
bx, by, w, h, dx, dy = 6.7, 1.7, 1.7, 1.7, 0.32, 0.32
planes = [(R_BLU, "B"), (R_GRN, "G"), (R_RED, "R")]  # draw back-to-front
for k, (col, lab) in enumerate(planes):
    ax.add_patch(Rectangle((bx + k * dx, by + k * dy), w, h, facecolor=col, alpha=0.55,
                           edgecolor="black", lw=1.0, zorder=3 + k))
    ax.text(bx + k * dx + 0.18, by + k * dy + h - 0.22, lab, fontsize=11, fontweight="bold",
            color="white", zorder=10)
ax.text(bx + w / 2 + dx, by - 0.5, "3-channel RGB input\n(one channel per arc)",
        ha="center", va="center", fontsize=10, fontweight="bold")

# ---- Arrows: each arc -> its channel ---------------------------------------
def arrow(p0, p1, col):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 lw=2.0, color=col, zorder=12,
                                 connectionstyle="arc3,rad=0.0"))

arrow((cx + r + 0.1, cy - 0.9), (bx + 0.05, by + 0.35), R_RED)        # R arc -> R plane
arrow((cx + r + 0.1, cy + 0.2), (bx + dx + 0.05, by + dy + h / 2), R_GRN)  # G
arrow((cx + r + 0.1, cy + 1.1), (bx + 2 * dx + 0.05, by + 2 * dy + h - 0.35), R_BLU)  # B
ax.text((cx + r + bx) / 2 + 0.2, cy + 1.75, "per-arc combine\n(random weights, training)",
        ha="center", va="center", fontsize=8, style="italic")

fig.tight_layout()
fig.savefig("ring_multiplex.png", dpi=160, bbox_inches="tight")
print("saved ring_multiplex.png")
