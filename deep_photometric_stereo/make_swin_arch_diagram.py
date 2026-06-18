"""Block diagram of the Swin-UnetPS architecture (Swin encoder + reused conv decoder)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_IO    = "#455a64"   # input / output
C_STEM  = "#9e9e9e"   # conv stem
C_SWIN  = "#2a9d8f"   # swin encoder stages
C_BN    = "#1d6f66"   # bottleneck
C_DEC   = "#e76f51"   # decoder blocks
C_HEAD  = "#3a7d44"   # multitask head
C_SKIP  = "#888888"   # skip connections

fig, ax = plt.subplots(figsize=(10.5, 8.2))
ax.set_xlim(0, 10); ax.set_ylim(1.0, 10.2); ax.axis("off")

W, H = 3.0, 0.82
xL, xR = 2.2, 7.8          # encoder / decoder column centres


def box(x, y, text, color, w=W, h=H, tcol="white", fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=color, edgecolor="black", linewidth=0.9, zorder=3))
    ax.text(x, y, text, ha="center", va="center", color=tcol, fontsize=fs, zorder=4, fontweight="bold")


def arrow(p0, p1, color="black", style="-|>", lw=1.8, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=lw,
                 color=color, zorder=2, linestyle="--" if dashed else "-",
                 connectionstyle=f"arc3,rad={rad}"))


# row y-positions keyed by stride
y_in, y1, y2, y4, y8, y16, y32 = 9.5, 8.4, 7.2, 6.0, 4.8, 3.6, 2.1

# ---- Encoder (left, downsampling) ----
box(xL, y_in, "Input: 3-ch RGB-multiplexed", C_IO)
box(xL, y1,  "Conv stem  ·  s1  ·  32", C_STEM)
box(xL, y2,  "Conv stem  ·  s2  ·  64", C_STEM)
box(xL, y4,  "Swin Stage 1  ·  s4  ·  96\n(W-MSA / SW-MSA)", C_SWIN, fs=8)
box(xL, y8,  "Swin Stage 2  ·  s8  ·  192\n(Patch Merging)", C_SWIN, fs=8)
box(xL, y16, "Swin Stage 3  ·  s16  ·  384\n(Patch Merging)", C_SWIN, fs=8)

# ---- Bottleneck (centre bottom) ----
box(5.0, y32, "Swin Stage 4 (bottleneck)  ·  s32  ·  768  ·  (Patch Merging)",
    C_BN, w=4.6, fs=8)

# ---- Decoder (right, upsampling) ----
box(xR, y16, "Decoder block  ·  s16  ·  256", C_DEC, fs=8)
box(xR, y8,  "Decoder block  ·  s8  ·  128", C_DEC, fs=8)
box(xR, y4,  "Decoder block  ·  s4  ·  64", C_DEC, fs=8)
box(xR, y2,  "Decoder block  ·  s2  ·  32", C_DEC, fs=8)
box(xR, y1,  "Decoder block  ·  s1  ·  32", C_DEC, fs=8)
box(xR, y_in, "MultiTaskHead\nNormal (3-ch)  +  Mask (1-ch)", C_HEAD, fs=8)

# ---- Encoder downward flow ----
for ya, yb in [(y_in, y1), (y1, y2), (y2, y4), (y4, y8), (y8, y16)]:
    arrow((xL, ya - H / 2), (xL, yb + H / 2), color="black")
arrow((xL, y16 - H / 2), (5.0 - 1.7, y32 + H / 2 + 0.05), color="black", rad=-0.15)  # S3 -> bottleneck

# ---- Decoder upward flow ----
arrow((5.0 + 1.7, y32 + H / 2 + 0.05), (xR, y16 - H / 2), color="black", rad=0.15)   # bottleneck -> up5
for ya, yb in [(y16, y8), (y8, y4), (y4, y2), (y2, y1), (y1, y_in)]:
    arrow((xR, ya + H / 2), (xR, yb - H / 2), color="black")

# ---- Skip connections (encoder -> decoder, same stride) ----
for y in [y1, y2, y4, y8, y16]:
    arrow((xL + W / 2, y), (xR - W / 2, y), color=C_SKIP, lw=1.6, dashed=True)
ax.text(5.0, y1 + 0.34, "skip connections (concat)", ha="center", va="bottom",
        fontsize=8, color=C_SKIP, style="italic")

# ---- legend ----
handles = [
    plt.Line2D([0], [0], marker="s", ls="", mfc=C_SWIN, mec="black", ms=11, label="Swin Transformer encoder"),
    plt.Line2D([0], [0], marker="s", ls="", mfc=C_DEC, mec="black", ms=11, label="Reused conv decoder + head"),
    plt.Line2D([0], [0], marker="s", ls="", mfc=C_STEM, mec="black", ms=11, label="Conv stem (s1/s2 skips)"),
    plt.Line2D([0], [0], ls="--", color=C_SKIP, label="skip connection"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=2, fontsize=8, frameon=False)

ax.set_title("Swin-UnetPS: Swin Transformer backbone with the shared conv decoder",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("swin_arch.png", dpi=160, bbox_inches="tight")
print("saved swin_arch.png")
