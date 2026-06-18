import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = {
    "TransUNetPS": "checkpoints/joint_888_run_trans_25Images/training_log_trans_new.csv",
    "LightweightUNetPS": "checkpoints/joint_888_run_lw_25Images/training_log_lw_new.csv",
    "GA-UNetPS": "checkpoints/joint_888_run_swin_25_tmp/training_log.csv",
}
COLORS = {"TransUNetPS": "#1f77b4", "LightweightUNetPS": "#2ca02c", "GA-UNetPS": "#d62728"}


def load(path):
    ep, tr, vm = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            e = int(row["Epoch"])
            ep.append(e)
            tr.append((e, float(row["Train_Loss"])))
            if row["Val_MAE"]:
                vm.append((e, float(row["Val_MAE"])))
    return tr, vm


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for name, path in LOGS.items():
    tr, vm = load(path)
    c = COLORS[name]
    ax1.plot([e for e, _ in tr], [v for _, v in tr], color=c, label=name, lw=1.4)
    ax2.plot([e for e, _ in vm], [v for _, v in vm], color=c, marker="o", ms=3, label=name, lw=1.4)
    # mark best val MAE
    be, bv = min(vm, key=lambda x: x[1])
    ax2.scatter([be], [bv], color=c, s=55, zorder=5, edgecolor="black", linewidth=0.6)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Composite training loss")
ax1.set_title("(a) Training loss")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

ax2.set_xlabel("Epoch")
ax2.set_ylabel(r"Validation MAE ($^\circ$)")
ax2.set_title("(b) Validation MAE")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

fig.tight_layout()
out = "training_curve.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
