"""Generate the fidelity/memory vs budget figure from curve28.csv."""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/curve28.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "../paper/preprint/curve28.pdf"

rows = list(csv.DictReader(open(src)))
# drop the exact budget=0 point from the log axis (noted in caption)
rows = [r for r in rows if float(r["budget"]) > 0]
b = [float(r["budget"]) for r in rows]
meas = [float(r["echo_fidelity"]) for r in rows]
est = [float(r["estimate"]) for r in rows]
bound = [float(r["bound"]) for r in rows]
mem = [float(r["peak_mb"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
ax1.set_xscale("log")
ax1.set_xlabel("compression budget $D$")
ax1.set_ylabel("fidelity")
ax1.set_ylim(-0.02, 1.05)
ax1.plot(b, meas, "o-", color="#1b7837", label="measured", zorder=3)
ax1.plot(b, est, "s--", color="#2166ac", label="estimate")
ax1.plot(b, bound, "^:", color="#b2182b", label="worst-case bound")
ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

ax2 = ax1.twinx()
ax2.set_ylabel("peak memory (MB)")
ax2.set_ylim(0, 2200)
ax2.plot(b, mem, "d-", color="#888888", alpha=0.7, label="peak MB")
ax2.legend(loc="center right", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
