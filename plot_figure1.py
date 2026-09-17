"""
Figure 1 generator: logical error rate vs physical error rate.

Reads the CSV written by bb_harness.py --csv and produces a publication-style
log-log plot with Wilson-score error bars and a fitted slope.

Usage:  python plot_figure1.py baseline_full.csv
Output: figure1.png (300 dpi) and figure1.pdf (vector, for the paper)

Requires: pip install matplotlib numpy
"""
import sys, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wilson(fails, shots, z=1.0):
    """Wilson score interval -- behaves correctly when fails == 0."""
    if shots == 0:
        return 0.0, 0.0, 0.0
    p = fails / shots
    d = 1 + z**2 / shots
    c = (p + z**2 / (2 * shots)) / d
    h = z * math.sqrt(max(p * (1 - p) / shots + z**2 / (4 * shots**2), 0)) / d
    return p, max(c - h, 0.0), min(c + h, 1.0)

def per_round(x, rounds):
    return 1 - (1 - x) ** (1 / rounds) if x > 0 else 0.0

path = sys.argv[1] if len(sys.argv) > 1 else "baseline_full.csv"
ps, mid, lo, hi, has_data = [], [], [], [], []
with open(path) as fh:
    for row in csv.DictReader(fh):
        p = float(row["p"]); r = int(row["rounds"])
        s = int(row["shots"]); f = int(row["fails"])
        c, a, b = wilson(f, s)
        ps.append(p)
        mid.append(per_round(c, r))
        lo.append(per_round(a, r))
        hi.append(per_round(b, r))
        has_data.append(f > 0)

ps = np.array(ps); mid = np.array(mid)
lo = np.array(lo); hi = np.array(hi); has_data = np.array(has_data)

fig, ax = plt.subplots(figsize=(5.2, 4.0))

m = has_data
ax.errorbar(ps[m], mid[m], yerr=[mid[m] - lo[m], hi[m] - mid[m]],
            fmt="o-", ms=5, lw=1.4, capsize=3, color="#1f4e79",
            label="BP-OSD (baseline)")

if (~m).any():
    ax.errorbar(ps[~m], hi[~m], yerr=[hi[~m] * 0.6, np.zeros((~m).sum())],
                fmt="v", ms=6, lw=1.2, color="#1f4e79", alpha=0.55,
                label="upper bound (0 failures observed)")

if m.sum() >= 2:
    sl, ic = np.polyfit(np.log10(ps[m]), np.log10(mid[m]), 1)
    xf = np.logspace(np.log10(ps.min()), np.log10(ps.max()), 50)
    ax.plot(xf, 10 ** (ic + sl * np.log10(xf)), "--", lw=1.0,
            color="#888888", label=f"fit: slope {sl:.2f}")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("physical error rate $p$")
ax.set_ylabel("logical error rate per round")
ax.set_title("[[72,12,6]] bivariate bicycle code\ncircuit-level noise, 6 rounds", fontsize=10)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("figure1.png", dpi=300)
fig.savefig("figure1.pdf")
print("wrote figure1.png and figure1.pdf")
print(f"\n{'p':>8} {'LER/round':>12} {'68% interval':>26}")
for i in range(len(ps)):
    tag = "" if has_data[i] else "  (upper bound)"
    print(f"{ps[i]:>8.4f} {mid[i]:>12.3e}   [{lo[i]:.2e}, {hi[i]:.2e}]{tag}")
if m.sum() >= 2:
    print(f"\nfitted slope = {sl:.2f}   "
          f"(a distance-d code under circuit noise is expected near {(6+1)//2})")
