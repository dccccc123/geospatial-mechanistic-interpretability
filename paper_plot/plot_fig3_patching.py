#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig 3: activation-patching heatmap (position x layer), paper palette.

Rows: c_A (mirror run), c_B (original run), the (original run).
Cell value = normalized metric m (lower = more critical).
Designed larger than column width; place in paper with width=\\linewidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ---- paper palette ----
ORANGE, LIGHT     = "#E07B39", "#FDF4EC"   # geographic gradient (low m -> orange = critical)
ESPRESSO          = "#2D1B0E"              # L26 accent
WARM_WHITE        = "#FAFAF8"
STONE, GRAY, DARK = "#C4BCB0", "#6B6055", "#1E1410"
cmap = LinearSegmentedColormap.from_list("patch", [ORANGE, LIGHT])

# ---- data ----
BASE = Path(__file__).resolve().parent.parent / "component_localization"
o  = np.load(BASE / "patching_results.npz")        # c_B-flip run
mr = np.load(BASE / "patching_results_cityA.npz")  # c_A-flip run
cA  = mr["metric_mlp_city_a"]
cB  = o["metric_mlp_city_b"]
the = o["metric_mlp_final"]
nL  = len(cB)
M   = np.stack([cA, cB, the], axis=0)   # rows: c_A, c_B, the ; cols: layers
print("m range:", round(float(M.min()), 3), "-", round(float(M.max()), 3))

# ---- plot ----
plt.rcParams.update({"font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(5.6, 2.5))
fig.patch.set_facecolor(WARM_WHITE)
ax.set_facecolor(WARM_WHITE)

im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=float(M.min()), vmax=1.0)

ax.set_yticks([0, 1, 2])
ax.set_yticklabels([r"$c_A$", r"$c_B$", "the"], fontsize=14, color=DARK)
ax.set_xticks(range(0, nL, 4))
ax.set_xticklabels(range(0, nL, 4), fontsize=11, color=GRAY)
ax.set_xlabel("Layer", fontsize=13, color=DARK)
ax.tick_params(length=2.5, colors=STONE)
for sp in ax.spines.values():
    sp.set_edgecolor(STONE)

# key layers: highlight box + top label (L8 orange, L26 espresso)
for L, col, lab in [(8, ORANGE, "L8"), (26, ESPRESSO, "L26")]:
    # full-column box (border only, no fill, to avoid changing heatmap colors)
    ax.axvspan(L - 0.5, L + 0.5, facecolor='none', edgecolor=col, linewidth=2.5)
    ax.text(L, -0.85, lab, color=col, fontsize=12, fontweight="bold",
            ha="center", va="bottom", clip_on=False)

cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
cb.set_label(r"$m$", fontsize=14, color=DARK, rotation=0, labelpad=8, va="center")
cb.ax.tick_params(labelsize=10, colors=STONE)
cb.outline.set_edgecolor(STONE)

plt.tight_layout()
out = Path(__file__).resolve().parent / "fig3_patching_heatmap.pdf"
plt.savefig(out, bbox_inches="tight", facecolor=WARM_WHITE)
print("saved", out)
