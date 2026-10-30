import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as pe
import geopandas as gpd
from esda.moran import Moran_Local
from libpysal.weights import W
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

RENAME = {
    "Kashgar": "Kaxgar", "Artux": "Atux", "Ürümqi": "Urumqi",
    "Ghulja": "Yining", "Aqsu": "Akesu", "Tumxuk": "Tumushuke",
    "Rikaze": "Xi'gazê", "Qamdo": "Qabdo", "Lhoka": "Shannan",
    "Qiqihar": "Qiqihaer", "Shangri-La": "Xianggelila",
}

acts = np.load(str(ROOT / "sae/activations/city_sae_acts.npy"))
coords = np.load(str(ROOT / "sae/activations/city_coords.npy"))
names = np.load(str(ROOT / "sae/activations/city_names.npy"), allow_pickle=True)
lon = coords[:, 0]; lat = coords[:, 1]

prov = gpd.read_file("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json")[["name", "geometry"]].copy()
prov["geometry"] = prov["geometry"].buffer(0)


def build_w(coords, k=8):
    lo = np.radians(coords[:, 0]); la = np.radians(coords[:, 1])
    a = np.sin((la[:, None] - la[None, :]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))); np.fill_diagonal(dist, np.inf)
    nb, wt = {}, {}
    for i in range(len(coords)):
        nb[i] = np.argsort(dist[i])[:k].tolist(); wt[i] = [1.0] * k
    return W(neighbors=nb, weights=wt)


w = build_w(coords, 8)
show = [(37159, "l8ex_a_northwest_xinjiang"), (425, "l8ex_b_southwest_tibet"),
        (22560, "l8ex_c_east_yangtze"),(9302, "l8ex_d_southeast_china_fujian"),
        (1665, "l8ex_e_south_guangxi"), (34353, "l8ex_f_southwest_sichuan_yunnan")]

W_IN, H_IN = 4.0, 4.3
R = W_IN / H_IN

for fid, slug in show:
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    fig.patch.set_facecolor("#FAFAF8"); ax.set_facecolor("#FAFAF8")
    local = Moran_Local(acts[:, fid].astype(np.float64), w, permutations=999, seed=0)
    hot = np.where((local.q == 1) & (local.p_sim < 0.05))[0]
    hlon, hlat = lon[hot], lat[hot]
    cx, cy = hlon.mean(), hlat.mean()
    s = 1.0 / np.cos(np.radians(cy))
    hlat0 = max((hlat.max() - hlat.min()) / 2, 2.0) + 1.6
    hlon0 = max((hlon.max() - hlon.min()) / 2, 2.0) + 1.6 * s
    target = R * s
    if hlon0 / hlat0 < target:
        half_lon, half_lat = hlat0 * target, hlat0
    else:
        half_lon, half_lat = hlon0, hlon0 / target
    half_lon *= 1.04; half_lat *= 1.04
    prov.plot(ax=ax, facecolor="#ECE5D8", edgecolor="#C4BCB0", linewidth=0.6, zorder=1, aspect=None)
    ax.scatter(hlon, hlat, s=64, c="#E07B39", edgecolors="#2D1B0E", linewidths=0.6, zorder=3)
    ax.set_xlim(cx - half_lon, cx + half_lon); ax.set_ylim(cy - half_lat, cy + half_lat)
    ax.set_aspect(s)
    texts = [ax.text(lon[i], lat[i], RENAME.get(str(names[i]), str(names[i])), fontsize=8, color="#2D1B0E",
                     zorder=5, clip_on=False, path_effects=[pe.withStroke(linewidth=2.2, foreground="#FAFAF8")])
             for i in hot]
    adjust_text(texts, ax=ax, expand=(1.15, 1.4), force_text=(0.4, 0.6),
                arrowprops=dict(arrowstyle="-", color="#9a8f80", lw=0.5))
    # ── geographic context: lon/lat tick units, scale bar, north arrow ──
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(labelsize=7, colors="#6B6055")

    # Scale bar (pick a "nice" km length near ~20% of the panel width)
    km_per_deg_lon = 111.32 * np.cos(np.radians(cy))
    target_km = (2 * half_lon) * 0.2 * km_per_deg_lon
    nice_steps = [10, 20, 50, 100, 200, 500, 1000]
    scale_km = min(nice_steps, key=lambda s: abs(s - target_km))
    scale_deg = scale_km / km_per_deg_lon
    scalebar = AnchoredSizeBar(
        ax.transData, scale_deg, f"{scale_km:.0f} km",
        loc="lower right", pad=0.3, borderpad=0.4, sep=4,
        frameon=False, size_vertical=0.15,
        fontproperties=fm.FontProperties(size=7),
    )
    ax.add_artist(scalebar)

    # North arrow
    ax.annotate(
        "N", xy=(0.93, 0.90), xytext=(0.93, 0.79),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color="#2D1B0E", lw=1.3),
        fontsize=9, fontweight="bold", zorder=6,
    )

    for sp in ax.spines.values():
        sp.set_color("#C4BCB0")
    plt.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.006)
    out_path = OUT_DIR / f"{slug}.png"
    plt.savefig(out_path, dpi=200, facecolor="#FAFAF8")
    plt.close(fig)
    print("saved", out_path)
