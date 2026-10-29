#!/usr/bin/env python3
"""
Report LISA hot-spot count and spatial concentration for the 21 FDR features.

For each feature, hot spots are cities with Moran_Local q == 1 (high-high) and
p_sim < 0.05. The script reports the number of hot spots, their maximum pairwise
distance (spread in km), and the top activating hot-spot cities.

Pure CPU; reads local caches, does not load the model.
"""
import numpy as np
from pathlib import Path
from libpysal.weights import W
from esda.moran import Moran_Local

ROOT = Path(__file__).resolve().parent.parent
acts = np.load(str(ROOT / "sae/activations/city_sae_acts.npy"))
coords = np.load(str(ROOT / "sae/activations/city_coords.npy"))
names = np.load(str(ROOT / "sae/activations/city_names.npy"), allow_pickle=True)
lon = coords[:, 0]
lat = coords[:, 1]

FIDS = [425, 1665, 8964, 9302, 13664, 17719, 21483, 22560, 23310, 34353,
        37159, 38144, 39778, 43181, 44266, 44842, 45671, 56981, 57830, 58472, 58833]


def build_w(coords, k=8):
    lo = np.radians(coords[:, 0])
    la = np.radians(coords[:, 1])
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(dist, np.inf)
    nb, wt = {}, {}
    for i in range(len(coords)):
        nb[i] = np.argsort(dist[i])[:k].tolist()
        wt[i] = [1.0] * k
    return W(neighbors=nb, weights=wt)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    lo1, la1, lo2, la2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlo, dla = lo2 - lo1, la2 - la1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    w = build_w(coords, 8)
    rows = []
    for fid in FIDS:
        local = Moran_Local(acts[:, fid].astype(np.float64), w, permutations=999, seed=0)
        hot = np.where((local.q == 1) & (local.p_sim < 0.05))[0]
        n_hot = len(hot)
        spread = 0.0
        if n_hot >= 2:
            for a in range(n_hot):
                for b in range(a + 1, n_hot):
                    d = haversine_km(lon[hot[a]], lat[hot[a]], lon[hot[b]], lat[hot[b]])
                    spread = max(spread, d)
        # top 6 hot-spot cities by activation, for region identification
        hot_sorted = hot[np.argsort(-acts[hot, fid])]
        top = [str(names[i]) for i in hot_sorted[:6]]
        rows.append((fid, n_hot, spread, top))

    print(f"{'fid':>6} {'n_hot':>6} {'spread(km)':>10}  top cities")
    for fid, n_hot, spread, top in sorted(rows, key=lambda x: (x[2], x[1])):
        print(f"#{fid:>5} {n_hot:>6} {spread:>10.0f}  {', '.join(top)}")


if __name__ == "__main__":
    main()
