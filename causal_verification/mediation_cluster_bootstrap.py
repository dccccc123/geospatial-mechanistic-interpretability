"""
Pair-level cluster bootstrap for the mediation analysis.

The upstream mediation (geographic -> direction feature, via the routing heads)
uses the same 180 prompts (90 undirected pairs), so it has the same pair
dependence. This script re-estimates the significance of the total and
head-mediated indirect effects with a recentered one-sided cluster bootstrap.

Pure CPU; reads cached results.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
RES_DIR = ROOT / "causal_verification" / "results"
OUT = ROOT / "causal_verification" / "results" / "mediation_cluster_bootstrap.json"

UPSTREAM = RES_DIR / "l8_geographic_to_l26_direction_mediation.json"

B_BOOT = 10000
SEED = 42


def load_cluster_id():
    """Group the 180 prompts into 90 undirected pairs; return cluster ids (same as the ablation scripts)."""
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    pair_to_idx = defaultdict(list)
    for i, r in enumerate(rows):
        pair_to_idx[frozenset([r["City_A"], r["City_B"]])].append(i)
    assert len(pair_to_idx) == 90, f"expected 90 pairs, got {len(pair_to_idx)}"
    assert all(len(v) == 2 for v in pair_to_idx.values()), "found an undirected pair with != 2 prompts"
    pair_keys = sorted(pair_to_idx.keys(), key=lambda k: min(pair_to_idx[k]))
    cluster_id = np.zeros(len(rows), dtype=np.int64)
    for c, k in enumerate(pair_keys):
        i, j = pair_to_idx[k]
        cluster_id[i] = cluster_id[j] = c
    return cluster_id


def cluster_bootstrap(effects, cluster_id, side):
    """Recentered cluster bootstrap (same as cluster_bootstrap.py)."""
    effects = np.asarray(effects, dtype=np.float64)
    cluster_id = np.asarray(cluster_id, dtype=np.int64)
    n_clusters = int(cluster_id.max()) + 1
    order = np.argsort(cluster_id, kind="stable")
    e = effects[order].reshape(n_clusters, -1)

    obs_mean = float(effects.mean())
    obs_median = float(np.median(effects))

    rng = np.random.default_rng(SEED)
    cl = rng.integers(0, n_clusters, size=(B_BOOT, n_clusters))

    boot_means = e[cl].reshape(B_BOOT, -1).mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    centered = e - obs_mean
    boot_centered = centered[cl].reshape(B_BOOT, -1).mean(axis=1)
    if side == "less":
        p = float((boot_centered <= obs_mean).mean())
    elif side == "greater":
        p = float((boot_centered >= obs_mean).mean())
    else:
        p = float((np.abs(boot_centered) >= abs(obs_mean)).mean())

    return {
        "observed_mean": obs_mean,
        "observed_median": obs_median,
        "bootstrap_mean_ci95": [float(lo), float(hi)],
        "p_value_one_sided": p,
        "side": side,
        "n_clusters": n_clusters,
        "n_observations": len(effects),
        "B": B_BOOT,
    }


def main():
    cluster_id = load_cluster_id()
    out = {}

    # ── upstream mediation (geo -> direction feature, via routing heads)─────────
    up = json.load(open(UPSTREAM))
    per = up["per_prompt"]
    delta_total = np.array([r["delta_total"] for r in per], dtype=np.float64)
    delta_indirect = np.array([r["delta_indirect"] for r in per], dtype=np.float64)

    out["upstream_geo_to_direction"] = {
        "total_effect_delta_total": cluster_bootstrap(delta_total, cluster_id, side="less"),
        "indirect_effect_through_heads": cluster_bootstrap(delta_indirect, cluster_id, side="less"),
        "mediated_fraction_median_ratio": up["summary"].get("mediated_fraction_median_ratio"),
    }

    # ── print ───────────────────────────────────────
    print("=== upstream mediation (geo -> direction feature, via routing heads) ===")
    for name, key in [("total effect delta_total (<0)", "total_effect_delta_total"),
                      ("indirect effect through heads (<0)", "indirect_effect_through_heads")]:
        b = out["upstream_geo_to_direction"][key]
        print(f"  {name}: mean={b['observed_mean']:+.4f}  median={b['observed_median']:+.4f}"
              f"  95%CI=[{b['bootstrap_mean_ci95'][0]:+.4f}, {b['bootstrap_mean_ci95'][1]:+.4f}]"
              f"  p={b['p_value_one_sided']:.4g}")
    print(f"  fraction through heads (median ratio): {out['upstream_geo_to_direction']['mediated_fraction_median_ratio']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
