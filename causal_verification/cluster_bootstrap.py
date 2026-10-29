"""
Pair-level cluster bootstrap and reversal-consistency check for the ablations.

The 180 directed prompts are formed by reversing 90 undirected city pairs, so the
samples are not independent. This script groups prompts by undirected pair,
re-estimates effect significance with a recentered one-sided cluster bootstrap
(95% percentile CI + p-value), and reports the model's reversal consistency
across paired prompts.

Covers the L8 geographic-feature ablation, the L26 direction-feature ablation,
and the Head 24 / Head 25 ablations. Pure CPU; reads cached results.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
PREDS = ROOT / "linear_probe" / "activations" / "baseline_preds.npy"
RES_DIR = ROOT / "causal_verification" / "results"
OUT = ROOT / "causal_verification" / "results" / "cluster_bootstrap.json"

L26 = RES_DIR / "l26_direction_feature_ablation.json"
L8_JOINT = RES_DIR / "l8_geographic_subspace_ablation_ca_joint.json"
HEAD = RES_DIR / "head_ablation_delta_logit.json"

DIR_LABELS = ["East", "West", "South", "North"]
OPPOSITE = {"East": "West", "West": "East", "South": "North", "North": "South"}
B_BOOT = 10000
SEED = 42


def load_pairs_and_clusters():
    """Group the 180 prompts into 90 undirected city pairs; return cluster ids and reversal consistency."""
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    preds = np.load(PREDS)  # [180],0=East,1=West,2=South,3=North

    pair_to_idx = defaultdict(list)
    for i, r in enumerate(rows):
        pair_to_idx[frozenset([r["City_A"], r["City_B"]])].append(i)
    assert len(pair_to_idx) == 90, f"expected 90 pairs, got {len(pair_to_idx)}"
    assert all(len(v) == 2 for v in pair_to_idx.values()), "found an undirected pair with != 2 prompts"

    pair_keys = sorted(pair_to_idx.keys(), key=lambda k: min(pair_to_idx[k]))
    cluster_id = np.zeros(len(rows), dtype=np.int64)
    n_cons = 0
    per_dir_cons = {d: [0, 0] for d in DIR_LABELS}
    for c, k in enumerate(pair_keys):
        i, j = pair_to_idx[k]
        cluster_id[i] = cluster_id[j] = c
        pred_i = DIR_LABELS[int(preds[i])]
        pred_j = DIR_LABELS[int(preds[j])]
        consistent = pred_j == OPPOSITE[pred_i]
        n_cons += int(consistent)
        d = rows[i]["Direction"]
        per_dir_cons[d][0] += int(consistent)
        per_dir_cons[d][1] += 1

    reversal = {
        "n_pairs": 90,
        "n_consistent": int(n_cons),
        "consistency_rate": float(n_cons / 90),
        "per_direction": {d: per_dir_cons[d][0] / per_dir_cons[d][1] for d in DIR_LABELS},
    }
    return cluster_id, reversal


def cluster_bootstrap(effects, cluster_id, side="less"):
    """
    Pair-level cluster bootstrap: resample the 90 undirected pairs with replacement.

    effects: [N] per-prompt effect; cluster_id: [N] cluster index.
    Returns observed mean/median, 95% percentile CI, and a one-sided p-value
    (recentered bootstrap).
    """
    effects = np.asarray(effects, dtype=np.float64)
    cluster_id = np.asarray(cluster_id, dtype=np.int64)
    n_clusters = int(cluster_id.max()) + 1
    order = np.argsort(cluster_id, kind="stable")
    e = effects[order].reshape(n_clusters, -1)  # [n_clusters, 2]

    obs_mean = float(effects.mean())
    obs_median = float(np.median(effects))

    rng = np.random.default_rng(SEED)
    cl = rng.integers(0, n_clusters, size=(B_BOOT, n_clusters))  # [B, n_clusters]

    boot_means = e[cl].reshape(B_BOOT, -1).mean(axis=1)          # [B]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    # recentered bootstrap: H0 mean = 0, one-sided test
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
        "p_value_one_sided_less": p,
        "n_clusters": n_clusters,
        "n_observations": len(effects),
        "B": B_BOOT,
    }


def report(name, main_effect, diff_effect=None, old_p=None):
    print(f"\n=== {name} ===")
    print(f"  main effect: mean={main_effect['observed_mean']:+.4f}  median={main_effect['observed_median']:+.4f}"
          f"  95%CI=[{main_effect['bootstrap_mean_ci95'][0]:+.4f}, {main_effect['bootstrap_mean_ci95'][1]:+.4f}]"
          f"  cluster-bootstrap p={main_effect['p_value_one_sided_less']:.4g}")
    if old_p is not None:
        print(f"  (old Wilcoxon p={old_p:.4g})")
    if diff_effect is not None:
        print(f"  target-control: mean={diff_effect['observed_mean']:+.4f}"
              f"  95%CI=[{diff_effect['bootstrap_mean_ci95'][0]:+.4f}, {diff_effect['bootstrap_mean_ci95'][1]:+.4f}]"
              f"  cluster-bootstrap p={diff_effect['p_value_one_sided_less']:.4g}")


def main():
    cluster_id, reversal = load_pairs_and_clusters()
    print("[Reversal consistency]")
    print(f"  90 pairs with self-consistent predicted direction (reverse = opposite of forward): {reversal['n_consistent']}  "
          f"consistency={reversal['consistency_rate']*100:.1f}%")
    for d in DIR_LABELS:
        print(f"    {d}: {reversal['per_direction'][d]*100:.1f}%")

    out = {"reversal_consistency": reversal}

    # ── L26 direction-feature ablation ─────────────────────────────
    l26 = json.load(open(L26))
    per = l26["per_prompt"]
    target = np.array([r["delta_target_from_recon"] for r in per])
    control = np.array([r["delta_control_from_recon_mean"] for r in per])
    out["l26_direction_feature"] = {
        "main_effect_target": cluster_bootstrap(target, cluster_id),
        "diff_effect_target_minus_control": cluster_bootstrap(target - control, cluster_id),
        "old_wilcoxon_p": l26["summary"].get("wilcoxon_p_target_less_than_control_recon"),
    }
    report("L26 direction-feature ablation", out["l26_direction_feature"]["main_effect_target"],
           out["l26_direction_feature"]["diff_effect_target_minus_control"],
           out["l26_direction_feature"]["old_wilcoxon_p"])

    # ── L8 geographic-feature ablation (joint c_A+c_B, 21 features)───────────────────────────
    l8 = json.load(open(L8_JOINT))
    per = l8["per_prompt"]
    geo = np.array([r["delta_geo_AB_from_recon"] for r in per])
    out["l8_geographic_feature"] = {
        "main_effect_geo": cluster_bootstrap(geo, cluster_id),
        "old_wilcoxon_p": l8["summary"].get("c_A_c_B_joint", {}).get("wilcoxon_p_less_than_0"),
    }
    report("L8 geographic-feature ablation (joint)", out["l8_geographic_feature"]["main_effect_geo"],
           old_p=out["l8_geographic_feature"]["old_wilcoxon_p"])

    # ── Head 24 / Head 25 ablation ──────────────────────
    head = json.load(open(HEAD))
    for h in ("24", "25"):
        deltas = np.array(head["target_results"][h]["delta_scores"])
        out[f"head_{h}"] = {
            "main_effect_delta_logit": cluster_bootstrap(deltas, cluster_id),
            "old_wilcoxon_p": head["target_results"][h].get(
                "paired_wilcoxon_p_target_less_than_control_mean"),
        }
        report(f"Head {h} ablation", out[f"head_{h}"]["main_effect_delta_logit"],
               old_p=out[f"head_{h}"]["old_wilcoxon_p"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
