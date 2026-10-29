"""
Re-report ablation effects split by correct vs incorrect predictions.

Partitions each ablation experiment's per-prompt results by whether the model's
clean prediction matched the ground-truth direction, and reports the effect size
(mean/median delta logit) and flip rate within each subset. This is a descriptive
grouping; pair-level significance is computed separately by cluster_bootstrap.py.

Pure CPU; reads cached results, does not load the model or re-run ablations.
"""
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "causal_verification" / "results"
OUT = ROOT / "causal_verification" / "results" / "correct_incorrect_split.json"

L26 = RES_DIR / "l26_direction_feature_ablation.json"
L8_JOINT = RES_DIR / "l8_geographic_subspace_ablation_ca_joint.json"
HEAD = RES_DIR / "head_ablation_delta_logit.json"

DIR_LABELS = ["East", "West", "South", "North"]


def group_stats(target, control, mask, flip):
    """Group by mask; return target/control effect and flip-rate statistics."""
    t = np.asarray(target, dtype=np.float64)[mask]
    c = np.asarray(control, dtype=np.float64)[mask]
    f = np.asarray(flip, dtype=np.float64)[mask]
    if len(t) == 0:
        return {"n": 0}
    try:
        _, p = wilcoxon(t, c, alternative="less", zero_method="wilcox")
        p = float(p)
    except Exception:
        p = None
    return {
        "n": int(len(t)),
        "mean_target": float(t.mean()),
        "median_target": float(np.median(t)),
        "mean_control": float(c.mean()),
        "median_control": float(np.median(c)),
        "mean_diff": float((t - c).mean()),
        "median_diff": float(np.median(t - c)),
        "flip_rate": float(f.mean()),
        "wilcoxon_p_target_less_control": p,
    }


def group_stats_onesample(target, mask, flip):
    """Group by mask; one-sample Wilcoxon on the joint delta logit vs recon (delta < 0)."""
    t = np.asarray(target, dtype=np.float64)[mask]
    f = np.asarray(flip, dtype=np.float64)[mask]
    if len(t) == 0:
        return {"n": 0}
    try:
        _, p = wilcoxon(t, alternative="less", zero_method="wilcox")
        p = float(p)
    except Exception:
        p = None
    return {
        "n": int(len(t)),
        "mean_target": float(t.mean()),
        "median_target": float(np.median(t)),
        "negative_rate": float((t < 0).mean()),
        "flip_rate": float(f.mean()),
        "wilcoxon_p_less_than_0": p,
    }


def head_group_stats(delta, mask):
    d = np.asarray(delta, dtype=np.float64)[mask]
    if len(d) == 0:
        return {"n": 0}
    try:
        _, p = wilcoxon(d, alternative="less", zero_method="wilcox")
        p = float(p)
    except Exception:
        p = None
    return {
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "negative_rate": float((d < 0).mean()),
        "wilcoxon_p_less_than_0": p,
    }


def per_direction_accuracy(per_prompt):
    acc = {d: [0, 0] for d in DIR_LABELS}
    for r in per_prompt:
        acc[r["ground_truth"]][1] += 1
        acc[r["ground_truth"]][0] += int(r["pred_dir_clean"] == r["ground_truth"])
    return {d: acc[d][0] / acc[d][1] for d in DIR_LABELS}


def main():
    out = {}

    # ── L26 direction-feature ablation ─────────────────────────────
    per = json.load(open(L26))["per_prompt"]
    correct = np.array([r["pred_dir_clean"] == r["ground_truth"] for r in per])
    target = [r["delta_target_from_recon"] for r in per]
    control = [r["delta_control_from_recon_mean"] for r in per]
    flip = [r["target_flip"] for r in per]
    acc = float(correct.mean())
    out["l26_direction_feature"] = {
        "accuracy": acc,
        "per_direction_accuracy": per_direction_accuracy(per),
        "correct": group_stats(target, control, correct, flip),
        "incorrect": group_stats(target, control, ~correct, flip),
    }
    print(f"\n=== L26 direction-feature ablation(overall accuracy {acc*100:.1f}%) ===")
    for k in ("correct", "incorrect"):
        g = out["l26_direction_feature"][k]
        if g["n"]:
            print(f"  [{k}] n={g['n']}: target mean={g['mean_target']:+.3f} median={g['median_target']:+.3f}"
                  f" | diff mean={g['mean_diff']:+.3f} | flip={g['flip_rate']*100:.1f}%"
                  f" | wilcoxon p={g['wilcoxon_p_target_less_control']:.3g}")

    # ── L8 geographic-feature ablation (joint c_A+c_B, 21 FDR features, recon baseline)─────────────
    per = json.load(open(L8_JOINT))["per_prompt"]
    correct = np.array([r["pred_dir_clean"] == r["ground_truth"] for r in per])
    target = [r["delta_geo_AB_from_recon"] for r in per]
    flip = [r["flip_AB"] for r in per]
    acc = float(correct.mean())
    out["l8_geographic_feature"] = {
        "accuracy": acc,
        "per_direction_accuracy": per_direction_accuracy(per),
        "correct": group_stats_onesample(target, correct, flip),
        "incorrect": group_stats_onesample(target, ~correct, flip),
    }
    print(f"\n=== L8 geographic-feature ablation (joint, overall accuracy {acc*100:.1f}%) ===")
    for k in ("correct", "incorrect"):
        g = out["l8_geographic_feature"][k]
        if g["n"]:
            print(f"  [{k}] n={g['n']}: target mean={g['mean_target']:+.3f} median={g['median_target']:+.3f}"
                  f" | negative={g['negative_rate']*100:.1f}% | flip={g['flip_rate']*100:.1f}%"
                  f" | wilcoxon p(<0)={g['wilcoxon_p_less_than_0']:.3g}")

    # ── Head 24 / Head 25 ablation ──────────────────────
    head = json.load(open(HEAD))
    per = head["per_prompt"]
    correct = np.array([r["pred_dir_clean"] == r["ground_truth"] for r in per])
    acc = float(correct.mean())
    for h in ("24", "25"):
        delta = [r["target_head_results"][h]["delta_logit"] for r in per]
        out[f"head_{h}"] = {
            "accuracy": acc,
            "per_direction_accuracy": per_direction_accuracy(per),
            "correct": head_group_stats(delta, correct),
            "incorrect": head_group_stats(delta, ~correct),
        }
        print(f"\n=== Head {h} ablation (overall accuracy {acc*100:.1f}%) ===")
        for k in ("correct", "incorrect"):
            g = out[f"head_{h}"][k]
            if g["n"]:
                print(f"  [{k}] n={g['n']}: delta mean={g['mean_delta']:+.3f} median={g['median_delta']:+.3f}"
                      f" | negative rate={g['negative_rate']*100:.1f}%"
                      f" | wilcoxon p(<0)={g['wilcoxon_p_less_than_0']:.3g}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
