"""
Ablation flip rate, winning-margin change, and cDLA-vs-ablation comparison.

Recomputes predictions from the raw four-direction logits with the axis rule and
reports (i) the flip rate and the winning-margin change
margin = logit(pred) - max(logit(other directions)) after each ablation, and
(ii) whether high-cDLA L26 direction features also show larger causal ablation
effects (delta logit).

Pure CPU; reads cached results, does not load the model or re-run ablations.
"""
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "causal_verification" / "results"
OUT = ROOT / "causal_verification" / "results" / "ablation_margin_and_dla.json"

L26 = RES_DIR / "l26_direction_feature_ablation.json"
L8_JOINT = RES_DIR / "l8_geographic_subspace_ablation_ca_joint.json"
HEAD = RES_DIR / "head_ablation_delta_logit.json"

DIRS = ["East", "West", "South", "North"]


def axis_predict(lg):
    s_ew = lg["East"] - lg["West"]
    s_ns = lg["South"] - lg["North"]
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


def winner_margin(lg, pred):
    return lg[pred] - max(lg[d] for d in DIRS if d != pred)


def margin_stats(clean_lg_list, after_lg_list):
    """Recompute predictions from raw logits with the axis rule; return flip rate and margin change."""
    flip, d_margin = [], []
    for clg, alg in zip(clean_lg_list, after_lg_list):
        pred = axis_predict(clg)
        flip.append(axis_predict(alg) != pred)
        d_margin.append(winner_margin(alg, pred) - winner_margin(clg, pred))
    flip = np.array(flip)
    d_margin = np.array(d_margin)
    return {
        "n": int(len(flip)),
        "flip_rate": float(flip.mean()),
        "mean_delta_margin": float(d_margin.mean()),
        "median_delta_margin": float(np.median(d_margin)),
    }


def main():
    out = {}

    # ── R3-2:flip + margin ─────────────────────────
    per = json.load(open(L26))["per_prompt"]
    out["l26_direction_feature"] = margin_stats(
        [r["clean_direction_logits"] for r in per],
        [r["target_direction_logits"] for r in per],
    )
    per = json.load(open(L8_JOINT))["per_prompt"]
    out["l8_geographic_feature"] = margin_stats(
        [r["clean_direction_logits"] for r in per],
        [r["geo_AB_direction_logits"] for r in per],
    )
    head = json.load(open(HEAD))
    per = head["per_prompt"]
    for h in ("24", "25"):
        out[f"head_{h}"] = margin_stats(
            [r["clean_direction_logits"] for r in per],
            [r["target_head_results"][h]["direction_logits_after"] for r in per],
        )

    print("=== flip rate + winning-margin change (axis rule) ===")
    for k in ("l26_direction_feature", "l8_geographic_feature", "head_24", "head_25"):
        g = out[k]
        print(f"  {k:24s}: flip={g['flip_rate']*100:.1f}%  "
              f"Δmargin mean={g['mean_delta_margin']:+.3f} median={g['median_delta_margin']:+.3f}")

    # ── cDLA vs actual ablation delta logit (L26)─────
    l26 = json.load(open(L26))
    per = l26["per_prompt"]
    cm = l26["control_matching"]["directions"]
    pred = [axis_predict(r["clean_direction_logits"]) for r in per]

    rows = []
    for direction in DIRS:
        info = cm[direction]
        mask = np.array([p == direction for p in pred])
        t_fid = info["target_feature"]
        t_cdla = info["target_metrics"]["cdla_by_direction"][direction]
        t_delta = float(np.mean([r["delta_target_from_recon"] for r, m in zip(per, mask) if m]))
        rows.append((direction, t_fid, "target", t_cdla, t_delta))
        for c in info["selected_controls"]:
            c_fid = c["feature_id"]
            c_cdla = c["metrics"]["cdla_by_direction"][direction]
            c_delta = float(np.mean([r["delta_control_from_recon_mean"] for r, m in zip(per, mask) if m]))
            rows.append((direction, c_fid, "control", c_cdla, c_delta))

    print("\n=== cDLA (heuristic) vs actual ablation delta logit (causal, L26) ===")
    print(f"  {'direction':>9} {'feature':>8} {'role':>8} {'cDLA':>8} {'mean Δlogit':>11}")
    for direction, fid, role, cdla, delta in rows:
        print(f"  {direction:>9} {fid:>8} {role:>8} {cdla:>8.3f} {delta:>11.3f}")

    out["r35_cdla_vs_ablation"] = [
        {"direction": d, "feature_id": f, "role": r, "cDLA": float(c), "mean_ablation_delta": float(v)}
        for d, f, r, c, v in rows
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
