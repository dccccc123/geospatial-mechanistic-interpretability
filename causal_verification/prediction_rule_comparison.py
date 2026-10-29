"""
Quantify the flat-argmax vs axis-based direction prediction rules.

The raw direction logits contain an axis-level offset (N/S vs E/W have different
baselines). This script measures the axis offset, the overall and per-direction
accuracy of both rules, their confusion matrices, and the cross-axis error
asymmetry of the flat argmax.

Pure CPU; reads cached results, does not load the model.
"""
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "causal_verification" / "results"
OUT = ROOT / "causal_verification" / "results" / "prediction_rule_comparison.json"

L26 = RES_DIR / "l26_direction_feature_ablation.json"

DIRS = ["East", "West", "South", "North"]
AXIS = {"East": "EW", "West": "EW", "South": "NS", "North": "NS"}


def flat_predict(lg):
    return max(DIRS, key=lambda d: lg[d])


def axis_predict(lg):
    s_ew = lg["East"] - lg["West"]
    s_ns = lg["South"] - lg["North"]
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


def accuracy(gt, pred):
    return float(np.mean([g == p for g, p in zip(gt, pred)]))


def per_dir_accuracy(gt, pred):
    acc = {d: [0, 0] for d in DIRS}
    for g, p in zip(gt, pred):
        acc[g][1] += 1
        acc[g][0] += int(g == p)
    return {d: acc[d][0] / acc[d][1] for d in DIRS}


def confusion_matrix(gt, pred):
    m = {g: {p: 0 for p in DIRS} for g in DIRS}
    for g, p in zip(gt, pred):
        m[g][p] += 1
    return m


def print_confusion(name, cm):
    print(f"\n  [{name}] confusion matrix (rows=GT, cols=prediction):")
    print(f"    {'':>8}" + "".join(f"{d:>8}" for d in DIRS))
    for g in DIRS:
        print(f"    {g:>8}" + "".join(f"{cm[g][p]:>8}" for p in DIRS))


def main():
    per = json.load(open(L26))["per_prompt"]
    gt = [r["ground_truth"] for r in per]
    lg = [r["clean_direction_logits"] for r in per]

    # 1) axis offset
    e = np.array([x["East"] for x in lg])
    w = np.array([x["West"] for x in lg])
    s = np.array([x["South"] for x in lg])
    n = np.array([x["North"] for x in lg])
    delta = float(((s + n) / 2 - (e + w) / 2).mean())
    per_dir_mean = {d: float(np.mean([x[d] for x in lg])) for d in DIRS}

    print("=== 1) axis-offset diagnostic ===")
    print(f"  mean logit per direction: " + str({d: round(per_dir_mean[d], 2) for d in DIRS}))
    print(f"  axis offset δ = mean((S+N)/2 - (E+W)/2) = {delta:+.3f}")
    print(f"  -> {'NS axis higher overall' if delta > 0 else 'EW axis higher overall' if delta < 0 else 'no axis offset'}")

    # 2) predictions of both rules
    flat = [flat_predict(x) for x in lg]
    axis = [axis_predict(x) for x in lg]

    print("\n=== 2) accuracy comparison ===")
    print(f"  flat argmax: overall {accuracy(gt, flat)*100:.1f}%  "
          f"per direction " + str({d: round(per_dir_accuracy(gt, flat)[d]*100, 1) for d in DIRS}))
    print(f"  axis rule  : overall {accuracy(gt, axis)*100:.1f}%  "
          f"per direction " + str({d: round(per_dir_accuracy(gt, axis)[d]*100, 1) for d in DIRS}))

    # 3) confusion matrices
    print_confusion("flat argmax", confusion_matrix(gt, flat))
    print_confusion("axis rule", confusion_matrix(gt, axis))

    # 4) axis agreement + cross-axis error asymmetry
    flat_axis_ok = sum(AXIS[g] == AXIS[p] for g, p in zip(gt, flat))
    axis_axis_ok = sum(AXIS[g] == AXIS[p] for g, p in zip(gt, axis))
    print("\n=== 3) axis agreement (prediction axis == GT axis) ===")
    print(f"  flat argmax: {flat_axis_ok}/{len(gt)} = {flat_axis_ok/len(gt)*100:.1f}%")
    print(f"  axis rule  : {axis_axis_ok}/{len(gt)} = {axis_axis_ok/len(gt)*100:.1f}%")

    wrong = [(g, p) for g, p in zip(gt, flat) if g != p]
    cross_axis = [(g, p) for g, p in wrong if AXIS[g] != AXIS[p]]
    ew_to_ns = sum(AXIS[g] == "EW" and AXIS[p] == "NS" for g, p in cross_axis)
    ns_to_ew = sum(AXIS[g] == "NS" and AXIS[p] == "EW" for g, p in cross_axis)
    print(f"\n  flat argmax mispredicts {len(wrong)} , of which cross-axis: {len(cross_axis)} "
          f"({len(cross_axis)/len(wrong)*100:.1f}%)")
    print(f"    cross-axis errors: EW-GT -> NS-pred {ew_to_ns}; NS-GT -> EW-pred {ns_to_ew}")
    if delta > 0:
        print(f"    δ={delta:+.2f}>0(NS higher): flat argmax is pulled toward NS, expect EW→NS ({ew_to_ns}) "
              f"more than NS→EW ({ns_to_ew})")
    elif delta < 0:
        print(f"    δ={delta:+.2f}<0(EW higher): flat argmax is pulled toward EW, expect NS→EW ({ns_to_ew}) "
              f"more than EW→NS ({ew_to_ns})")

    out = {
        "axis_offset_delta": delta,
        "per_direction_mean_logit": per_dir_mean,
        "flat_argmax": {
            "accuracy": accuracy(gt, flat),
            "per_direction_accuracy": per_dir_accuracy(gt, flat),
            "confusion_matrix": confusion_matrix(gt, flat),
            "axis_agreement": flat_axis_ok / len(gt),
        },
        "axis_rule": {
            "accuracy": accuracy(gt, axis),
            "per_direction_accuracy": per_dir_accuracy(gt, axis),
            "confusion_matrix": confusion_matrix(gt, axis),
            "axis_agreement": axis_axis_ok / len(gt),
        },
        "flat_cross_axis_errors": {"ew_to_ns": ew_to_ns, "ns_to_ew": ns_to_ew, "total": len(cross_axis)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
