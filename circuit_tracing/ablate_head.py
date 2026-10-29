"""
Ablate L26 attention heads to test whether Head 24 and Head 25 are necessary paths.

Zeroes each target head's slice of the o_proj input (Head 24 carries city_b, Head 25
carries city_a) and compares the direction-score drop and flip rate against five
randomly chosen control heads, then plots the results.
"""

import os, json, warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────
# paths
# ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data_process/dataset/balanced_city_pairs.jsonl"
PRED_PATH  = ROOT / "linear_probe/activations/baseline_preds.npy"
RESULT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR    = Path(__file__).resolve().parent / "figures"

for d in [RESULT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3-8B"
TARGET_LAYER  = 26
N_HEADS       = 32
D_MODEL       = 4096
HEAD_DIM      = D_MODEL // N_HEADS   # 128
PROMPT_TPL    = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID  = 279
N_PER_DIR     = 4    # 4 prompts per direction, 16 total

DIR_LABELS    = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# Target heads: Head 24 (city_b) and Head 25 (city_a)
TARGET_HEADS = {
    24: "city_b (expected)",
    25: "city_a (expected)",
}

# Control heads: 5 random non-target heads
rng = np.random.default_rng(42)
CONTROL_HEADS = sorted(
    rng.choice(
        [h for h in range(N_HEADS) if h not in TARGET_HEADS],
        size=5, replace=False
    ).tolist()
)


# ─────────────────────────────────────────────────────────────
# Step 1: load data
# ─────────────────────────────────────────────────────────────
def load_prompts():
    data        = [json.loads(l) for l in DATA_PATH.open()]
    model_preds = np.load(str(PRED_PATH)).tolist()

    correct_by_dir = {d: [] for d in DIR_LABELS}
    for item, pred_idx in zip(data, model_preds):
        gt_dir   = item["Direction"]
        pred_dir = DIR_LABELS[pred_idx]
        if pred_dir == gt_dir:
            correct_by_dir[gt_dir].append(item)

    print(f"[Filter] correct samples per direction:")
    for d in DIR_LABELS:
        print(f"  {d}: {len(correct_by_dir[d])} / 45")

    selected = []
    for d in DIR_LABELS:
        pool = correct_by_dir[d]
        n    = min(N_PER_DIR, len(pool))
        for item in pool[:n]:
            selected.append({
                "city_a":    item["City_A"],
                "city_b":    item["City_B"],
                "direction": item["Direction"],
                "prompt":    PROMPT_TPL.format(
                    city_a=item["City_A"], city_b=item["City_B"]
                ),
            })

    print(f"\n[Selected] {len(selected)} prompts ({N_PER_DIR} per direction)")
    print(f"[Target heads] {list(TARGET_HEADS.keys())}")
    print(f"[Control heads] {CONTROL_HEADS}")
    return selected


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def compute_score(logits, direction):
    if direction in ("East", "West"):
        return (logits[DIR_TOKEN_IDS["East"]].item()
                - logits[DIR_TOKEN_IDS["West"]].item())
    else:
        return (logits[DIR_TOKEN_IDS["South"]].item()
                - logits[DIR_TOKEN_IDS["North"]].item())


def is_correct(score, direction):
    return score > 0 if direction in ("East", "South") else score < 0


def make_ablation_hook(head_idx):
    """
    Return a pre_hook that zeroes head_idx's slice at the o_proj input.
    Qwen3-8B: d_model=4096, n_heads=32, head_dim=128
    Head i occupies slice [i*128 : (i+1)*128]
    """
    start = head_idx * HEAD_DIM
    end   = start + HEAD_DIM

    def pre_hook(module, input):
        inp       = list(input)
        inp[0]    = inp[0].clone()
        inp[0][:, :, start:end] = 0.0
        return tuple(inp)

    return pre_hook


# ─────────────────────────────────────────────────────────────
# Step 2: baseline forward
# ─────────────────────────────────────────────────────────────
def baseline_forward(model, tokenizer, prompts_data):
    baselines = []
    for item in prompts_data:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        score   = compute_score(logits, item["direction"])
        correct = is_correct(score, item["direction"])
        baselines.append({"score": float(score), "correct": correct})

    acc = np.mean([b["correct"] for b in baselines])
    print(f"[Baseline] accuracy: {acc*100:.1f}%  "
          f"mean_score: {np.mean([b['score'] for b in baselines]):+.3f}")
    return baselines


# ─────────────────────────────────────────────────────────────
# Step 3: ablate a single head
# ─────────────────────────────────────────────────────────────
def ablate_head(model, tokenizer, prompts_data, baselines, head_idx):
    delta_scores, flip_flags = [], []
    pre_hook = make_ablation_hook(head_idx)

    for item, base in zip(prompts_data, baselines):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        handle = model.model.layers[TARGET_LAYER].self_attn.o_proj\
                     .register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        handle.remove()

        score_after = compute_score(logits, item["direction"])
        delta       = score_after - base["score"]
        flipped     = (is_correct(score_after, item["direction"]) !=
                       is_correct(base["score"],  item["direction"]))
        delta_scores.append(float(delta))
        flip_flags.append(bool(flipped))

    return delta_scores, flip_flags


# ─────────────────────────────────────────────────────────────
# Step 4: statistics
# ─────────────────────────────────────────────────────────────
def analyze_one_head(head_idx, label, delta_scores, flip_flags,
                     all_ctrl_deltas, all_ctrl_flips):
    mean_d    = np.mean(delta_scores)
    std_d     = np.std(delta_scores)
    flip_rate = np.mean(flip_flags)
    ctrl_flip = np.mean(all_ctrl_flips)

    print(f"\n  [Head {head_idx} — {label}]")
    print(f"  Δscore mean: {mean_d:+.4f}  std: {std_d:.4f}  flip rate: {flip_rate*100:.1f}%")
    print(f"  control flip rate: {ctrl_flip*100:.1f}%")

    # Wilcoxon test
    try:
        n    = len(delta_scores)
        stat_w, pval_w = wilcoxon(
            [abs(d) for d in delta_scores],
            [abs(d) for d in all_ctrl_deltas[:n]],
            alternative="greater"
        )
    except Exception:
        stat_w, pval_w = float("nan"), float("nan")

    # Mann-Whitney U
    stat_u, pval_u = mannwhitneyu(
        [abs(d) for d in delta_scores],
        [abs(d) for d in all_ctrl_deltas],
        alternative="greater"
    )

    print(f"  Wilcoxon p={pval_w:.4e}  Mann-Whitney p={pval_u:.4e}")

    if pval_u < 0.05 and flip_rate > ctrl_flip:
        verdict = f"✅ necessary path (p={pval_u:.2e}, flip rate {flip_rate*100:.1f}% vs control {ctrl_flip*100:.1f}%)"
    elif pval_u < 0.05:
        verdict = f"⚠️  significant (p={pval_u:.2e}) but flip rate not above control"
    else:
        verdict = f"❌ no significant difference from control heads (p={pval_u:.4f})"
    print(f"  Conclusion: {verdict}")

    return {
        "head_idx":        head_idx,
        "label":           label,
        "mean_delta":      round(float(mean_d),    4),
        "std_delta":       round(float(std_d),     4),
        "flip_rate":       round(float(flip_rate), 4),
        "ctrl_flip_rate":  round(float(ctrl_flip), 4),
        "wilcoxon_p":      float(pval_w) if not np.isnan(pval_w) else None,
        "mannwhitney_p":   float(pval_u),
        "verdict":         verdict,
        "delta_scores":    delta_scores,
        "flip_flags":      flip_flags,
    }


# ─────────────────────────────────────────────────────────────
# Step 5: visualization
# ─────────────────────────────────────────────────────────────
def plot_results(results_target, results_ctrl, prompts_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#F8F8F8")

    # ── left: Δscore boxplot ─────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F0F0F0")

    all_ctrl_deltas = [d for r in results_ctrl for d in r["delta_scores"]]
    colors_box = ["#FCA5A5", "#BFDBFE", "#D1D5DB"]   # H24 red, H25 blue, ctrl gray
    labels_box = [
        f"Head 24\n({results_target[24]['label']})",
        f"Head 25\n({results_target[25]['label']})",
        "Control\nHeads",
    ]
    data_box = [
        results_target[24]["delta_scores"],
        results_target[25]["delta_scores"],
        all_ctrl_deltas,
    ]

    bp = ax.boxplot(
        data_box,
        tick_labels=labels_box,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    face_colors = ["#FCA5A5", "#BFDBFE", "#D1D5DB"]
    dot_colors  = ["#DC2626", "#2563EB", "#6B7280"]
    for patch, fc in zip(bp["boxes"], face_colors):
        patch.set_facecolor(fc)

    rng2 = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(data_box, dot_colors)):
        jitter = rng2.uniform(-0.12, 0.12, len(vals))
        ax.scatter([i + 1 + j for j in jitter], vals,
                   color=col, s=30, zorder=3, alpha=0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Δscore (after - before ablation)", fontsize=10)
    ax.set_title("Δscore Distribution After Head Ablation\n"
                 "(more negative / larger |Δ| = head is more important)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)

    # ── right: flip rate + significance ──────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#F0F0F0")

    heads_to_show = [24, 25] + CONTROL_HEADS
    flip_rates    = (
        [results_target[24]["flip_rate"] * 100,
         results_target[25]["flip_rate"] * 100] +
        [np.mean(r["flip_flags"]) * 100 for r in results_ctrl]
    )
    bar_colors = (
        ["#DC2626", "#2563EB"] +
        ["#CBD5E1"] * len(CONTROL_HEADS)
    )
    bar_labels = (
        [f"H24\n(city_b)", f"H25\n(city_a)"] +
        [f"H{h}" for h in CONTROL_HEADS]
    )

    bars = ax2.bar(range(len(heads_to_show)), flip_rates,
                   color=bar_colors, alpha=0.88, zorder=2, width=0.6)

    # significance markers
    for i, (head_idx, flip) in enumerate(zip([24, 25], flip_rates[:2])):
        p = results_target[head_idx]["mannwhitney_p"]
        mark = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax2.text(i, flip + 0.5, mark, ha="center",
                 fontsize=11, fontweight="bold",
                 color="#DC2626" if head_idx == 24 else "#2563EB")

    ax2.set_xticks(range(len(heads_to_show)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel("Prediction Flip Rate (%)", fontsize=10)
    ax2.set_title("Flip Rate by Head\n(* p<0.05, ** p<0.01, *** p<0.001)",
                  fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(flip_rates) * 1.5 + 5 if max(flip_rates) > 0 else 20)
    ax2.grid(axis="y", alpha=0.4)

    # legend
    import matplotlib.patches as mpatches
    ax2.legend(handles=[
        mpatches.Patch(color="#DC2626", label="Head 24 (city_b path)"),
        mpatches.Patch(color="#2563EB", label="Head 25 (city_a path)"),
        mpatches.Patch(color="#CBD5E1", label="Control Heads"),
    ], fontsize=8, loc="upper right")

    save_path = FIG_DIR / "ablate_head_results.png"
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Attention head ablation experiment")
    print("  Head 24 (city_b) vs Head 25 (city_a) vs control heads")
    print("=" * 60)

    print(f"\n=== Step 1: load data ===")
    prompts_data = load_prompts()

    print(f"\n=== Step 2: load model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager"
    )
    model.eval()
    print(f"[Load] done, HEAD_DIM={HEAD_DIM}")

    print(f"\n=== Step 3: baseline forward ===")
    baselines = baseline_forward(model, tokenizer, prompts_data)

    print(f"\n=== Step 4: ablation experiment ===")

    # ablate target heads (Head 24 and Head 25)
    raw_target = {}
    for head_idx, label in TARGET_HEADS.items():
        print(f"\n  [Ablate Head {head_idx} — {label}]")
        d, f = ablate_head(model, tokenizer, prompts_data, baselines, head_idx)
        raw_target[head_idx] = (d, f)
        print(f"  Δscore={np.mean(d):+.4f}  flip rate={np.mean(f)*100:.1f}%")

    # ablate control heads
    raw_ctrl = []
    for head_idx in CONTROL_HEADS:
        print(f"\n  [Ablate Head {head_idx} — control]")
        d, f = ablate_head(model, tokenizer, prompts_data, baselines, head_idx)
        raw_ctrl.append((head_idx, d, f))
        print(f"  Δscore={np.mean(d):+.4f}  flip rate={np.mean(f)*100:.1f}%")

    del model
    torch.cuda.empty_cache()
    print(f"\n[Model] memory released.")

    print(f"\n=== Step 5: statistical analysis ===")
    all_ctrl_deltas = [d for _, deltas, _ in raw_ctrl for d in deltas]
    all_ctrl_flips  = [f for _, _, flips in raw_ctrl for f in flips]

    results_target = {}
    for head_idx, label in TARGET_HEADS.items():
        d, f = raw_target[head_idx]
        results_target[head_idx] = analyze_one_head(
            head_idx, label, d, f, all_ctrl_deltas, all_ctrl_flips
        )

    print(f"\n  [Control heads summary]")
    results_ctrl = []
    for head_idx, d, f in raw_ctrl:
        mean_d    = np.mean(d)
        flip_rate = np.mean(f)
        print(f"  Head {head_idx:2d}: Δscore={mean_d:+.4f}  flip rate={flip_rate*100:.1f}%")
        results_ctrl.append({
            "head_idx":    head_idx,
            "mean_delta":  round(float(mean_d),    4),
            "flip_rate":   round(float(flip_rate), 4),
            "delta_scores": d,
            "flip_flags":   f,
        })

    print(f"\n  [Overall conclusion]")
    for head_idx, label in TARGET_HEADS.items():
        print(f"  Head {head_idx} ({label}): {results_target[head_idx]['verdict']}")

    print(f"\n=== Step 6: visualization ===")
    plot_results(results_target, results_ctrl, prompts_data)

    print(f"\n=== Step 7: save results ===")
    output = {
        "model":         MODEL_NAME,
        "target_layer":  TARGET_LAYER,
        "target_heads":  list(TARGET_HEADS.keys()),
        "control_heads": CONTROL_HEADS,
        "n_prompts":     len(prompts_data),
        "target_results": {
            str(hid): results_target[hid]
            for hid in TARGET_HEADS
        },
        "control_results": results_ctrl,
        "summary": {
            str(hid): results_target[hid]["verdict"]
            for hid in TARGET_HEADS
        },
    }
    path = RESULT_DIR / "ablate_head.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[Saved] {path}")

    print(f"\n{'='*60}")
    print(f"  Experiment done")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()