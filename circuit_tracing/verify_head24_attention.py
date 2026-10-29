"""
Verify the L26 Head 24 attention pattern over the city_a and city_b tokens.

For correctly predicted prompts, extracts Head 24's attention from the "the"
position, checks whether it attends to city_b, city_a, or both, and plots the
per-prompt pattern, the head ranking, and cross-direction stability.
"""

import os, json, warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
MODEL_NAME   = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
TARGET_HEAD  = 24
N_HEADS      = 32
THE_TOKEN_ID = 279
PROMPT_TPL   = "Relative to {city_a}, {city_b} is to the"
N_PER_DIR    = 2

DIR_LABELS    = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}


# ─────────────────────────────────────────────────────────────
# Step 1: select test prompts
# ─────────────────────────────────────────────────────────────
def select_test_prompts():
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
        if len(pool) < N_PER_DIR:
            print(f"  [Warning] fewer than {N_PER_DIR} correct {d} samples, taking all")
        for item in pool[:N_PER_DIR]:
            selected.append({
                "city_a":    item["City_A"],
                "city_b":    item["City_B"],
                "direction": item["Direction"],
            })

    print(f"\n[Selected] {len(selected)} test prompts:")
    for s in selected:
        print(f"  [{s['direction']:5s}] {s['city_a']} → {s['city_b']}")
    return selected


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def get_the_pos(input_ids):
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    return positions[-1].item()


def get_city_last_pos(input_ids, tokenizer, city_name):
    """Locate the last token of a city name (used for both city_a and city_b)."""
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
    city_lower = city_name.lower()
    last_pos   = -1
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("▁Ġ").lower()
        if clean and city_lower.endswith(clean):
            last_pos = i
    return last_pos, tokens


def model_predict(logits, direction):
    if direction in ("East", "West"):
        score = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
        pred  = "East" if score > 0 else "West"
    else:
        score = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
        pred  = "South" if score > 0 else "North"
    return pred == direction, pred


def head_rank(all_heads_attn, target_head):
    """Rank of target_head among all heads (1 = highest)."""
    return int(np.sum(all_heads_attn > all_heads_attn[target_head])) + 1


# ─────────────────────────────────────────────────────────────
# Step 2-5: extract attention weights and analyze
# ─────────────────────────────────────────────────────────────
def analyze_attention(model, tokenizer, test_prompts):
    all_records = []

    for idx, item in enumerate(test_prompts):
        city_a    = item["city_a"]
        city_b    = item["city_b"]
        direction = item["direction"]
        prompt    = PROMPT_TPL.format(city_a=city_a, city_b=city_b)

        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)

        # locate both city_a and city_b
        ca_pos, tokens = get_city_last_pos(inputs.input_ids, tokenizer, city_a)
        cb_pos, _      = get_city_last_pos(inputs.input_ids, tokenizer, city_b)

        display = [t.lstrip("Ġ▁") or t for t in tokens]

        print(f"\n  [{direction:5s}] {city_a} → {city_b}")
        print(f"    tokens : {display}")
        print(f"    city_a : pos={ca_pos} "
              f"('{tokens[ca_pos] if ca_pos >= 0 else 'N/A'}')")
        print(f"    city_b : pos={cb_pos} "
              f"('{tokens[cb_pos] if cb_pos >= 0 else 'N/A'}')")
        print(f"    the    : pos={the_pos}")

        if ca_pos < 0 or cb_pos < 0:
            print(f"    [Skip] city_a or city_b not located")
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        correct, pred = model_predict(outputs.logits[0, -1], direction)
        print(f"    model prediction: {pred} ({'✅ correct' if correct else '❌ wrong'})")

        # L26 Head 24 attention matrix
        attn_l26        = outputs.attentions[TARGET_LAYER][0].cpu().float()  # [32, seq, seq]
        head24_from_the = attn_l26[TARGET_HEAD, the_pos, :].numpy()          # [seq_len]

        # ── city_b analysis ──────────────────────────────────
        attn_to_cb      = float(head24_from_the[cb_pos])
        all_heads_to_cb = attn_l26[:, the_pos, cb_pos].numpy()   # [32]
        rank_cb         = head_rank(all_heads_to_cb, TARGET_HEAD)

        # ── city_a analysis ──────────────────────────────────
        attn_to_ca      = float(head24_from_the[ca_pos])
        all_heads_to_ca = attn_l26[:, the_pos, ca_pos].numpy()   # [32]
        rank_ca         = head_rank(all_heads_to_ca, TARGET_HEAD)

        # ── top attended token (excluding "the" itself) ──────
        attn_excl_self           = head24_from_the.copy()
        attn_excl_self[the_pos]  = 0.0
        top_pos                  = int(np.argmax(attn_excl_self))
        top_is_city_b            = (top_pos == cb_pos)
        top_is_city_a            = (top_pos == ca_pos)

        print(f"    Head24 → city_b : attn={attn_to_cb:.4f}  rank=#{rank_cb}/32")
        print(f"    Head24 → city_a : attn={attn_to_ca:.4f}  rank=#{rank_ca}/32")
        print(f"    top-attention token : pos={top_pos} '{tokens[top_pos]}'  "
              f"{'is city_b ✅' if top_is_city_b else 'is city_a ✅' if top_is_city_a else 'other ❌'}")

        all_records.append({
            "idx":              idx,
            "city_a":           city_a,
            "city_b":           city_b,
            "direction":        direction,
            "prompt":           prompt,
            "tokens":           tokens,
            "ca_pos":           ca_pos,
            "cb_pos":           cb_pos,
            "the_pos":          the_pos,
            "model_correct":    correct,
            "model_pred":       pred,
            # city_b
            "attn_to_cb":       round(attn_to_cb, 6),
            "rank_cb":          rank_cb,
            "all_heads_to_cb":  all_heads_to_cb.tolist(),
            # city_a
            "attn_to_ca":       round(attn_to_ca, 6),
            "rank_ca":          rank_ca,
            "all_heads_to_ca":  all_heads_to_ca.tolist(),
            # full-sequence attention
            "head24_from_the":  head24_from_the.tolist(),
            "top_is_city_b":    top_is_city_b,
            "top_is_city_a":    top_is_city_a,
        })

    return all_records


# ─────────────────────────────────────────────────────────────
# Step 6: visualization
# ─────────────────────────────────────────────────────────────
def plot_single_prompt(record, save_path):
    """
    Bar chart of Head 24 attention for one prompt; city_a in blue, city_b in red.
    """
    tokens  = record["tokens"]
    attn    = np.array(record["head24_from_the"])
    ca_pos  = record["ca_pos"]
    cb_pos  = record["cb_pos"]
    the_pos = record["the_pos"]
    display = [t.lstrip("Ġ▁") or t for t in tokens]

    colors = []
    for i in range(len(tokens)):
        if i == cb_pos:
            colors.append("#DC2626")    # city_b red
        elif i == ca_pos:
            colors.append("#2563EB")    # city_a blue
        elif i == the_pos:
            colors.append("#94A3B8")    # the gray
        else:
            colors.append("#CBD5E1")    # other light gray

    fig, ax = plt.subplots(figsize=(max(9, len(tokens) * 0.85), 4))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F0F0F0")

    ax.bar(range(len(tokens)), attn, color=colors, zorder=2, width=0.7)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(display, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Attention Weight", fontsize=10)
    ax.set_ylim(0, min(1.0, attn.max() * 1.45))
    ax.set_title(
        f"L26 Head 24: Attention from 'the' Position\n"
        f"[{record['direction']}] {record['city_a']} → {record['city_b']}  |  "
        f"city_a={record['attn_to_ca']:.3f}(#{record['rank_ca']}/32)  "
        f"city_b={record['attn_to_cb']:.3f}(#{record['rank_cb']}/32)",
        fontsize=10, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.4)

    legend_elements = [
        mpatches.Patch(color="#DC2626",
                       label=f"city_b: {record['city_b']} "
                             f"(attn={record['attn_to_cb']:.3f}, "
                             f"rank=#{record['rank_cb']}/32)"),
        mpatches.Patch(color="#2563EB",
                       label=f"city_a: {record['city_a']} "
                             f"(attn={record['attn_to_ca']:.3f}, "
                             f"rank=#{record['rank_ca']}/32)"),
        mpatches.Patch(color="#94A3B8", label="'the' (self)"),
        mpatches.Patch(color="#CBD5E1", label="Other tokens"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_head_ranking(records, save_path):
    """
    All 32 heads' attention to city_a (top) and city_b (bottom), per direction.
    Highlights Head 25 for city_a and Head 24 for city_b.
    """
    present = [d for d in DIR_LABELS
               if any(r["direction"] == d for r in records)]
    n_dirs  = len(present)

    fig, axes = plt.subplots(2, n_dirs,
                             figsize=(5 * n_dirs, 7),
                             sharex=False)
    fig.patch.set_facecolor("#F8F8F8")
    if n_dirs == 1:
        axes = axes.reshape(2, 1)

    TARGET_HEAD_CA = 25   # city_a routing head
    TARGET_HEAD_CB = 24   # city_b routing head

    for col, direction in enumerate(present):
        recs = [r for r in records if r["direction"] == direction]

        # ── top: city_a, highlight Head 25 ────────────────────
        ax       = axes[0, col]
        avg_attn = np.mean([r["all_heads_to_ca"] for r in recs], axis=0)
        colors   = ["#2563EB" if h == TARGET_HEAD_CA else "#CBD5E1"
                    for h in range(N_HEADS)]

        ax.set_facecolor("#F0F0F0")
        ax.bar(range(N_HEADS), avg_attn, color=colors, zorder=2, width=0.8)
        ax.axvline(TARGET_HEAD_CA, color="#1e3a5f",
                   linestyle="--", linewidth=1.2, alpha=0.7)

        h25_val  = avg_attn[TARGET_HEAD_CA]
        h25_rank = int(np.sum(avg_attn > h25_val)) + 1
        ax.set_title(
            f"{direction} | city_a\n"
            f"Head25: {h25_val:.3f}  rank=#{h25_rank}/32",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Head Index", fontsize=8)
        ax.set_ylabel("Avg Attn to city_a", fontsize=8)
        ax.set_xlim(-0.5, N_HEADS - 0.5)
        ax.grid(axis="y", alpha=0.4)
        ax.text(TARGET_HEAD_CA, h25_val + 0.003, "H25",
                ha="center", fontsize=7,
                color="#1e3a5f", fontweight="bold")

        # ── bottom: city_b, highlight Head 24 ─────────────────
        ax       = axes[1, col]
        avg_attn = np.mean([r["all_heads_to_cb"] for r in recs], axis=0)
        colors   = ["#DC2626" if h == TARGET_HEAD_CB else "#CBD5E1"
                    for h in range(N_HEADS)]

        ax.set_facecolor("#F0F0F0")
        ax.bar(range(N_HEADS), avg_attn, color=colors, zorder=2, width=0.8)
        ax.axvline(TARGET_HEAD_CB, color="#991B1B",
                   linestyle="--", linewidth=1.2, alpha=0.7)

        h24_val  = avg_attn[TARGET_HEAD_CB]
        h24_rank = int(np.sum(avg_attn > h24_val)) + 1
        ax.set_title(
            f"{direction} | city_b\n"
            f"Head24: {h24_val:.3f}  rank=#{h24_rank}/32",
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("Head Index", fontsize=8)
        ax.set_ylabel("Avg Attn to city_b", fontsize=8)
        ax.set_xlim(-0.5, N_HEADS - 0.5)
        ax.grid(axis="y", alpha=0.4)
        ax.text(TARGET_HEAD_CB, h24_val + 0.003, "H24",
                ha="center", fontsize=7,
                color="#991B1B", fontweight="bold")

    fig.suptitle(
        "All 32 Heads: Attention to city_a (top) and city_b (bottom)\n"
        "from 'the' Position at L26  |  Blue bar = Head 25 (city_a)  |  Red bar = Head 24 (city_b)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stability(records, save_path):
    """
    Cross-direction stability heatmap; one row per prompt, red box = city_b, blue = city_a.
    """
    n = len(records)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n))
    fig.patch.set_facecolor("#F8F8F8")
    if n == 1:
        axes = [axes]

    for ax, record in zip(axes, records):
        attn    = np.array(record["head24_from_the"])
        tokens  = record["tokens"]
        ca_pos  = record["ca_pos"]
        cb_pos  = record["cb_pos"]
        the_pos = record["the_pos"]
        display = [t.lstrip("Ġ▁") or t for t in tokens]

        im = ax.imshow(
            attn[np.newaxis, :], cmap="YlOrRd",
            aspect="auto", vmin=0, vmax=max(attn.max(), 0.01)
        )

        for j, val in enumerate(attn):
            if val > 0.04:
                ax.text(j, 0, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.5 else "white")

        # red box = city_b, blue box = city_a
        if 0 <= cb_pos < len(tokens):
            ax.add_patch(plt.Rectangle(
                (cb_pos - 0.5, -0.5), 1, 1,
                fill=False, edgecolor="#DC2626", linewidth=2.5
            ))
        if 0 <= ca_pos < len(tokens):
            ax.add_patch(plt.Rectangle(
                (ca_pos - 0.5, -0.5), 1, 1,
                fill=False, edgecolor="#2563EB", linewidth=2.5
            ))

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(display, fontsize=8, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_ylabel(
            f"[{record['direction']}]\n"
            f"{record['city_a']}→{record['city_b']}",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=65
        )
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    # legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="#DC2626",
                       linewidth=2, label="city_b"),
        mpatches.Patch(facecolor="none", edgecolor="#2563EB",
                       linewidth=2, label="city_a"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8,
                   loc="upper right", framealpha=0.8)

    fig.suptitle(
        "L26 Head 24: Attention Pattern Across Directions\n"
        "(Red box = city_b, Blue box = city_a)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# summary statistics
# ─────────────────────────────────────────────────────────────
def print_summary(records):
    print(f"\n{'='*70}")
    print(f"  Summary statistics")
    print(f"{'='*70}")
    print(f"  {'dir':>6}  {'Prompt':>24}  "
          f"{'city_a attn':>12} {'rank':>6}  "
          f"{'city_b attn':>12} {'rank':>6}")
    print(f"  {'─'*70}")
    for r in records:
        print(f"  {r['direction']:>6}  "
              f"{r['city_a']:>10}→{r['city_b']:<12}  "
              f"{r['attn_to_ca']:>12.4f} #{r['rank_ca']:>2}/32  "
              f"{r['attn_to_cb']:>12.4f} #{r['rank_cb']:>2}/32")

    print(f"\n  [Overall statistics]")
    mean_attn_ca  = np.mean([r["attn_to_ca"]  for r in records])
    mean_attn_cb  = np.mean([r["attn_to_cb"]  for r in records])
    mean_rank_ca  = np.mean([r["rank_ca"]      for r in records])
    mean_rank_cb  = np.mean([r["rank_cb"]      for r in records])
    top3_ca       = np.mean([r["rank_ca"] <= 3 for r in records])
    top3_cb       = np.mean([r["rank_cb"] <= 3 for r in records])
    top_is_cb     = np.mean([r["top_is_city_b"] for r in records])
    top_is_ca     = np.mean([r["top_is_city_a"] for r in records])

    print(f"  city_a mean attention: {mean_attn_ca:.4f}  "
          f"mean rank: #{mean_rank_ca:.1f}  Top3 rate: {top3_ca*100:.1f}%")
    print(f"  city_b mean attention: {mean_attn_cb:.4f}  "
          f"mean rank: #{mean_rank_cb:.1f}  Top3 rate: {top3_cb*100:.1f}%")
    print(f"  top attention to city_b: {top_is_cb*100:.1f}%")
    print(f"  top attention to city_a: {top_is_ca*100:.1f}%")

    print(f"\n  [Conclusion]")
    if top3_cb >= 0.75 and top3_ca >= 0.5:
        print(f"  ✅ Head 24 attends to both city_b (Top3={top3_cb*100:.0f}%)"
              f"and city_a (Top3={top3_ca*100:.0f}%),"
              f"supporting the dual-input hypothesis.")
    elif top3_cb >= 0.75:
        print(f"  ⚠️  Head 24 attends mostly to city_b (Top3={top3_cb*100:.0f}%),"
              f"city_a attention is weaker (Top3={top3_ca*100:.0f}%),"
              f"the dual-input hypothesis needs further validation.")
    else:
        print(f"  ❌ Head 24 attention to city_b is unstable (Top3={top3_cb*100:.0f}%),"
              f"the activation-patching conclusion should be revisited.")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Verify L26 Head 24 attention pattern")
    print("  (tracing city_a and city_b)")
    print("=" * 55)

    print(f"\n=== Step 1: select test prompts from real data ===")
    test_prompts = select_test_prompts()

    print(f"\n=== Step 2: load model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model.eval()
    print(f"[Load] done, device: {next(model.parameters()).device}")

    print(f"\n=== Step 3-5: extract attention and analyze ===")
    records = analyze_attention(model, tokenizer, test_prompts)

    del model
    torch.cuda.empty_cache()
    print(f"\n[Model] memory released.")

    print(f"\n=== Step 6: visualization ===")
    for record in records:
        fname = (f"head24_attn_prompt{record['idx']}_"
                 f"{record['direction']}_{record['city_b']}.png")
        plot_single_prompt(record, FIG_DIR / fname)
        print(f"  [Fig 1] saved: {fname}")

    plot_head_ranking(records, FIG_DIR / "head24_attn_ranking.png")
    print(f"  [Fig 2] saved: head24_attn_ranking.png")

    plot_stability(records, FIG_DIR / "head24_attn_stability.png")
    print(f"  [Fig 3] saved: head24_attn_stability.png")

    print_summary(records)

    result_path = RESULT_DIR / "head24_attention.json"
    result_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\n[Saved] results written to: {result_path}")

    print(f"\n{'='*55}")
    print(f"  Experiment done")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()