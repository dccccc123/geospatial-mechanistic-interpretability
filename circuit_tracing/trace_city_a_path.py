"""
Trace which layers and heads attend to city_a across the model.

For every layer L0-L35, extracts the attention from the "the" position to the
city_a and city_b tokens, finds each layer's top attending head, and locates the
layers/heads that carry city_a information into the "the" position. Saves
per-layer statistics and plots heatmaps and line plots.
"""

import os, json, warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
N_LAYERS     = 36
N_HEADS      = 32
THE_TOKEN_ID = 279
PROMPT_TPL   = "Relative to {city_a}, {city_b} is to the"
N_PER_DIR    = 2    # 2 prompts per direction, 8 total

DIR_LABELS    = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
TARGET_LAYER  = 26
TARGET_HEAD   = 24


# ─────────────────────────────────────────────────────────────
# Step 1: load data and keep correctly predicted prompts
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

    print(f"\n[Selected] {len(selected)} prompts:")
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
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])
    city_lower = city_name.lower()
    last_pos   = -1
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("▁Ġ").lower()
        if clean and city_lower.endswith(clean):
            last_pos = i
    return last_pos, tokens


# ─────────────────────────────────────────────────────────────
# Step 2: extract attention weights across all layers
# ─────────────────────────────────────────────────────────────
def extract_attention_all_layers(model, tokenizer, prompts_data):
    """
    For each prompt, extract, in each of layers L0..L35:
      the attention from "the" to city_a (all 32 heads), and
      the attention from "the" to city_b (all 32 heads, for comparison).

    Returns:
      records: list of dicts, one per prompt with its full attention data.
    """
    print(f"\n[Extract] extracting all-layer attention for {len(prompts_data)} prompts...")
    records = []

    for item in prompts_data:
        city_a = item["city_a"]
        city_b = item["city_b"]
        prompt = item["prompt"]

        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)
        ca_pos, tokens = get_city_last_pos(inputs.input_ids, tokenizer, city_a)
        cb_pos, _      = get_city_last_pos(inputs.input_ids, tokenizer, city_b)

        if ca_pos < 0 or cb_pos < 0:
            print(f"  [Skip] {city_a}→{city_b}: position not found")
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # extract the attention matrix per layer
        # outputs.attentions: tuple of [1, n_heads, seq_len, seq_len], 36 layers total
        attn_to_ca_by_layer = []   # [36, 32]: attention to city_a per layer/head
        attn_to_cb_by_layer = []   # [36, 32]: attention to city_b per layer/head

        for layer_idx in range(N_LAYERS):
            attn = outputs.attentions[layer_idx][0].cpu().float()  # [32, seq, seq]
            attn_to_ca = attn[:, the_pos, ca_pos].numpy()  # [32]
            attn_to_cb = attn[:, the_pos, cb_pos].numpy()  # [32]
            attn_to_ca_by_layer.append(attn_to_ca.tolist())
            attn_to_cb_by_layer.append(attn_to_cb.tolist())

        records.append({
            "city_a":              city_a,
            "city_b":              city_b,
            "direction":           item["direction"],
            "tokens":              tokens,
            "ca_pos":              ca_pos,
            "cb_pos":              cb_pos,
            "the_pos":             the_pos,
            "attn_to_ca_by_layer": attn_to_ca_by_layer,  # [36, 32]
            "attn_to_cb_by_layer": attn_to_cb_by_layer,  # [36, 32]
        })

        print(f"  [{item['direction']:5s}] {city_a}→{city_b}  "
              f"ca_pos={ca_pos}('{tokens[ca_pos]}')  "
              f"cb_pos={cb_pos}('{tokens[cb_pos]}')  "
              f"the_pos={the_pos}")

    return records


# ─────────────────────────────────────────────────────────────
# Step 3: aggregate analysis
# ─────────────────────────────────────────────────────────────
def analyze_records(records):
    """
    Aggregate across prompts:
      mean_attn_to_ca[layer, head]: mean attention to city_a per layer/head
      mean_attn_to_cb[layer, head]: mean attention to city_b per layer/head
      top_head_per_layer_ca[layer]: head with max attention to city_a per layer
      max_attn_per_layer_ca[layer]: max attention to city_a per layer
    """
    n = len(records)
    mean_ca = np.zeros((N_LAYERS, N_HEADS))
    mean_cb = np.zeros((N_LAYERS, N_HEADS))

    for r in records:
        mean_ca += np.array(r["attn_to_ca_by_layer"])
        mean_cb += np.array(r["attn_to_cb_by_layer"])

    mean_ca /= n
    mean_cb /= n

    # top head per layer (to city_a)
    top_head_ca    = np.argmax(mean_ca, axis=1)           # [36]
    max_attn_ca    = np.max(mean_ca, axis=1)              # [36]
    mean_max_cb    = np.max(mean_cb, axis=1)              # [36]

    # Head 24 attention to city_a per layer
    h24_attn_ca    = mean_ca[:, TARGET_HEAD]              # [36]
    # Head 24 attention to city_b per layer
    h24_attn_cb    = mean_cb[:, TARGET_HEAD]              # [36]

    # print key-layer info
    print(f"\n{'─'*65}")
    print(f"  {'Layer':>6}  {'Top Head(ca)':>12}  {'Max Attn(ca)':>13}  "
          f"{'H24 Attn(ca)':>13}  {'H24 Attn(cb)':>13}")
    print(f"{'─'*65}")
    for l in range(N_LAYERS):
        mark_l26 = " ← L26" if l == TARGET_LAYER else ""
        mark_high = " ★" if max_attn_ca[l] > 0.1 else ""
        print(f"  Layer {l:2d}:  "
              f"Head {top_head_ca[l]:2d}        "
              f"{max_attn_ca[l]:>12.4f}  "
              f"{h24_attn_ca[l]:>12.4f}  "
              f"{h24_attn_cb[l]:>12.4f}"
              f"{mark_l26}{mark_high}")

    # top 5 layers with strongest attention to city_a
    top5_layers = np.argsort(max_attn_ca)[::-1][:5]
    print(f"\n  [Top 5 layers with strongest attention to city_a]")
    for l in top5_layers:
        print(f"  Layer {l:2d}: Top Head = Head {top_head_ca[l]:2d}  "
              f"max_attn = {max_attn_ca[l]:.4f}")

    return {
        "mean_ca":       mean_ca,
        "mean_cb":       mean_cb,
        "top_head_ca":   top_head_ca,
        "max_attn_ca":   max_attn_ca,
        "mean_max_cb":   mean_max_cb,
        "h24_attn_ca":   h24_attn_ca,
        "h24_attn_cb":   h24_attn_cb,
        "top5_layers_ca": top5_layers.tolist(),
    }


# ─────────────────────────────────────────────────────────────
# Step 4: visualization
# ─────────────────────────────────────────────────────────────
def plot_heatmap(mean_ca, mean_cb, save_path):
    """
    Two side-by-side heatmaps: mean attention to city_a and to city_b per layer/head.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#F8F8F8")

    for ax, data, title, cmap in zip(
        axes,
        [mean_ca, mean_cb],
        ["the → city_a  (avg across prompts)",
         "the → city_b  (avg across prompts)"],
        ["Blues", "Reds"],
    ):
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=0, vmax=data.max())
        ax.set_xlabel("Head Index", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(f"L26 Head 24 Attention Weights:\n{title}",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(range(0, N_HEADS, 4))
        ax.set_yticks(range(0, N_LAYERS, 2))

        # annotate L26 and Head 24
        ax.axhline(TARGET_LAYER - 0.5, color="#DC2626",
                   linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(TARGET_HEAD - 0.5, color="#DC2626",
                   linewidth=1.5, linestyle="--", alpha=0.8)
        ax.text(TARGET_HEAD, TARGET_LAYER, "H24",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.suptitle("Attention Weight from 'the' Position\n"
                 "to city_a (left) and city_b (right) — All Layers & Heads",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 1] saved: {save_path.name}")


def plot_top_heads(mean_ca, top_head_ca, max_attn_ca, h24_attn_ca, save_path):
    """
    Per-layer max attention to city_a vs Head 24 attention.
    """
    layers = list(range(N_LAYERS))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F0F0F0")

    # top attending head per layer
    ax.plot(layers, max_attn_ca, color="#2563EB", linewidth=2,
            label="Max Attn to city_a (Top Head per layer)", zorder=3)

    # Head 24 attention
    ax.plot(layers, h24_attn_ca, color="#DC2626", linewidth=1.5,
            linestyle="--", label=f"Head 24 Attn to city_a", zorder=3)

    # annotate top-head index
    for l in layers:
        if max_attn_ca[l] > 0.05:
            ax.text(l, max_attn_ca[l] + 0.005,
                    f"H{top_head_ca[l]}",
                    ha="center", fontsize=6, color="#1D4ED8")

    # annotate L26
    ax.axvline(TARGET_LAYER, color="#6B7280", linestyle=":",
               linewidth=1.5, alpha=0.7)
    ax.text(TARGET_LAYER + 0.3, ax.get_ylim()[1] * 0.95,
            "L26", fontsize=9, color="#6B7280", fontweight="bold")

    ax.axhline(0.1, color="gray", linestyle="--",
               linewidth=1, alpha=0.5, label="Threshold=0.1")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Avg Attention Weight", fontsize=11)
    ax.set_title("city_a Information Flow to 'the' Position\n"
                 "(Max attention per layer and Head 24 attention)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 2] saved: {save_path.name}")


def plot_ca_vs_cb(max_attn_ca, mean_max_cb, h24_attn_cb, save_path):
    """
    city_a vs city_b attention-flow comparison.
    """
    layers = list(range(N_LAYERS))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F0F0F0")

    ax.plot(layers, max_attn_ca,  color="#2563EB", linewidth=2,
            label="Max Attn to city_a (any head)", zorder=3)
    ax.plot(layers, mean_max_cb,  color="#DC2626", linewidth=2,
            label="Max Attn to city_b (any head)", zorder=3)
    ax.plot(layers, h24_attn_cb,  color="#DC2626", linewidth=1.5,
            linestyle="--", label="Head 24 Attn to city_b", zorder=3)

    ax.axvline(TARGET_LAYER, color="#6B7280", linestyle=":",
               linewidth=1.5, alpha=0.7)
    ax.text(TARGET_LAYER + 0.3, ax.get_ylim()[1] * 0.95,
            "L26", fontsize=9, color="#6B7280", fontweight="bold")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Avg Attention Weight", fontsize=11)
    ax.set_title("Information Flow Comparison:\ncity_a (blue) vs city_b (red) → 'the' Position",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig 3] saved: {save_path.name}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  city_a information flow analysis")
    print("  (layer-wise attention weight tracing)")
    print("=" * 55)

    print(f"\n=== Step 1: load data ===")
    prompts_data = load_prompts()

    print(f"\n=== Step 2: load model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager"
    )
    model.eval()
    print(f"[Load] done, device: {next(model.parameters()).device}")

    print(f"\n=== Step 3: extract all-layer attention ===")
    records = extract_attention_all_layers(model, tokenizer, prompts_data)

    del model
    torch.cuda.empty_cache()
    print(f"\n[Model] memory released.")

    print(f"\n=== Step 4: aggregate analysis ===")
    stats = analyze_records(records)

    print(f"\n=== Step 5: visualization ===")
    plot_heatmap(
        stats["mean_ca"], stats["mean_cb"],
        FIG_DIR / "city_a_attn_heatmap.png"
    )
    plot_top_heads(
        stats["mean_ca"], stats["top_head_ca"],
        stats["max_attn_ca"], stats["h24_attn_ca"],
        FIG_DIR / "city_a_attn_top_heads.png"
    )
    plot_ca_vs_cb(
        stats["max_attn_ca"], stats["mean_max_cb"],
        stats["h24_attn_cb"],
        FIG_DIR / "city_a_attn_vs_cb.png"
    )

    print(f"\n=== Step 6: save results ===")
    output = {
        "model":         MODEL_NAME,
        "n_prompts":     len(records),
        "n_layers":      N_LAYERS,
        "n_heads":       N_HEADS,
        "target_layer":  TARGET_LAYER,
        "target_head":   TARGET_HEAD,
        "top5_layers_ca": stats["top5_layers_ca"],
        "per_layer": [
            {
                "layer":          l,
                "top_head_ca":    int(stats["top_head_ca"][l]),
                "max_attn_ca":    round(float(stats["max_attn_ca"][l]), 6),
                "h24_attn_ca":    round(float(stats["h24_attn_ca"][l]), 6),
                "h24_attn_cb":    round(float(stats["h24_attn_cb"][l]), 6),
                "all_heads_ca":   [round(float(v), 6)
                                   for v in stats["mean_ca"][l]],
            }
            for l in range(N_LAYERS)
        ],
        "prompts": [
            {
                "city_a":    r["city_a"],
                "city_b":    r["city_b"],
                "direction": r["direction"],
                "ca_pos":    r["ca_pos"],
                "cb_pos":    r["cb_pos"],
                "the_pos":   r["the_pos"],
            }
            for r in records
        ],
    }
    path = RESULT_DIR / "city_a_attn_path.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[Saved] {path}")

    print(f"\n{'='*55}")
    print(f"  Experiment done")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()