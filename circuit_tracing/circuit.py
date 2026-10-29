#!/usr/bin/env python3
"""
Assemble the geographic-direction reasoning circuit from three edge weights.

Draws the circuit for four example prompts using:
  edge 1  city token -> L8 geographic feature   (SAE activation)
  edge 2  L8 feature -> L26 via routing head     (attention weight)
  edge 3  L26 direction feature -> output logit  (contrastive DLA)

L26 direction features are selected by direction selectivity x DLA.
"""
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from safetensors.torch import load_file

# ─── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
L26_SAE_DIR      = BASE / "sae/final_sae/layer_26"
FEAT_L8_JSON     = BASE / "feature_discovery/results/city_features_L8.json"
CITY_ACTS        = BASE / "sae/activations/city_sae_acts.npy"
CITY_NAMES       = BASE / "sae/activations/city_names.npy"
THE_ACTS_L26     = BASE / "sae/activations/the_sae_acts_l26.npy"
PAIRS_JSONL      = BASE / "data_process/dataset/balanced_city_pairs.jsonl"
ATTN_HEAD24_JSON = BASE / "circuit_tracing/results/head24_attention.json"
ATTN_HEAD25_JSON = BASE / "circuit_tracing/results/city_a_attn_path.json"
FIG_DIR          = BASE / "circuit_tracing/figures"
OUT_JSON         = BASE / "circuit_tracing/results/circuit.json"

MODEL_NAME   = "Qwen/Qwen3-8B"
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
H24, H25     = 24, 25

# named features for semantic labels
NAMED_FEATS = {37159: "Xinjiang", 425: "Tibet", 58833: "NE-China"}

# four example prompts, one per direction
FOUR_PROMPTS = [
    {"City_A": "Lhasa",   "City_B": "Shenyang",  "Direction": "East"},
    {"City_A": "Shenyang",  "City_B": "Lhasa",   "Direction": "West"},
    {"City_A": "Ürümqi",  "City_B": "Lhasa",   "Direction": "South"},
    {"City_A": "Lhasa",   "City_B": "Ürümqi",  "Direction": "North"},
]

# L26 feature-selection parameters
TOP_L26_SELECT = 3    # L26 features shown per diagram
MIN_ACTIVE_RATIO = 0.1  # minimum activation ratio in that direction


# ─── SAE (W_dec only) ─────────────────────────────────────────────────────────
def load_w_dec(sae_dir: Path, device):
    """Load only W_dec to save memory (no full SAE)."""
    raw = load_file(str(sae_dir / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec"))
    return w_dec.to(device).float()   # [d_sae, d_model]


# ─── Step 2: edge 1, city activations from city_sae_acts ─────────────────────
def get_city_l8_top(city_name: str, city_names, city_acts, sig_l8, topk=3):
    """
    Return the top-k significant L8 features by activation for a city.
    [(fid, activation_value), ...]
    """
    idx_arr = np.where(city_names == city_name)[0]
    if len(idx_arr) == 0:
        # fuzzy match after stripping special characters
        for i, n in enumerate(city_names):
            if city_name.lower() in str(n).lower():
                idx_arr = np.array([i]); break
    if len(idx_arr) == 0:
        print(f"  [Warning] city {city_name} not found in city_names")
        return []
    idx = idx_arr[0]
    acts = city_acts[idx]                          # [65536]
    sig_acts = [(fid, float(acts[fid])) for fid in sig_l8 if acts[fid] > 0]
    sig_acts.sort(key=lambda x: x[1], reverse=True)
    return sig_acts[:topk]


# ─── Step 3: select L26 direction features (selectivity + DLA) ───────────────
def select_direction_features(direction: str,
                               the_acts_l26: np.ndarray,
                               pair_directions: list,
                               w_dec_l26: torch.Tensor,
                               w_u: torch.Tensor,
                               correct_token: int,
                               top_k: int = TOP_L26_SELECT):
    dirs = ["East", "West", "South", "North"]
    dir_indices = {d: [i for i, pd in enumerate(pair_directions) if pd == d]
                   for d in dirs}

    target_idx = dir_indices[direction]
    if len(target_idx) == 0:
        return []

    # mean activation per direction
    dir_means = {}
    for d in dirs:
        idx = dir_indices[d]
        if idx:
            dir_means[d] = the_acts_l26[idx].mean(axis=0)
        else:
            dir_means[d] = np.zeros(the_acts_l26.shape[1])

    # activation-ratio filter
    active_ratio = (the_acts_l26[target_idx] > 0).mean(axis=0)
    valid_mask   = active_ratio >= MIN_ACTIVE_RATIO

    # per-direction DLA
    u_vecs = {d: w_u[DIR_TOKEN_IDS[d]].float() for d in dirs}
    dla_by_dir = {}
    for d in dirs:
        proj = (w_dec_l26 @ u_vecs[d]).cpu().numpy()
        dla_by_dir[d] = dir_means[d] * proj

    # contrastive DLA: target direction minus mean of others
    other_dirs = [d for d in dirs if d != direction]
    dla_target       = dla_by_dir[direction]
    dla_others_mean  = np.stack([dla_by_dir[d] for d in other_dirs]).mean(axis=0)
    contrastive_dla  = dla_target - dla_others_mean

    # combined score: keep positive contrastive contribution
    score = np.maximum(contrastive_dla, 0) * valid_mask.astype(float)

    top_fids = np.argsort(score)[::-1][:top_k * 5]

    results = []
    for fid in top_fids:
        if not valid_mask[fid]:
            continue
        results.append((int(fid), float(score[fid]), float(contrastive_dla[fid])))
        if len(results) == top_k:
            break

    return results


# ─── Step 4: compute edge 3 (DLA) ────────────────────────────────────────────
def compute_dla(fids, the_acts_l26, pair_directions, direction,
                w_dec_l26, w_u, correct_token):
    dirs = ["East", "West", "South", "North"]
    dir_indices = {d: [i for i, pd in enumerate(pair_directions) if pd == d]
                   for d in dirs}

    dir_means = {}
    for d in dirs:
        idx = dir_indices[d]
        dir_means[d] = the_acts_l26[idx].mean(axis=0) if idx else np.zeros(the_acts_l26.shape[1])

    u_vecs = {d: w_u[DIR_TOKEN_IDS[d]].float() for d in dirs}
    dla_by_dir = {}
    for d in dirs:
        proj = (w_dec_l26 @ u_vecs[d]).cpu().numpy()
        dla_by_dir[d] = dir_means[d] * proj

    other_dirs = [d for d in dirs if d != direction]
    dla_target      = dla_by_dir[direction]
    dla_others_mean = np.stack([dla_by_dir[d] for d in other_dirs]).mean(axis=0)
    contrastive_dla = dla_target - dla_others_mean

    return {fid: float(contrastive_dla[fid]) for fid in fids}


# ─── Step 5: read edge 2 (attention) from cached results ─────────────────────
def load_attention_weights(attn_h24_path, attn_h25_path):
    with open(attn_h24_path) as f:
        h24_data = json.load(f)
    with open(attn_h25_path) as f:
        h25_data = json.load(f)

    # h24_data is a list; average attn_to_cb over prompts
    h24_cb = float(np.mean([item["attn_to_cb"] for item in h24_data]))

    # h25_data is a dict; take layer-26 max attention to city_a
    per_layer = {entry["layer"]: entry for entry in h25_data["per_layer"]}
    h25_ca = float(per_layer[26]["max_attn_ca"])

    return h24_cb, h25_ca


# ─── Step 6: draw the circuit ─────────────────────────────────────────────────
def draw_circuit(prompt_info: dict,
                 ca_top3: list,
                 cb_top3: list,
                 l26_feats: list,
                 dla_vals: dict,
                 h24_attn: float,
                 h25_attn: float,
                 out_path: Path):

    city_a    = prompt_info["City_A"]
    city_b    = prompt_info["City_B"]
    direction = prompt_info["Direction"]
    total_dla = sum(dla_vals.get(fid, 0) for fid, *_ in l26_feats)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("#FAFAF8")
    fig.patch.set_facecolor("#FAFAF8")

    C_TOKEN = "#F5EFE6"
    C_L8    = "#EDE3D4"
    C_L26   = "#C8B89A"
    C_LOGIT = "#8B5E3C"
    C_POS   = "#6B3A2A"
    C_NEG   = "#2A4A6B"
    EC      = "#7A6550"

    def box(ax, x, y, w, h, text, fc, ec=EC, fontsize=8.5,
            bold=False, fc_text="black"):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              fc=fc, ec=ec, lw=1.3, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=fc_text, zorder=4)

    def arrow(ax, x0, y0, x1, y1, weight, max_w=5.0,
              label=None, label_side="left"):
        w   = min(max(abs(weight) * 1.5, 0.5), max_w)
        col = C_POS if weight >= 0 else C_NEG
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=w, mutation_scale=12), zorder=2)
        if label:
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            offset_x = -0.25 if label_side == "left" else 0.25
            ax.text(mx + offset_x, my, label, fontsize=6, color="#555",
                    ha="center", va="center", zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    # ── layout ────────────────────────────────────────────────────────────────
    y0 = 1.0
    y1 = 3.2
    y2 = 6.0
    y3 = 8.2

    tok_w, tok_h   = 2.2, 0.60
    feat_w, feat_h = 2.2, 0.72
    l26_w,  l26_h  = 2.6, 0.72

    x_ca = 3.5
    x_cb = 11.5

    # ── Row 0: tokens ──────────────────────────────────────────────────────────
    box(ax, x_ca, y0, tok_w, tok_h, city_a, C_TOKEN, fontsize=9.5)
    box(ax, x_cb, y0, tok_w, tok_h, city_b, C_TOKEN, fontsize=9.5)

    # ── Row 1: L8 features ────────────────────────────────────────────────────
    ca_xs = [1.4, 3.5, 5.6]
    cb_xs = [9.4, 11.5, 13.6]
    ca_nodes, cb_nodes = [], []

    for i, (fid, act_val) in enumerate(ca_top3):
        x = ca_xs[i]
        name = NAMED_FEATS.get(fid, "")
        label = f"#{fid}\n({name})" if name else f"#{fid}"
        box(ax, x, y1, feat_w, feat_h, label, C_L8, fontsize=7.5)
        ca_nodes.append((x, y1, act_val))
        arrow(ax, x_ca, y0+tok_h/2, x, y1-feat_h/2,
              act_val/20.0, label=f"act={act_val:.1f}", label_side="left")

    for i, (fid, act_val) in enumerate(cb_top3):
        x = cb_xs[i]
        name = NAMED_FEATS.get(fid, "")
        label = f"#{fid}\n({name})" if name else f"#{fid}"
        box(ax, x, y1, feat_w, feat_h, label, C_L8, fontsize=7.5)
        cb_nodes.append((x, y1, act_val))
        arrow(ax, x_cb, y0+tok_h/2, x, y1-feat_h/2,
              act_val/20.0, label=f"act={act_val:.1f}", label_side="right")

    # ── Row 2: L26 direction features ─────────────────────────────────────────
    n = len(l26_feats)
    if n == 1:   l26_xs = [7.5]
    elif n == 2: l26_xs = [5.5, 9.5]
    else:        l26_xs = [4.0, 7.5, 11.0]

    l26_nodes = []
    for i, (fid, sel, _) in enumerate(l26_feats):
        x = l26_xs[i]
        dla = dla_vals.get(fid, 0.0)
        box(ax, x, y2, l26_w, l26_h, f"#{fid}", C_L26, fontsize=9)
        l26_nodes.append((x, y2, dla))

    # L8 -> L26: ca side, top1 L8 -> top1 L26
    arrow(ax, ca_nodes[0][0], ca_nodes[0][1]+feat_h/2,
          l26_nodes[0][0], l26_nodes[0][1]-l26_h/2,
          h25_attn,
          label=f"via Head25\nattn={h25_attn:.3f}",
          label_side="left")

    # L8 -> L26: cb side, top1 L8 -> top1 L26
    arrow(ax, cb_nodes[0][0], cb_nodes[0][1]+feat_h/2,
          l26_nodes[0][0], l26_nodes[0][1]-l26_h/2,
          h24_attn,
          label=f"via Head24\nattn={h24_attn:.3f}",
          label_side="right")

    # ── Row 3: logit ───────────────────────────────────────────────────────────
    box(ax, 7.5, y3, 3.8, 0.80,
        f"logit[{direction}]  ΣDLA={total_dla:+.3f}",
        C_LOGIT, ec="#4A2810", fontsize=9.5, bold=True, fc_text="white")

    for (x, y, dla) in l26_nodes:
        arrow(ax, x, y+l26_h/2, 7.5, y3-0.40, dla,
              label=f"cDLA={dla:+.2f}", label_side="left")

    # ── row labels ────────────────────────────────────────────────────────────
    for y, txt in [(y0, "Input\ntokens"),
                   (y1, "L8 SAE\ncity features"),
                   (y2, "L26 SAE\ndirection features"),
                   (y3, "Output logit")]:
        ax.text(15.4, y, txt, ha="left", va="center",
                fontsize=7, color="#7A6550", style="italic", clip_on=False)

    # ── caption ───────────────────────────────────────────────────────────────
    ax.text(7.5, 0.25,
            "edge①: SAE activation value  |  "
            "edge②: attention weight (via Head)  |  "
            "edge③: Contrastive Direct Logit Attribution",
            ha="center", fontsize=7, color="#7A6550")

    # ── title ─────────────────────────────────────────────────────────────────
    prompt_str = f"Relative to {city_a}, {city_b} is to the"
    ax.set_title(f'Circuit: "{prompt_str}"  →  {direction}',
                 fontsize=11, fontweight="bold", pad=10, color="#3A2010")

    # ── legend ────────────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(fc=C_TOKEN, ec=EC,        label="Input token"),
        mpatches.Patch(fc=C_L8,   ec=EC,         label="L8 city feature"),
        mpatches.Patch(fc=C_L26,  ec=EC,         label="L26 direction feature"),
        mpatches.Patch(fc=C_LOGIT,ec="#4A2810",  label="Output logit (ΣDLA)"),
        mpatches.Patch(fc=C_POS,  ec="none",     label="Positive weight"),
        mpatches.Patch(fc=C_NEG,  ec="none",     label="Negative weight"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=7,
              framealpha=0.9, edgecolor=EC)

    feat_desc = "  |  ".join(f"#{fid} = {name}"
                              for fid, name in NAMED_FEATS.items())
    ax.text(7.5, -0.1, feat_desc, ha="center", fontsize=7.5,
            color="#7A6550", transform=ax.transData)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Figure] {out_path.name}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("circuit.py  circuit from SAE activations / attention weights / DLA")
    print("=" * 60)

    # ── load static data ──────────────────────────────────────────────────────
    print("\n=== Step 1: load data ===")

    with open(FEAT_L8_JSON) as f:
        l8_data = json.load(f)
    sig_l8 = [int(x["feature_id"]) for x in l8_data["significant_features"]]
    print(f"[L8] {len(sig_l8)} significant features")

    city_acts  = np.load(CITY_ACTS)
    city_names = np.load(CITY_NAMES, allow_pickle=True)
    print(f"[city_sae_acts] {city_acts.shape}")

    the_acts_l26 = np.load(THE_ACTS_L26)
    print(f"[the_sae_acts_l26] {the_acts_l26.shape}")

    pair_directions = []
    with open(PAIRS_JSONL) as f:
        for line in f:
            item = json.loads(line)
            pair_directions.append(item["Direction"])
    print(f"[pairs] {len(pair_directions)} prompts, direction distribution: "
          + str({d: pair_directions.count(d) for d in ["East","West","South","North"]}))

    h24_attn, h25_attn = load_attention_weights(ATTN_HEAD24_JSON, ATTN_HEAD25_JSON)
    print(f"[Attention] Head24→cb: {h24_attn:.3f}  Head25→ca: {h25_attn:.3f}")

    # ── load model weight matrices (no forward pass) ──────────────────────────
    print("\n=== Step 2: load weight matrices ===")
    from transformers import AutoModelForCausalLM
    print("  loading lm_head.weight ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    w_u = model.lm_head.weight.detach().float()   # [vocab, d_model]
    del model
    torch.cuda.empty_cache()
    print(f"  W_U: {w_u.shape}")

    print("  loading L26 SAE W_dec ...")
    w_dec_l26 = load_w_dec(L26_SAE_DIR, device="cpu")
    print(f"  W_dec_l26: {w_dec_l26.shape}")

    # ── four-direction circuit diagrams ───────────────────────────────────────
    print("\n=== Step 3: compute and plot per direction ===")
    all_results = {}

    for item in FOUR_PROMPTS:
        city_a    = item["City_A"]
        city_b    = item["City_B"]
        direction = item["Direction"]
        correct_token = DIR_TOKEN_IDS[direction]
        print(f"\n[{direction}] {city_a} → {city_b}")

        # edge 1: L8 activation
        ca_top3 = get_city_l8_top(city_a, city_names, city_acts, sig_l8, topk=3)
        cb_top3 = get_city_l8_top(city_b, city_names, city_acts, sig_l8, topk=3)
        print(f"  ca top3: {[(fid, f'{v:.2f}') for fid,v in ca_top3]}")
        print(f"  cb top3: {[(fid, f'{v:.2f}') for fid,v in cb_top3]}")

        # Step 3: select L26 features
        l26_feats = select_direction_features(
            direction, the_acts_l26, pair_directions,
            w_dec_l26, w_u, correct_token
        )
        print(f"  L26 top{TOP_L26_SELECT}: "
              + str([(fid, f"sel={s:.2f}", f"dla={d:+.3f}")
                     for fid, s, d in l26_feats]))

        # edge③: DLA
        fids = [fid for fid, *_ in l26_feats]
        dla_vals = compute_dla(fids, the_acts_l26, pair_directions,
                               direction, w_dec_l26, w_u, correct_token)
        print(f"  DLA: {dla_vals}")

        # check named-feature ranking
        print(f"  validation features in ca_top3:")
        for fid, name in NAMED_FEATS.items():
            ranks = [i+1 for i, (f, _) in enumerate(ca_top3) if f == fid]
            print(f"    {name}(#{fid}): {'#'+str(ranks[0]) if ranks else '-'}")
        print(f"  validation features in cb_top3:")
        for fid, name in NAMED_FEATS.items():
            ranks = [i+1 for i, (f, _) in enumerate(cb_top3) if f == fid]
            print(f"    {name}(#{fid}): {'#'+str(ranks[0]) if ranks else '-'}")

        # plot
        out_path = FIG_DIR / f"circuit_{direction.lower()}.png"
        draw_circuit(item, ca_top3, cb_top3, l26_feats, dla_vals,
                     h24_attn, h25_attn, out_path)

        all_results[direction] = {
            "city_a": city_a, "city_b": city_b,
            "ca_top3": [(int(f), float(v)) for f, v in ca_top3],
            "cb_top3": [(int(f), float(v)) for f, v in cb_top3],
            "l26_feats": [(int(f), float(s), float(d)) for f, s, d in l26_feats],
            "dla_vals": {str(k): float(v) for k, v in dla_vals.items()},
            "h24_attn": h24_attn, "h25_attn": h25_attn,
        }

    # ── save JSON ─────────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {OUT_JSON}")
    print("\n" + "=" * 60)
    print("Done")


if __name__ == "__main__":
    main()