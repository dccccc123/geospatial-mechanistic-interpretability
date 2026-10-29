#!/usr/bin/env python3
"""
circuit_grid.py

Single-panel weighted circuit diagram for the South prompt
(Relative to Urumqi, Lhasa is to the ___), compact bottom-up layout, PDF output.
Every edge weight is read from the SAME forward pass of the prompt:

  edge 1  city token -> L8 geographic feature   = SAE activation a_f (per prompt)
  edge 2  L8 feature -> L26 via head             = attention weight  (per prompt)
  edge 3  L26 direction feature -> output logit   = cDLA (Eq. 5, discovery-set)
"""
import os, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parent.parent
L26_SAE_DIR  = BASE / "sae/final_sae/layer_26"
CITY_ACTS    = BASE / "sae/activations/city_sae_acts.npy"
CITY_NAMES   = BASE / "sae/activations/city_names.npy"
FEAT_L8_JSON = BASE / "feature_discovery/results/city_features_L8.json"
FIG_DIR      = BASE / "circuit_tracing/figures"
OUT_PDF      = FIG_DIR / "circuit_south.pdf"
OUT_JSON     = BASE / "circuit_tracing/results/circuit_perprompt.json"
THE_ACTS     = BASE / "sae/activations/the_sae_acts_l26.npy"
PAIRS        = BASE / "data_process/dataset/balanced_city_pairs.jsonl"

# ── constants (project conventions) ──────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3-8B"
THE_TOKEN_ID  = 279
OUTLIER_DIMS  = [2276, 233]
PROMPT_TPL    = "Relative to {city_a}, {city_b} is to the"
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
DIRS          = ["East", "West", "South", "North"]
H24, H25      = 24, 25
TARGET_LAYER  = 26

# fixed direction-selective output-node features (Table III)
DIR_FEATS = {"East": 47636, "West": 56725, "South": 10801, "North": 26800}
NAMED_FEATS = {37159: "Xinjiang", 425: "Tibet", 58833: "NE China", 43181: "C-E China"}

# the panel we draw (South). all four are extracted for the JSON record.
FOUR = [
    ("Lhasa",    "Shenyang", "East"),
    ("Shenyang", "Lhasa",    "West"),
    ("Ürümqi",   "Lhasa",    "South"),
    ("Lhasa",    "Ürümqi",   "North"),
]
DRAW_DIRECTION = "South"
DISPLAY = {"Ürümqi": "Urumqi"}

# ── paper palette ────────────────────────────────────────────────────────────
BG       = "#FAFAF8"
STONE    = "#C4BCB0"
GRAY     = "#6B6055"
FAINT    = "#A89E92"
DARK     = "#1E1410"
ORANGE   = "#E07B39"
ORANGE_T = "#B45A1E"   # deep orange text
ORANGE_S = "#C77A3C"   # secondary orange text
ORANGE_F = "#FBEEE0"   # light orange fill
SIENNA   = "#A0522D"
SIENNA_F = "#F6EAE0"
ESPRESSO = "#2D1B0E"
ESP_SOFT = "#5B4636"   # softened espresso (edge 3 / secondary text)
ESP_F    = "#EBE3DA"   # light espresso fill
LOGIT_F  = "#F1EAE2"
TOK_F    = "#F1EDE6"


# ── data extraction ──────────────────────────────────────────────────────────
def load_w_dec(sae_dir, device="cpu"):
    raw = load_file(str(sae_dir / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec"))
    return w_dec.to(device).float()


def find_positions(prompt, tokenizer, city_a, city_b):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                    return_offsets_mapping=True)
    ids  = enc["input_ids"][0]
    offs = enc["offset_mapping"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())

    the_matches = [i for i, (s, e) in enumerate(offs)
                   if prompt[s:e].strip().lower() == "the"]
    if not the_matches:
        the_matches = [i for i, t in enumerate(toks)
                       if t.lstrip("▁Ġ").lower() == "the"]
    the_pos = the_matches[-1]

    def city_last(city):
        cl, last = city.lower(), -1
        for i, t in enumerate(toks):
            c = t.lstrip("▁Ġ").lower()
            if c and cl.endswith(c):
                last = i
        return last

    return ids.unsqueeze(0), the_pos, city_last(city_a), city_last(city_b)


def top1_l8(city, city_names, city_acts, sig):
    idx = np.where(city_names == city)[0]
    if len(idx) == 0:
        for i, n in enumerate(city_names):
            if city.lower() in n.lower():
                idx = np.array([i]); break
    if len(idx) == 0:
        raise ValueError(f"city not found in city_names: {city}")
    acts = city_acts[idx[0]]
    cand = [(int(f), float(acts[f])) for f in sig if acts[f] > 0]
    cand.sort(key=lambda x: -x[1])
    return cand[0]


def compute_eq5_cdla(fid, the_acts, dir_idx, w_dec, w_u_dir):
    """Eq. (5) cDLA: cDLA(f,d) = mean_act_d * proj_d - mean_{d' != d}(mean_act_{d'} * proj_{d'}).

    Uses discovery-set mean activations, consistent with Table IV / cdla_rank_L26.py.
    Returns {direction: cDLA}.
    """
    mean_act = {d: float(the_acts[dir_idx[d]][:, fid].mean()) for d in DIRS}
    proj = {d: float(w_dec[fid] @ w_u_dir[d]) for d in DIRS}
    cdla = {}
    for d in DIRS:
        others = [o for o in DIRS if o != d]
        cdla[d] = mean_act[d] * proj[d] - float(np.mean([mean_act[o] * proj[o] for o in others]))
    return cdla


def extract():
    print("[load] city SAE acts / names / sig features")
    city_acts  = np.load(str(CITY_ACTS))
    city_names = np.array([str(x) for x in
                           np.load(str(CITY_NAMES), allow_pickle=True)])
    sig = [int(f["feature_id"])
           for f in json.load(open(FEAT_L8_JSON))["significant_features"]]

    print("[load] Qwen3-8B (eager attn)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="eager",
        trust_remote_code=True, device_map="cuda")
    model.eval()

    print("[load] L26 SAE + W_dec + W_U rows")
    sae = SAE.load_from_disk(str(L26_SAE_DIR), device="cuda"); sae.eval()
    w_dec = load_w_dec(L26_SAE_DIR, "cpu")
    w_u_dir = {d: model.lm_head.weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
               for d in DIRS}

    print("[load] discovery-set L26 activations for Eq(5) cDLA")
    the_acts = np.load(str(THE_ACTS))
    dirs = [json.loads(l)["Direction"] for l in open(PAIRS) if l.strip()]
    dir_idx = {d: [i for i, x in enumerate(dirs) if x == d] for d in DIRS}
    eq5_cdla = {fid: compute_eq5_cdla(fid, the_acts, dir_idx, w_dec, w_u_dir)
                for fid in DIR_FEATS.values()}

    target_layer = model.model.layers[TARGET_LAYER]
    panels = []

    for city_a, city_b, direction in FOUR:
        prompt = PROMPT_TPL.format(city_a=city_a, city_b=city_b)
        ids, the_pos, ca_pos, cb_pos = find_positions(prompt, tokenizer, city_a, city_b)
        assert ca_pos >= 0 and cb_pos >= 0, f"position not found: {prompt}"

        cap = {}

        def hook(_m, _i, o):
            h = o[0].clone() if isinstance(o, tuple) else o.clone()
            h[:, :, OUTLIER_DIMS] = 0.0
            cap["hidden"] = h[:, the_pos, :].detach()
            return None

        handle = target_layer.register_forward_hook(hook)
        with torch.inference_mode():
            out = model(ids.to("cuda"), output_attentions=True)
        handle.remove()

        attn = out.attentions[TARGET_LAYER][0].float().cpu()
        h25_attn = float(attn[H25, the_pos, ca_pos])
        h24_attn = float(attn[H24, the_pos, cb_pos])

        hidden_vec = cap["hidden"].to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            feats = sae.encode(hidden_vec).squeeze(0).float().cpu().numpy()
        fid = DIR_FEATS[direction]
        a_f = float(feats[fid])
        cdla = eq5_cdla[fid][direction]   # Eq. (5) cDLA (discovery-set, consistent with Table IV)

        ca_fid, ca_act = top1_l8(city_a, city_names, city_acts, sig)
        cb_fid, cb_act = top1_l8(city_b, city_names, city_acts, sig)

        pred_logits = out.logits[0, -1].float().cpu()
        _ew = pred_logits[DIR_TOKEN_IDS["East"]].item() - pred_logits[DIR_TOKEN_IDS["West"]].item()
        _ns = pred_logits[DIR_TOKEN_IDS["South"]].item() - pred_logits[DIR_TOKEN_IDS["North"]].item()
        if abs(_ew) >= abs(_ns):
            pred_dir = "East" if _ew > 0 else "West"
        else:
            pred_dir = "South" if _ns > 0 else "North"

        p = {
            "direction": direction, "city_a": city_a, "city_b": city_b,
            "ca_fid": ca_fid, "ca_act": ca_act,
            "ca_region": NAMED_FEATS.get(ca_fid, f"#{ca_fid}"),
            "cb_fid": cb_fid, "cb_act": cb_act,
            "cb_region": NAMED_FEATS.get(cb_fid, f"#{cb_fid}"),
            "h25_attn": h25_attn, "h24_attn": h24_attn,
            "dir_fid": fid, "a_f": a_f, "cdla": cdla,
            "pred_dir": pred_dir, "pred_correct": pred_dir == direction,
        }
        print(f"  [{direction}] {city_a}->{city_b}  "
              f"ca #{ca_fid}({p['ca_region']}) a={ca_act:.2f} | "
              f"cb #{cb_fid}({p['cb_region']}) a={cb_act:.2f} | "
              f"h25={h25_attn:.3f} h24={h24_attn:.3f} | "
              f"#{fid} a_f={a_f:.2f} cDLA={cdla:+.2f} | pred={pred_dir}")
        panels.append(p)

    json.dump(panels, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_JSON}")
    return panels


# ── drawing (coordinates in SVG space: y grows downward, axis inverted) ───────
W, H = 560, 318


def _rbox(ax, x, y, w, h, fc, ec, lw):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=6",
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))


def _arr(ax, p0, p1, lw, color, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=1, shrinkB=2, capstyle="round", zorder=2))


def widths(p):
    e1 = max(p["ca_act"], p["cb_act"])
    e2 = max(p["h25_attn"], p["h24_attn"])
    e3 = abs(p["cdla"])
    def mk(mx):
        return lambda v: 2.0 + 1.5 * (abs(v) / mx if mx > 0 else 0)
    return mk(e1), mk(e2), mk(e3)


def draw_south(ax, p):
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_facecolor(BG)
    w1, w2, w3 = widths(p)
    LX, RX, CX = 150, 410, 280
    da = DISPLAY.get(p["city_a"], p["city_a"])   # reference (left, Head 25)
    db = DISPLAY.get(p["city_b"], p["city_b"])   # target (right, Head 24)
    prompt = PROMPT_TPL.format(city_a=da, city_b=db)

    # title: full prompt
    ax.text(CX, 20, f"“{prompt} ___”", ha="center", va="center",
            fontsize=11, style="italic", color=GRAY)

    # left rail
    for txt, yy in [("logit", 55), ("L26", 131), ("heads", 177),
                    ("L8", 223), ("tokens", 297)]:
        ax.text(28, yy, txt, ha="center", va="center", fontsize=8, color=FAINT)

    # output logit (top)
    _rbox(ax, 212, 40, 136, 30, LOGIT_F, ESP_SOFT, 1.2)
    ax.text(CX, 56, "logit: South", ha="center", va="center",
            fontsize=12, fontweight="medium", color=ESPRESSO)

    # edge 3
    _arr(ax, (CX, 108), (CX, 73), w3(p["cdla"]), ESP_SOFT)
    ax.text(CX + 9, 92, f"cDLA {p['cdla']:.2f}", ha="left", va="center",
            fontsize=9, color=GRAY)

    # L26 direction feature
    _rbox(ax, 184, 110, 192, 42, ESP_F, ESP_SOFT, 1.2)
    ax.text(CX, 127, f"#{p['dir_fid']}", ha="center", va="center",
            fontsize=12, fontweight="medium", color=ESPRESSO)
    ax.text(CX, 144, "South direction feature", ha="center", va="center",
            fontsize=9, color=ESP_SOFT)
    ax.text(CX, 168, "the position", ha="center", va="center",
            fontsize=8, style="italic", color=FAINT)

    # edge 2: converging curves via heads
    _arr(ax, (LX, 200), (252, 154), w2(p["h25_attn"]), SIENNA, rad=-0.25)
    _arr(ax, (RX, 200), (308, 154), w2(p["h24_attn"]), SIENNA, rad=0.25)
    _rbox(ax, 160, 167, 56, 20, SIENNA_F, SIENNA, 0.9)
    ax.text(188, 177, "Head 25", ha="center", va="center", fontsize=9, color=SIENNA)
    _rbox(ax, 344, 167, 56, 20, SIENNA_F, SIENNA, 0.9)
    ax.text(372, 177, "Head 24", ha="center", va="center", fontsize=9, color=SIENNA)
    ax.text(LX, 151, f"{p['h25_attn']:.2f}", ha="center", va="center",
            fontsize=9, color=GRAY)
    ax.text(RX, 151, f"{p['h24_attn']:.2f}", ha="center", va="center",
            fontsize=9, color=GRAY)

    # L8 geographic features (no ca/cb labels)
    _rbox(ax, 88, 202, 124, 42, ORANGE_F, ORANGE, 1.2)
    ax.text(LX, 219, f"#{p['ca_fid']}", ha="center", va="center",
            fontsize=12, fontweight="medium", color=ORANGE_T)
    ax.text(LX, 236, p["ca_region"], ha="center", va="center",
            fontsize=9, color=ORANGE_S)
    _rbox(ax, 348, 202, 124, 42, ORANGE_F, ORANGE, 1.2)
    ax.text(RX, 219, f"#{p['cb_fid']}", ha="center", va="center",
            fontsize=12, fontweight="medium", color=ORANGE_T)
    ax.text(RX, 236, p["cb_region"], ha="center", va="center",
            fontsize=9, color=ORANGE_S)

    # edge 1
    _arr(ax, (LX, 280), (LX, 247), w1(p["ca_act"]), ORANGE)
    _arr(ax, (RX, 280), (RX, 247), w1(p["cb_act"]), ORANGE)
    ax.text(LX + 9, 266, f"a {p['ca_act']:.1f}", ha="left", va="center",
            fontsize=9, color=GRAY)
    ax.text(RX + 9, 266, f"a {p['cb_act']:.1f}", ha="left", va="center",
            fontsize=9, color=GRAY)

    # tokens (bottom)
    _rbox(ax, 94, 282, 112, 30, TOK_F, STONE, 0.9)
    ax.text(LX, 298, da, ha="center", va="center", fontsize=11, color=DARK)
    _rbox(ax, 354, 282, 112, 30, TOK_F, STONE, 0.9)
    ax.text(RX, 298, db, ha="center", va="center", fontsize=11, color=DARK)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panels = extract()
    p = next(x for x in panels if x["direction"] == DRAW_DIRECTION)

    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.2, 3.66))
    fig.patch.set_facecolor(BG)
    draw_south(ax, p)
    fig.savefig(str(OUT_PDF), facecolor=BG, bbox_inches="tight", pad_inches=0.04)
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
