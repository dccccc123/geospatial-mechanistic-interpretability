#!/usr/bin/env python3
"""
Locate the bypass path from L8 geographic features to the L26 direction feature.

Cumulatively freezes the full attention output of layers 9..26 (all heads) while
ablating the L8 geographic features, and measures the remaining effect on the
direction feature at each step. This identifies which layer carries city
information into the "the" position outside the routing heads.
"""
import argparse
import gc
import json
import os
import warnings
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
CITY_FEAT_L8 = ROOT / "feature_discovery" / "results" / "city_features_L8.json"
L8_SAE_PATH = ROOT / "sae" / "final_sae" / "layer_8"
L26_SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
OUT_PATH = ROOT / "causal_verification" / "results" / "l8_geographic_bypass_layerscan.json"

MODEL_NAME = "Qwen/Qwen3-8B"
L8_LAYER = 8
L26_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
DIRECTION_FEATURES = {"East": 47636, "West": 56725, "South": 10801, "North": 26800}

# scanned layers: L9..L26 (18 layers, cumulative freeze)
SCAN_LAYERS = list(range(9, 27))


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["prompt"] = PROMPT_TPL.format(city_a=item["City_A"], city_b=item["City_B"])
            rows.append(item)
    return rows


def load_geographic_features():
    data = json.loads(CITY_FEAT_L8.read_text(encoding="utf-8"))
    return [int(x["feature_id"]) for x in data["significant_features"]]


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID=279 not found")
    return int(positions[-1].item())


def get_city_last_pos(input_ids: torch.Tensor, tokenizer, city: str) -> int:
    prompt_ids = input_ids[0].tolist()
    for city_ids in (
        tokenizer(" " + city, add_special_tokens=False).input_ids,
        tokenizer(city, add_special_tokens=False).input_ids,
    ):
        if not city_ids:
            continue
        last_pos = -1
        width = len(city_ids)
        for i in range(0, len(prompt_ids) - width + 1):
            if prompt_ids[i:i + width] == city_ids:
                last_pos = i + width - 1
        if last_pos >= 0:
            return last_pos
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    city_lower = city.lower()
    last_pos = -1
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("▁Ġ").lower()
        if clean and city_lower.endswith(clean):
            last_pos = i
    return last_pos


def model_predict_from_logits(logits: torch.Tensor) -> str:
    s_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    s_ns = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


# ── hooks ──────────────────────────────────────────────────────────
def make_l8_geo_hook(sae: SAE, positions, geo_fids):
    fid_tensor = torch.tensor(geo_fids, dtype=torch.long, device="cuda")

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None
        hidden_for_sae = hidden.clone()
        hidden_for_sae[:, :, OUTLIER_DIMS] = 0.0
        with torch.inference_mode():
            for pos in positions:
                h = hidden_for_sae[:, pos, :].to(device="cuda", dtype=torch.bfloat16)
                feats = sae.encode(h)
                feats[:, fid_tensor] = 0.0
                h_new = sae.decode(feats)
                hidden[:, pos, :] = h_new.to(dtype=hidden.dtype)
        if rest is not None:
            return (hidden, *rest)
        return hidden

    return hook


def make_l26_direction_capture_hook(sae: SAE, the_pos: int, store: dict):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_for_sae = hidden.clone()
        hidden_for_sae[:, :, OUTLIER_DIMS] = 0.0
        vec = hidden_for_sae[:, the_pos, :].to(device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            feat = sae.encode(vec)
        store["direction_feature_acts"] = {
            d: float(feat[0, DIRECTION_FEATURES[d]].detach().float().cpu().item())
            for d in DIR_LABELS
        }

    return hook


def make_capture_attn_hook(the_pos: int, store: dict, layer: int):
    def pre_hook(_module, inputs):
        store[f"clean_attn_{layer}"] = inputs[0][:, the_pos, :].clone()
        return inputs

    return pre_hook


def make_freeze_attn_hook(the_pos: int, clean_attn: torch.Tensor):
    def pre_hook(_module, inputs):
        inputs = list(inputs)
        inp = inputs[0].clone()
        inp[:, the_pos, :] = clean_attn
        inputs[0] = inp
        return tuple(inputs)

    return pre_hook


def load_model_and_saes():
    print("[Model] loading tokenizer/model/L8 SAE/L26 SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    l8_sae = SAE.load_from_disk(str(L8_SAE_PATH), device="cuda")
    l8_sae.eval()
    l26_sae = SAE.load_from_disk(str(L26_SAE_PATH), device="cuda")
    l26_sae.eval()
    return model, tokenizer, l8_sae, l26_sae


def run_experiment(args):
    rows = load_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    geo_fids = load_geographic_features()
    print(f"[Feature] geographic={len(geo_fids)}")
    model, tokenizer, l8_sae, l26_sae = load_model_and_saes()

    n_layers = len(SCAN_LAYERS)
    records = []
    for idx, row in enumerate(tqdm(rows, desc="[L8 geo bypass layerscan]", unit="prompt")):
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)
        ca_pos = get_city_last_pos(inputs.input_ids, tokenizer, row["City_A"])
        cb_pos = get_city_last_pos(inputs.input_ids, tokenizer, row["City_B"])
        if ca_pos < 0 or cb_pos < 0:
            continue

        # 1) clean forward: capture clean attention of L9..L26 at "the" + direction feature
        clean_store = {}
        clean_handles = []
        for L in SCAN_LAYERS:
            clean_handles.append(
                model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
                    make_capture_attn_hook(the_pos, clean_store, L)
                )
            )
        clean_handles.append(
            model.model.layers[L26_LAYER].register_forward_hook(
                make_l26_direction_capture_hook(l26_sae, the_pos, clean_store)
            )
        )
        with torch.inference_mode():
            clean_out = model(**inputs)
        for h in clean_handles:
            h.remove()

        pred = model_predict_from_logits(clean_out.logits[0, -1].detach().float().cpu())
        a_clean = clean_store["direction_feature_acts"][pred]

        # 2) geographic ablation only: total effect T
        geo_store = {}
        geo_handles = [
            model.model.layers[L8_LAYER].register_forward_hook(
                make_l8_geo_hook(l8_sae, [ca_pos, cb_pos], geo_fids)
            ),
            model.model.layers[L26_LAYER].register_forward_hook(
                make_l26_direction_capture_hook(l26_sae, the_pos, geo_store)
            ),
        ]
        with torch.inference_mode():
            _ = model(**inputs)
        for h in geo_handles:
            h.remove()
        a_geo = geo_store["direction_feature_acts"][pred]
        delta_total = a_geo - a_clean

        # 3) cumulative freeze: k=1 freezes L26; k=2 freezes L25..L26; ...
        D = {}
        for k in range(1, n_layers + 1):
            frozen_layers = SCAN_LAYERS[-k:]  # the top k layers
            freeze_store = {}
            freeze_handles = [
                model.model.layers[L8_LAYER].register_forward_hook(
                    make_l8_geo_hook(l8_sae, [ca_pos, cb_pos], geo_fids)
                )
            ]
            for L in frozen_layers:
                freeze_handles.append(
                    model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
                        make_freeze_attn_hook(the_pos, clean_store[f"clean_attn_{L}"])
                    )
                )
            freeze_handles.append(
                model.model.layers[L26_LAYER].register_forward_hook(
                    make_l26_direction_capture_hook(l26_sae, the_pos, freeze_store)
                )
            )
            with torch.inference_mode():
                _ = model(**inputs)
            for h in freeze_handles:
                h.remove()
            a_freeze = freeze_store["direction_feature_acts"][pred]
            D[k] = a_freeze - a_clean

        records.append({
            "index": idx,
            "pred_dir_clean": pred,
            "delta_total": delta_total,
            "D": D,  # {k: delta_direct}
        })

    del model, l8_sae, l26_sae
    gc.collect()
    torch.cuda.empty_cache()

    # summary
    dt = np.array([r["delta_total"] for r in records], dtype=np.float64)
    Dk = {
        k: np.array([r["D"][k] for r in records], dtype=np.float64)
        for k in range(1, n_layers + 1)
    }

    summary = {
        "n_prompts": len(records),
        "median_delta_total": float(np.median(dt)),
        "cumulative_freeze": {},
    }
    # per-layer contribution = D(k-1) - D(k), with D(0) = T
    prev = dt
    for k in range(1, n_layers + 1):
        layer_name = f"L{SCAN_LAYERS[-k]}" if k == 1 else f"L{SCAN_LAYERS[-k]}-L26"
        dk = Dk[k]
        contribution = prev - dk  # extra effect removed by freezing this layer
        summary["cumulative_freeze"][str(k)] = {
            "frozen_layers": [int(L) for L in SCAN_LAYERS[-k:]],
            "median_D": float(np.median(dk)),
            "median_contribution_of_newest_layer": float(np.median(contribution)),
        }
        prev = dk

    result = {
        "experiment": "l8_geographic_bypass_layerscan",
        "model_name": MODEL_NAME,
        "geographic_features": [int(x) for x in geo_fids],
        "summary": summary,
        "per_prompt": records,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary] remaining effect D on the direction feature after cumulative freeze (median)")
    print(f"  total effect T (no freeze)                : {summary['median_delta_total']:+.3f}")
    for k in range(1, n_layers + 1):
        s = summary["cumulative_freeze"][str(k)]
        newest = s["frozen_layers"][0]
        print(f"  k={k:>2} freeze L{newest}..L26   : D={s['median_D']:+.3f}  "
              f"(this layer's contribution {s['median_contribution_of_newest_layer']:+.3f})")
    print(f"[Done] results saved to: {OUT_PATH}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="L8 geographic bypass layer scan")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
