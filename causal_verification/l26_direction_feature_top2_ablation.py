#!/usr/bin/env python3
"""
Ablate the top-1 vs top-2 L26 direction features, jointly and separately.

For prompts on which the model predicts a given direction, zero that direction's
cDLA top-1 feature, its top-2 feature, and both, then measure the drop in the
predicted-direction logit (relative to an SAE reconstruction baseline) and the
flip rate. This checks whether the top-1 cDLA feature is the dominant causal
feature in every direction (it is not, for East).
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
from scipy.stats import wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
OUT_PATH = ROOT / "causal_verification" / "results" / "l26_direction_feature_top2_ablation.json"

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# cDLA top-1 / top-2 per direction (from cdla_rank_stability.json)
DIRECTION_FEATURES = {
    "East":  {"top1": 47636, "top2": 18707},
    "West":  {"top1": 56725, "top2": 38915},
    "South": {"top1": 10801, "top2": 38915},
    "North": {"top1": 26800, "top2": 18707},
}


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["prompt"] = PROMPT_TPL.format(
                city_a=item["City_A"], city_b=item["City_B"]
            )
            rows.append(item)
    return rows


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID=279 not found")
    return int(positions[-1].item())


def direction_logits(logits: torch.Tensor):
    return {d: float(logits[DIR_TOKEN_IDS[d]].item()) for d in DIR_LABELS}


def model_predict_from_logits(logits: torch.Tensor) -> str:
    s_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    s_ns = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


def make_l26_hook(sae: SAE, the_pos: int, ablate_fids):
    fid_tensor = None
    if ablate_fids is not None:
        fid_tensor = torch.tensor(ablate_fids, dtype=torch.long, device="cuda")

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None

        hidden_for_sae = hidden.clone()
        hidden_for_sae[:, :, OUTLIER_DIMS] = 0.0
        vec = hidden_for_sae[:, the_pos, :].to(device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            feats = sae.encode(vec)
            if fid_tensor is not None:
                feats[:, fid_tensor] = 0.0
            decoded = sae.decode(feats)

        hidden[:, the_pos, :] = decoded.to(dtype=hidden.dtype)
        if rest is not None:
            return (hidden, *rest)
        return hidden

    return hook


def run_clean(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def run_replacement(model, tokenizer, sae, prompt: str, ablate_fids) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    the_pos = get_the_pos(inputs.input_ids)
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(
        make_l26_hook(sae, the_pos, ablate_fids)
    )
    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def load_model_and_sae():
    print("[Model] loading tokenizer/model/L26 SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    sae = SAE.load_from_disk(str(SAE_PATH), device="cuda")
    sae.eval()
    return model, tokenizer, sae


def run_experiment(args):
    rows = load_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    model, tokenizer, sae = load_model_and_sae()

    records = {d: [] for d in DIR_LABELS}
    for idx, row in enumerate(tqdm(rows, desc="[L26 top1/top2 ablation]", unit="prompt")):
        prompt = row["prompt"]
        clean = run_clean(model, tokenizer, prompt)
        pred = model_predict_from_logits(clean)

        feats = DIRECTION_FEATURES[pred]
        top1, top2 = feats["top1"], feats["top2"]
        pred_id = DIR_TOKEN_IDS[pred]

        recon = run_replacement(model, tokenizer, sae, prompt, None)
        single_top1 = run_replacement(model, tokenizer, sae, prompt, [top1])
        single_top2 = run_replacement(model, tokenizer, sae, prompt, [top2])
        joint = run_replacement(model, tokenizer, sae, prompt, [top1, top2])

        recon_logit = float(recon[pred_id].item())
        records[pred].append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "pred_dir_clean": pred,
            "clean_direction_logits": direction_logits(clean),
            "recon_direction_logits": direction_logits(recon),
            "top1_direction_logits": direction_logits(single_top1),
            "top2_direction_logits": direction_logits(single_top2),
            "joint_direction_logits": direction_logits(joint),
            "delta_top1_from_recon": float(single_top1[pred_id].item()) - recon_logit,
            "delta_top2_from_recon": float(single_top2[pred_id].item()) - recon_logit,
            "delta_joint_from_recon": float(joint[pred_id].item()) - recon_logit,
            "flip_top1": model_predict_from_logits(single_top1) != pred,
            "flip_top2": model_predict_from_logits(single_top2) != pred,
            "flip_joint": model_predict_from_logits(joint) != pred,
        })

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary = {}
    for d in DIR_LABELS:
        recs = records[d]
        if not recs:
            summary[d] = {"n_prompts": 0, "note": "no prompts predicted as this direction"}
            continue
        n = len(recs)
        d1 = np.array([r["delta_top1_from_recon"] for r in recs], dtype=np.float64)
        d2 = np.array([r["delta_top2_from_recon"] for r in recs], dtype=np.float64)
        dj = np.array([r["delta_joint_from_recon"] for r in recs], dtype=np.float64)

        try:
            stat12, p12 = wilcoxon(d1, d2, zero_method="wilcox")  # two-sided
            stat12, p12 = float(stat12), float(p12)
        except Exception:
            stat12, p12 = None, None

        summary[d] = {
            "n_prompts": n,
            "top1_feature": DIRECTION_FEATURES[d]["top1"],
            "top2_feature": DIRECTION_FEATURES[d]["top2"],
            "median_delta_top1_from_recon": float(np.median(d1)),
            "median_delta_top2_from_recon": float(np.median(d2)),
            "median_delta_joint_from_recon": float(np.median(dj)),
            "mean_delta_top1_from_recon": float(np.mean(d1)),
            "mean_delta_top2_from_recon": float(np.mean(d2)),
            "mean_delta_joint_from_recon": float(np.mean(dj)),
            "flip_rate_top1": float(np.mean([r["flip_top1"] for r in recs])),
            "flip_rate_top2": float(np.mean([r["flip_top2"] for r in recs])),
            "flip_rate_joint": float(np.mean([r["flip_joint"] for r in recs])),
            "n_top2_more_negative_than_top1": int(np.sum(d2 < d1)),
            "n_joint_more_negative_than_top1": int(np.sum(dj < d1)),
            "n_joint_more_negative_than_top2": int(np.sum(dj < d2)),
            "wilcoxon_top1_vs_top2": {"stat": stat12, "p": p12},
        }

    result = {
        "experiment": "l26_direction_feature_top2_ablation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "direction_features": {d: DIRECTION_FEATURES[d] for d in DIR_LABELS},
        "summary": summary,
        "per_prompt": records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary] median delta logit (vs recon) for top1 / top2 / joint per direction")
    print(f"  {'dir':>6} {'n':>4} {'top1':>8} {'top2':>8} {'joint':>8} "
          f"{'flip1':>7} {'flip2':>7} {'flipJ':>7} {'p(top1 vs top2)':>16}")
    for d in DIR_LABELS:
        s = summary[d]
        if s.get("n_prompts", 0) == 0:
            print(f"  {d:>6} {'0':>4}  (no prompts predicted as this direction)")
            continue
        p12 = s["wilcoxon_top1_vs_top2"]["p"]
        p12s = f"{p12:.2e}" if p12 is not None else "None"
        print(f"  {d:>6} {s['n_prompts']:>4} "
              f"{s['median_delta_top1_from_recon']:>+8.3f} "
              f"{s['median_delta_top2_from_recon']:>+8.3f} "
              f"{s['median_delta_joint_from_recon']:>+8.3f} "
              f"{s['flip_rate_top1']*100:>6.1f}% "
              f"{s['flip_rate_top2']*100:>6.1f}% "
              f"{s['flip_rate_joint']*100:>6.1f}% "
              f"{p12s:>16}")
    print(f"[Done] results saved to: {OUT_PATH}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="L26 four-direction top1/top2 feature ablation")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
