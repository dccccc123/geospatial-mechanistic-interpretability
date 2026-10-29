#!/usr/bin/env python3
"""
Ablate the two East direction features (#47636 and #18707) separately and jointly.

For prompts predicted as East, zeroes #47636 alone, #18707 alone, and both, and
compares the East-logit drop (relative to an SAE reconstruction baseline) and the
flip rate. This checks whether the unstable top-1 feature #47636 hides a causal
contribution from #18707.
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
SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
OUT_PATH = ROOT / "causal_verification" / "results" / "l26_east_joint_ablation.json"

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# The two East direction features: top-1 #47636 and secondary #18707.
EAST_FEATURES = [47636, 18707]


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
    east_id = DIR_TOKEN_IDS["East"]

    records = []
    for idx, row in enumerate(tqdm(rows, desc="[L26 East joint ablation]", unit="prompt")):
        prompt = row["prompt"]
        clean = run_clean(model, tokenizer, prompt)
        pred = model_predict_from_logits(clean)
        if pred != "East":
            continue

        recon = run_replacement(model, tokenizer, sae, prompt, None)
        single_47636 = run_replacement(model, tokenizer, sae, prompt, [47636])
        single_18707 = run_replacement(model, tokenizer, sae, prompt, [18707])
        joint = run_replacement(model, tokenizer, sae, prompt, [47636, 18707])

        recon_east = float(recon[east_id].item())
        records.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "pred_dir_clean": pred,
            "clean_direction_logits": direction_logits(clean),
            "recon_direction_logits": direction_logits(recon),
            "single_47636_direction_logits": direction_logits(single_47636),
            "single_18707_direction_logits": direction_logits(single_18707),
            "joint_direction_logits": direction_logits(joint),
            "east_logit_clean": float(clean[east_id].item()),
            "east_logit_recon": recon_east,
            "delta_east_47636_from_recon": float(single_47636[east_id].item()) - recon_east,
            "delta_east_18707_from_recon": float(single_18707[east_id].item()) - recon_east,
            "delta_east_joint_from_recon": float(joint[east_id].item()) - recon_east,
            "flip_47636": model_predict_from_logits(single_47636) != "East",
            "flip_18707": model_predict_from_logits(single_18707) != "East",
            "flip_joint": model_predict_from_logits(joint) != "East",
        })

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    n = len(records)
    if n == 0:
        print("[Warning] no prompts predicted as East; cannot run the joint ablation.")
        return None

    d1 = np.array([r["delta_east_47636_from_recon"] for r in records], dtype=np.float64)
    d2 = np.array([r["delta_east_18707_from_recon"] for r in records], dtype=np.float64)
    dj = np.array([r["delta_east_joint_from_recon"] for r in records], dtype=np.float64)

    summary = {
        "n_east_prompts": n,
        "east_features": EAST_FEATURES,
        "median_delta_east_47636_from_recon": float(np.median(d1)),
        "median_delta_east_18707_from_recon": float(np.median(d2)),
        "median_delta_east_joint_from_recon": float(np.median(dj)),
        "mean_delta_east_47636_from_recon": float(np.mean(d1)),
        "mean_delta_east_18707_from_recon": float(np.mean(d2)),
        "mean_delta_east_joint_from_recon": float(np.mean(dj)),
        "flip_rate_47636": float(np.mean([r["flip_47636"] for r in records])),
        "flip_rate_18707": float(np.mean([r["flip_18707"] for r in records])),
        "flip_rate_joint": float(np.mean([r["flip_joint"] for r in records])),
        "n_joint_more_negative_than_47636": int(np.sum(dj < d1)),
        "n_joint_more_negative_than_18707": int(np.sum(dj < d2)),
    }

    result = {
        "experiment": "l26_east_joint_ablation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "east_features": EAST_FEATURES,
        "summary": summary,
        "per_prompt": records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary] East prompts =", n)
    print(f"  #47636 alone   median ΔEast-logit: {summary['median_delta_east_47636_from_recon']:+.3f}  flip={summary['flip_rate_47636']*100:.1f}%")
    print(f"  #18707 alone   median ΔEast-logit: {summary['median_delta_east_18707_from_recon']:+.3f}  flip={summary['flip_rate_18707']*100:.1f}%")
    print(f"  joint ablation       median ΔEast-logit: {summary['median_delta_east_joint_from_recon']:+.3f}  flip={summary['flip_rate_joint']*100:.1f}%")
    print(f"  samples where joint is more negative than #47636 alone: {summary['n_joint_more_negative_than_47636']}/{n}")
    print(f"  samples where joint is more negative than #18707 alone: {summary['n_joint_more_negative_than_18707']}/{n}")
    print(f"[Done] results saved to: {OUT_PATH}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="L26 East joint feature ablation")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
