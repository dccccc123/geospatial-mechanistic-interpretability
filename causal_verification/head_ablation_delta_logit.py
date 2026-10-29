#!/usr/bin/env python3
"""
Head ablation with predicted direction delta logit.

This script updates the previous head ablation experiment to match the feature
ablation metrics used in the paper:

  Delta logit = logit_after(d_pred) - logit_before(d_pred)

where d_pred is the model's clean prediction among the four direction tokens.

Targets:
  - Layer 26 Head 24
  - Layer 26 Head 25

Controls:
  - The other 30 heads in the same layer.

Run:
  /opt/miniconda3/envs/pytorch_env/bin/python \
      causal_verification/head_ablation_delta_logit.py
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "causal_verification" / "results"


def resolve_paths(dataset: str):
    """Return (input jsonl, output json) based on --dataset."""
    if dataset == "holdout":
        return (
            ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl",
            RESULT_DIR / "head_ablation_delta_logit_holdout.json",
        )
    return (
        ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl",
        RESULT_DIR / "head_ablation_delta_logit.json",
    )

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
N_HEADS = 32
D_MODEL = 4096
HEAD_DIM = D_MODEL // N_HEADS
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
TARGET_HEADS = [24, 25]


def load_rows(data_path: Path) -> List[Dict]:
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["prompt"] = PROMPT_TPL.format(
                city_a=item["City_A"], city_b=item["City_B"]
            )
            rows.append(item)
    return rows


def direction_logits(logits: torch.Tensor) -> Dict[str, float]:
    return {d: float(logits[DIR_TOKEN_IDS[d]].item()) for d in DIR_LABELS}


def model_predict_from_logits(logits: torch.Tensor) -> str:
    """Axis decision rule: subtract within-axis, then compare the two axis scores."""
    score_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    score_sn = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(score_ew) >= abs(score_sn):
        return "East" if score_ew > 0 else "West"
    return "South" if score_sn > 0 else "North"


def make_head_ablation_hook(head_idx: int):
    start = head_idx * HEAD_DIM
    end = start + HEAD_DIM

    def pre_hook(_module, inputs):
        inputs = list(inputs)
        hidden = inputs[0].clone()
        hidden[:, :, start:end] = 0.0
        inputs[0] = hidden
        return tuple(inputs)

    return pre_hook


def run_clean(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def run_head_ablation(model, tokenizer, prompt: str, head_idx: int) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    handle = model.model.layers[TARGET_LAYER].self_attn.o_proj.register_forward_pre_hook(
        make_head_ablation_hook(head_idx)
    )
    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def load_model():
    print("[Model] loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[Model] done, device={next(model.parameters()).device}")
    return model, tokenizer


def summarize_head(head_idx: int, deltas_by_head: np.ndarray, control_heads: List[int]) -> Dict:
    target = deltas_by_head[:, head_idx]
    controls = deltas_by_head[:, control_heads]
    control_mean = controls.mean(axis=1)
    control_flat = controls.reshape(-1)

    try:
        stat_w, p_w = wilcoxon(target, control_mean, alternative="less", zero_method="wilcox")
        stat_w, p_w = float(stat_w), float(p_w)
    except Exception:
        stat_w, p_w = None, None

    stat_u, p_u = mannwhitneyu(target, control_flat, alternative="less")

    return {
        "head_idx": int(head_idx),
        "n_prompts": int(deltas_by_head.shape[0]),
        "median_delta": float(np.median(target)),
        "mean_delta": float(np.mean(target)),
        "negative_delta_rate": float(np.mean(target < 0)),
        "control_heads": [int(h) for h in control_heads],
        "median_control_delta": float(np.median(control_flat)),
        "mean_control_delta": float(np.mean(control_flat)),
        "control_negative_delta_rate": float(np.mean(control_flat < 0)),
        "median_target_minus_control_mean": float(np.median(target - control_mean)),
        "mean_target_minus_control_mean": float(np.mean(target - control_mean)),
        "paired_wilcoxon_stat_target_less_than_control_mean": stat_w,
        "paired_wilcoxon_p_target_less_than_control_mean": p_w,
        "mannwhitney_stat_target_less_than_controls": float(stat_u),
        "mannwhitney_p_target_less_than_controls": float(p_u),
        "delta_scores": [float(x) for x in target.tolist()],
    }


def run_experiment(args) -> Dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_path, out_path = resolve_paths(args.dataset)
    rows = load_rows(data_path)
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    model, tokenizer = load_model()

    per_prompt = []
    deltas_by_head = []

    for idx, row in enumerate(tqdm(rows, desc="[head ablation delta logit]", unit="prompt")):
        clean_logits = run_clean(model, tokenizer, row["prompt"])
        pred_dir = model_predict_from_logits(clean_logits)
        pred_id = DIR_TOKEN_IDS[pred_dir]
        clean_pred_logit = float(clean_logits[pred_id].item())

        head_deltas = []
        head_logits = {}
        for head_idx in range(N_HEADS):
            ablated_logits = run_head_ablation(model, tokenizer, row["prompt"], head_idx)
            after_logit = float(ablated_logits[pred_id].item())
            delta = after_logit - clean_pred_logit
            head_deltas.append(delta)
            if head_idx in TARGET_HEADS:
                head_logits[str(head_idx)] = {
                    "logit_pred_after": after_logit,
                    "delta_logit": float(delta),
                    "direction_logits_after": direction_logits(ablated_logits),
                }

        deltas_by_head.append(head_deltas)
        per_prompt.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "prompt": row["prompt"],
            "pred_dir_clean": pred_dir,
            "logit_pred_clean": clean_pred_logit,
            "clean_direction_logits": direction_logits(clean_logits),
            "target_head_results": head_logits,
        })

    del model
    gc.collect()
    torch.cuda.empty_cache()

    deltas_arr = np.array(deltas_by_head, dtype=np.float64)
    control_heads = [h for h in range(N_HEADS) if h not in TARGET_HEADS]
    target_summaries = {
        str(h): summarize_head(h, deltas_arr, control_heads)
        for h in TARGET_HEADS
    }

    control_flat = deltas_arr[:, control_heads].reshape(-1)
    result = {
        "experiment": "head_ablation_delta_logit",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "n_heads": N_HEADS,
        "target_heads": TARGET_HEADS,
        "control_heads": control_heads,
        "n_prompts": len(rows),
        "metric": "delta_logit = logit_after(d_pred) - logit_before(d_pred)",
        "prediction_rule": "d_pred is the max logit among East, West, South, North in clean forward",
        "control_summary": {
            "median_delta": float(np.median(control_flat)),
            "mean_delta": float(np.mean(control_flat)),
            "negative_delta_rate": float(np.mean(control_flat < 0)),
        },
        "target_results": target_summaries,
        "per_prompt": per_prompt,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    print(f"  control median Δlogit: {result['control_summary']['median_delta']:+.4f}")
    for h in TARGET_HEADS:
        s = target_summaries[str(h)]
        print(
            f"  Head {h} median Δlogit: {s['median_delta']:+.4f} "
            f"vs control {s['median_control_delta']:+.4f}; "
            f"Wilcoxon p={s['paired_wilcoxon_p_target_less_than_control_mean']}"
        )
    print(f"[Done] results saved to: {out_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Head ablation using predicted direction delta logit")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "holdout"],
        help="train=balanced_city_pairs.jsonl; holdout=holdout_city_pairs.jsonl (city-disjoint holdout set).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
