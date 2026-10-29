#!/usr/bin/env python3
"""
Head ablation effect on L26 direction feature activation.

This experiment tests whether ablating the routing heads weakens the L26 SAE
direction feature associated with the model's original prediction.

For each task prompt:
  1. Run the clean prompt.
  2. Select d_pred as the largest logit among East, West, South, and North.
  3. Read the L26 "the" token SAE activation of the direction feature f(d_pred).
  4. Rerun the prompt while ablating each L26 attention head.
  5. Measure delta activation:

       delta_act = act_after(f(d_pred)) - act_clean(f(d_pred))

Targets:
  - Layer 26 Head 24
  - Layer 26 Head 25

Controls:
  - The other 30 heads in the same layer.

Run:
  /opt/miniconda3/envs/pytorch_env/bin/python \
      circuit_tracing/head_to_direction_feature_activation.py
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import mannwhitneyu, wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE


ROOT = Path(__file__).resolve().parent.parent
SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
RESULT_DIR = ROOT / "circuit_tracing" / "results"


def resolve_paths(dataset: str):
    """Return (input jsonl, output json) based on --dataset."""
    if dataset == "holdout":
        return (
            ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl",
            RESULT_DIR / "head_to_direction_feature_activation_holdout.json",
        )
    return (
        ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl",
        RESULT_DIR / "head_to_direction_feature_activation.json",
    )

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
N_HEADS = 32
D_MODEL = 4096
HEAD_DIM = D_MODEL // N_HEADS
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
DIRECTION_FEATURES = {
    "East": 47636,
    "West": 56725,
    "South": 10801,
    "North": 26800,
}
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


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID=279 not found; check prompt/tokenizer.")
    return int(positions[-1].item())


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


def make_l26_feature_capture_hook(sae: SAE, the_pos: int, store: Dict):
    direction_fids = [DIRECTION_FEATURES[d] for d in DIR_LABELS]

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
        store["direction_feature_ids"] = {
            d: int(DIRECTION_FEATURES[d])
            for d in DIR_LABELS
        }
        store["all_direction_feature_acts"] = [
            float(feat[0, fid].detach().float().cpu().item())
            for fid in direction_fids
        ]

    return hook


def run_forward_capture(
    model,
    tokenizer,
    sae: SAE,
    prompt: str,
    head_idx: Optional[int] = None,
) -> Dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    the_pos = get_the_pos(inputs.input_ids)
    capture: Dict = {}
    handles = []
    handles.append(
        model.model.layers[TARGET_LAYER].register_forward_hook(
            make_l26_feature_capture_hook(sae, the_pos, capture)
        )
    )
    if head_idx is not None:
        handles.append(
            model.model.layers[TARGET_LAYER].self_attn.o_proj.register_forward_pre_hook(
                make_head_ablation_hook(head_idx)
            )
        )

    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    if "direction_feature_acts" not in capture:
        raise RuntimeError("Failed to capture L26 SAE direction feature activations.")

    logits = out.logits[0, -1].detach().float().cpu()
    return {
        "logits": logits,
        "direction_logits": direction_logits(logits),
        "direction_feature_acts": capture["direction_feature_acts"],
        "direction_feature_ids": capture["direction_feature_ids"],
    }


def load_model_and_sae():
    print("[Model] loading tokenizer/model/L26 SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    sae = SAE.load_from_disk(str(SAE_PATH), device="cuda")
    sae.eval()
    print(f"[Model] done, device={next(model.parameters()).device}")
    return model, tokenizer, sae


def summarize_head(head_idx: int, deltas_by_head: np.ndarray, control_heads: List[int]) -> Dict:
    target = deltas_by_head[:, head_idx]
    controls = deltas_by_head[:, control_heads]
    control_mean = controls.mean(axis=1)
    control_flat = controls.reshape(-1)

    try:
        stat_w, p_w = wilcoxon(
            target,
            control_mean,
            alternative="less",
            zero_method="wilcox",
        )
        stat_w, p_w = float(stat_w), float(p_w)
    except Exception:
        stat_w, p_w = None, None

    stat_u, p_u = mannwhitneyu(target, control_flat, alternative="less")

    return {
        "head_idx": int(head_idx),
        "n_prompts": int(deltas_by_head.shape[0]),
        "median_delta_activation": float(np.median(target)),
        "mean_delta_activation": float(np.mean(target)),
        "negative_delta_activation_rate": float(np.mean(target < 0)),
        "control_heads": [int(h) for h in control_heads],
        "median_control_delta_activation": float(np.median(control_flat)),
        "mean_control_delta_activation": float(np.mean(control_flat)),
        "control_negative_delta_activation_rate": float(np.mean(control_flat < 0)),
        "median_target_minus_control_mean": float(np.median(target - control_mean)),
        "mean_target_minus_control_mean": float(np.mean(target - control_mean)),
        "paired_wilcoxon_stat_target_less_than_control_mean": stat_w,
        "paired_wilcoxon_p_target_less_than_control_mean": p_w,
        "mannwhitney_stat_target_less_than_controls": float(stat_u),
        "mannwhitney_p_target_less_than_controls": float(p_u),
        "delta_activation_scores": [float(x) for x in target.tolist()],
    }


def summarize_by_predicted_direction(per_prompt: List[Dict], target_head: int) -> Dict:
    out = {}
    for direction in DIR_LABELS:
        rows = [x for x in per_prompt if x["pred_dir_clean"] == direction]
        if not rows:
            continue
        deltas = np.array(
            [x["head_delta_activation"][str(target_head)] for x in rows],
            dtype=np.float64,
        )
        out[direction] = {
            "n": len(rows),
            "direction_feature": int(DIRECTION_FEATURES[direction]),
            "median_delta_activation": float(np.median(deltas)),
            "mean_delta_activation": float(np.mean(deltas)),
            "negative_delta_activation_rate": float(np.mean(deltas < 0)),
        }
    return out


def run_experiment(args) -> Dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_path, out_path = resolve_paths(args.dataset)
    rows = load_rows(data_path)
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    model, tokenizer, sae = load_model_and_sae()

    per_prompt = []
    deltas_by_head = []

    for idx, row in enumerate(
        tqdm(rows, desc="[head -> L26 direction feature]", unit="prompt")
    ):
        clean = run_forward_capture(model, tokenizer, sae, row["prompt"], head_idx=None)
        pred_dir = model_predict_from_logits(clean["logits"])
        pred_feature = DIRECTION_FEATURES[pred_dir]
        clean_act = clean["direction_feature_acts"][pred_dir]

        head_deltas = []
        target_head_records = {}
        for head_idx in range(N_HEADS):
            ablated = run_forward_capture(
                model,
                tokenizer,
                sae,
                row["prompt"],
                head_idx=head_idx,
            )
            after_act = ablated["direction_feature_acts"][pred_dir]
            delta = after_act - clean_act
            head_deltas.append(delta)
            if head_idx in TARGET_HEADS:
                target_head_records[str(head_idx)] = {
                    "activation_after": float(after_act),
                    "delta_activation": float(delta),
                    "direction_feature_acts_after": ablated["direction_feature_acts"],
                    "direction_logits_after": ablated["direction_logits"],
                }

        deltas_by_head.append(head_deltas)
        per_prompt.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "prompt": row["prompt"],
            "pred_dir_clean": pred_dir,
            "pred_direction_feature": int(pred_feature),
            "clean_direction_logits": clean["direction_logits"],
            "clean_direction_feature_acts": clean["direction_feature_acts"],
            "clean_pred_direction_feature_activation": float(clean_act),
            "target_head_results": target_head_records,
            "head_delta_activation": {
                str(head_idx): float(delta)
                for head_idx, delta in enumerate(head_deltas)
            },
        })

    del model, sae
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
        "experiment": "head_to_direction_feature_activation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "n_heads": N_HEADS,
        "target_heads": TARGET_HEADS,
        "control_heads": control_heads,
        "n_prompts": len(rows),
        "direction_features_top1_cdla": DIRECTION_FEATURES,
        "metric": (
            "delta_activation = activation_after(f(d_pred)) "
            "- activation_clean(f(d_pred))"
        ),
        "prediction_rule": (
            "d_pred is the max logit among East, West, South, North in clean forward"
        ),
        "control_summary": {
            "median_delta_activation": float(np.median(control_flat)),
            "mean_delta_activation": float(np.mean(control_flat)),
            "negative_delta_activation_rate": float(np.mean(control_flat < 0)),
        },
        "target_results": target_summaries,
        "by_predicted_direction": {
            str(h): summarize_by_predicted_direction(per_prompt, h)
            for h in TARGET_HEADS
        },
        "per_prompt": per_prompt,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    print(
        "  control median Δactivation: "
        f"{result['control_summary']['median_delta_activation']:+.4f}"
    )
    for h in TARGET_HEADS:
        s = target_summaries[str(h)]
        print(
            f"  Head {h} median Δactivation: "
            f"{s['median_delta_activation']:+.4f} "
            f"vs control {s['median_control_delta_activation']:+.4f}; "
            f"Wilcoxon p={s['paired_wilcoxon_p_target_less_than_control_mean']}"
        )
    print(f"[Done] results saved to: {out_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head ablation effect on L26 direction feature activation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "holdout"],
        help="train=balanced_city_pairs.jsonl; holdout=holdout_city_pairs.jsonl (city-disjoint validation set).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N prompts for smoke test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
