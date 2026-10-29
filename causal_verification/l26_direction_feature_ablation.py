#!/usr/bin/env python3
"""
L26 direction-feature ablation.

This script causally tests the four top-1 L26 cDLA direction features:
    East  -> #47636
    West  -> #56725
    South -> #10801
    North -> #26800

Main analysis:
  1. Run the clean prompt and obtain the model-predicted direction.
  2. At the L26 the token, encode the block output with the L26 SAE.
  3. Zero the SAE feature corresponding to the model-predicted direction.
  4. Decode, replace the L26 the residual, and continue the forward pass.
  5. Compare the predicted-direction logit drop against matched control
     features with similar activation statistics but low cDLA selectivity.

The experiment is deliberately model-prediction based, not GT based:
if the model predicts West on an East-labeled prompt, we ablate the West
feature because the question is whether the feature participates in the
model's actual output computation.

Run:
    /opt/miniconda3/envs/pytorch_env/bin/python \
        causal_verification/l26_direction_feature_ablation.py
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE


# ─────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
RESULT_DIR = ROOT / "causal_verification" / "results"


def resolve_paths(dataset: str):
    """Return (input jsonl, L26 activation cache npy, output json) based on --dataset.

    train keeps the original paths; holdout uses *_holdout suffixes to avoid
    overwriting training-set results.
    """
    if dataset == "holdout":
        return (
            ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl",
            ROOT / "sae" / "activations" / "the_sae_acts_l26_holdout.npy",
            RESULT_DIR / "l26_direction_feature_ablation_holdout.json",
        )
    return (
        ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl",
        ROOT / "sae" / "activations" / "the_sae_acts_l26.npy",
        RESULT_DIR / "l26_direction_feature_ablation.json",
    )

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# Stage-2 top-1 cDLA direction features. Keep these fixed for paper consistency.
DIRECTION_FEATURES = {
    "East": 47636,
    "West": 56725,
    "South": 10801,
    "North": 26800,
}


# ─────────────────────────────────────────────────────────────
# Data/model helpers
# ─────────────────────────────────────────────────────────────
def load_rows(data_path: Path) -> List[Dict]:
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item["prompt"] = PROMPT_TPL.format(
                city_a=item["City_A"],
                city_b=item["City_B"],
            )
            rows.append(item)
    return rows


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID=279 not found; check the prompt/tokenizer.")
    return int(positions[-1].item())


def direction_logits(logits: torch.Tensor) -> Dict[str, float]:
    return {d: float(logits[DIR_TOKEN_IDS[d]].item()) for d in DIR_LABELS}


def model_predict_from_logits(logits: torch.Tensor) -> str:
    score_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    score_sn = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(score_ew) >= abs(score_sn):
        return "East" if score_ew > 0 else "West"
    return "South" if score_sn > 0 else "North"


def load_model_and_sae():
    print("[Model] loading tokenizer/model/SAE...")
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
    print(f"[Model] done, model device={next(model.parameters()).device}")
    return model, tokenizer, sae


def load_w_dec_cpu() -> torch.Tensor:
    raw = load_file(str(SAE_PATH / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec"))
    if w_dec is None:
        raise KeyError("W_dec / sae.W_dec not found in sae_weights.safetensors")
    return w_dec.float().cpu()


def compute_or_load_the_sae_acts(model, tokenizer, sae, rows, act_path):
    """Cache L26 the-position SAE activations [N, d_sae], aligned with rows.

    Loads the existing the_sae_acts_l26.npy for train; computes and saves for holdout.
    """
    if act_path.exists():
        acts = np.load(str(act_path))
        if acts.shape[0] >= len(rows):
            print(f"[Cache] loading L26 the SAE activations: {act_path}, shape={acts.shape}")
            return acts
        print(f"[Cache] old L26 the activation cache shape={acts.shape} < rows={len(rows)}, recomputing.")

    all_acts = []
    print("[Cache] computing L26 the SAE activations for cDLA/matched controls...")
    for row in tqdm(rows, desc="[extract L26 the SAE acts]", unit="prompt"):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)
        captured = {}

        def hook(_module, _input, output):
            hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
            hidden[:, :, OUTLIER_DIMS] = 0.0
            captured["h_the"] = hidden[:, the_pos, :].detach()
            return None

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                _ = model(**inputs)
        finally:
            handle.remove()

        with torch.inference_mode():
            feats = sae.encode(captured["h_the"].to(device="cuda", dtype=torch.bfloat16))
        all_acts.append(feats.squeeze(0).detach().float().cpu().numpy())

    acts = np.stack(all_acts, axis=0).astype(np.float32)
    act_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(act_path), acts)
    print(f"[Cache] saved L26 the SAE activations: {act_path}, shape={acts.shape}")
    return acts


# ─────────────────────────────────────────────────────────────
# cDLA and matched controls
# ─────────────────────────────────────────────────────────────
def compute_cdla_matrix(
    acts: np.ndarray,
    rows: List[Dict],
    w_dec: torch.Tensor,
    dir_unembeds: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Return cDLA matrix [4, d_sae] in DIR_LABELS order."""
    print("[Control] computing L26 full-feature cDLA/selectivity for matched controls...")
    directions = [r["Direction"] for r in rows]
    dir_idx = {d: [i for i, x in enumerate(directions) if x == d] for d in DIR_LABELS}

    proj = {
        d: (w_dec @ dir_unembeds[d]).numpy()
        for d in DIR_LABELS
    }
    means = {
        d: acts[dir_idx[d]].mean(axis=0)
        for d in DIR_LABELS
    }
    dla = {
        d: means[d] * proj[d]
        for d in DIR_LABELS
    }

    cdla = []
    for d in DIR_LABELS:
        others = np.stack([dla[o] for o in DIR_LABELS if o != d], axis=0).mean(axis=0)
        cdla.append(dla[d] - others)
    return np.stack(cdla, axis=0).astype(np.float32)


def feature_metrics(acts: np.ndarray, w_dec: torch.Tensor, cdla: np.ndarray) -> Dict[str, np.ndarray]:
    mean_act = acts.mean(axis=0).astype(np.float32)
    firing_freq = (acts > 0).mean(axis=0).astype(np.float32)
    decoder_norm = torch.linalg.norm(w_dec, dim=1).numpy().astype(np.float32)

    sorted_cdla = np.sort(cdla, axis=0)
    selectivity = (sorted_cdla[-1] - sorted_cdla[-2]).astype(np.float32)
    max_cdla = sorted_cdla[-1].astype(np.float32)

    return {
        "mean_act": mean_act,
        "firing_freq": firing_freq,
        "decoder_norm": decoder_norm,
        "selectivity": selectivity,
        "max_cdla": max_cdla,
    }


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def build_matched_controls(
    acts: np.ndarray,
    w_dec: torch.Tensor,
    cdla: np.ndarray,
    n_controls: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict]]:
    """
    Pick nearest low-selectivity controls for each fixed direction feature.

    Matching dimensions:
      - log1p(mean activation)
      - firing frequency
      - log decoder norm

    Low selectivity is enforced by keeping only features below the median
    direction-selectivity score.
    """
    metrics = feature_metrics(acts, w_dec, cdla)
    target_ids = set(DIRECTION_FEATURES.values())
    selectivity_cutoff = float(np.median(metrics["selectivity"]))

    match_space = np.stack([
        standardize(np.log1p(metrics["mean_act"])),
        standardize(metrics["firing_freq"]),
        standardize(np.log(metrics["decoder_norm"] + 1e-12)),
    ], axis=1)

    low_sel = metrics["selectivity"] <= selectivity_cutoff
    active = metrics["firing_freq"] > 0
    finite = np.isfinite(match_space).all(axis=1)

    controls: Dict[str, List[int]] = {}
    diagnostics: Dict[str, Dict] = {
        "selectivity_cutoff_median": selectivity_cutoff,
        "n_controls_per_direction": n_controls,
        "directions": {},
    }

    for direction, target_fid in DIRECTION_FEATURES.items():
        candidate_mask = low_sel & active & finite
        candidate_mask[target_fid] = False
        for fid in target_ids:
            candidate_mask[fid] = False

        target_vec = match_space[target_fid]
        candidate_ids = np.where(candidate_mask)[0]
        distances = np.linalg.norm(match_space[candidate_ids] - target_vec[None, :], axis=1)
        order = np.argsort(distances)
        chosen = candidate_ids[order[:n_controls]].astype(int).tolist()
        controls[direction] = chosen

        diagnostics["directions"][direction] = {
            "target_feature": int(target_fid),
            "target_metrics": summarize_feature_metrics(target_fid, metrics, cdla),
            "candidate_count": int(len(candidate_ids)),
            "selected_controls": [
                {
                    "feature_id": int(fid),
                    "match_distance": float(distances[order[i]]),
                    "metrics": summarize_feature_metrics(int(fid), metrics, cdla),
                }
                for i, fid in enumerate(chosen)
            ],
        }

    return controls, diagnostics


def summarize_feature_metrics(fid: int, metrics: Dict[str, np.ndarray], cdla: np.ndarray) -> Dict:
    dir_cdla = {d: float(cdla[i, fid]) for i, d in enumerate(DIR_LABELS)}
    return {
        "mean_act": float(metrics["mean_act"][fid]),
        "firing_freq": float(metrics["firing_freq"][fid]),
        "decoder_norm": float(metrics["decoder_norm"][fid]),
        "selectivity": float(metrics["selectivity"][fid]),
        "max_cdla": float(metrics["max_cdla"][fid]),
        "cdla_by_direction": dir_cdla,
    }


# ─────────────────────────────────────────────────────────────
# Forward interventions
# ─────────────────────────────────────────────────────────────
def run_forward(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def make_l26_sae_hook(sae: SAE, the_pos: int, ablate_feature: int | None):
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
            feat = sae.encode(vec)
            if ablate_feature is not None:
                feat[:, int(ablate_feature)] = 0.0
            decoded = sae.decode(feat)

        hidden[:, the_pos, :] = decoded.to(dtype=hidden.dtype)
        if rest is not None:
            return (hidden, *rest)
        return hidden

    return hook


def run_l26_replacement(
    model,
    tokenizer,
    sae: SAE,
    prompt: str,
    ablate_feature: int | None,
) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    the_pos = get_the_pos(inputs.input_ids)
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(
        make_l26_sae_hook(sae, the_pos, ablate_feature)
    )
    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


# ─────────────────────────────────────────────────────────────
# Experiment and analysis
# ─────────────────────────────────────────────────────────────
def run_experiment(args) -> Dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_path, act_path, out_path = resolve_paths(args.dataset)
    all_rows = load_rows(data_path)
    rows = all_rows
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} samples.")

    model, tokenizer, sae = load_model_and_sae()
    acts = compute_or_load_the_sae_acts(model, tokenizer, sae, all_rows, act_path)
    dir_unembeds = {
        d: model.get_output_embeddings().weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
        for d in DIR_LABELS
    }

    w_dec = load_w_dec_cpu()
    cdla = compute_cdla_matrix(acts, all_rows, w_dec, dir_unembeds)
    controls, control_diagnostics = build_matched_controls(
        acts=acts,
        w_dec=w_dec,
        cdla=cdla,
        n_controls=args.n_controls,
    )

    print("[Control] matched controls selected per direction:")
    for d in DIR_LABELS:
        print(f"  {d}: target=#{DIRECTION_FEATURES[d]}, controls={controls[d]}")

    per_prompt = []
    for idx, row in enumerate(tqdm(rows, desc="[L26 feature ablation]", unit="prompt")):
        prompt = row["prompt"]

        clean_logits = run_forward(model, tokenizer, prompt)
        pred_dir = model_predict_from_logits(clean_logits)
        target_fid = DIRECTION_FEATURES[pred_dir]

        recon_logits = run_l26_replacement(model, tokenizer, sae, prompt, ablate_feature=None)
        target_logits = run_l26_replacement(model, tokenizer, sae, prompt, ablate_feature=target_fid)

        control_records = []
        for ctrl_fid in controls[pred_dir]:
            ctrl_logits = run_l26_replacement(model, tokenizer, sae, prompt, ablate_feature=ctrl_fid)
            control_records.append({
                "feature_id": int(ctrl_fid),
                "logit_pred_after": float(ctrl_logits[DIR_TOKEN_IDS[pred_dir]].item()),
                "delta_from_clean": float(
                    ctrl_logits[DIR_TOKEN_IDS[pred_dir]].item()
                    - clean_logits[DIR_TOKEN_IDS[pred_dir]].item()
                ),
                "delta_from_recon": float(
                    ctrl_logits[DIR_TOKEN_IDS[pred_dir]].item()
                    - recon_logits[DIR_TOKEN_IDS[pred_dir]].item()
                ),
                "pred_after": model_predict_from_logits(ctrl_logits),
            })

        clean_pred_logit = float(clean_logits[DIR_TOKEN_IDS[pred_dir]].item())
        recon_pred_logit = float(recon_logits[DIR_TOKEN_IDS[pred_dir]].item())
        target_pred_logit = float(target_logits[DIR_TOKEN_IDS[pred_dir]].item())

        per_prompt.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "prompt": prompt,
            "pred_dir_clean": pred_dir,
            "target_feature": int(target_fid),
            "clean_direction_logits": direction_logits(clean_logits),
            "recon_direction_logits": direction_logits(recon_logits),
            "target_direction_logits": direction_logits(target_logits),
            "logit_pred_clean": clean_pred_logit,
            "logit_pred_recon": recon_pred_logit,
            "logit_pred_target_ablate": target_pred_logit,
            "delta_target_from_clean": float(target_pred_logit - clean_pred_logit),
            "delta_target_from_recon": float(target_pred_logit - recon_pred_logit),
            "pred_dir_target_after": model_predict_from_logits(target_logits),
            "target_flip": bool(model_predict_from_logits(target_logits) != pred_dir),
            "controls": control_records,
            "delta_control_from_clean_mean": float(np.mean([c["delta_from_clean"] for c in control_records])),
            "delta_control_from_recon_mean": float(np.mean([c["delta_from_recon"] for c in control_records])),
            "control_flip_rate": float(np.mean([c["pred_after"] != pred_dir for c in control_records])),
        })

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary = summarize_results(per_prompt)
    result = {
        "experiment": "l26_direction_feature_ablation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "analysis_basis": "model_predicted_direction",
        "direction_features_top1_cdla": DIRECTION_FEATURES,
        "n_prompts": len(per_prompt),
        "n_controls_per_direction": args.n_controls,
        "metrics_note": {
            "delta_target_from_clean": "target-ablation logit(pred_dir) - clean logit(pred_dir)",
            "delta_target_from_recon": "target-ablation logit(pred_dir) - SAE-reconstruction-only logit(pred_dir)",
            "main_test": "paired Wilcoxon on delta_target_from_recon vs matched-control mean delta_from_recon; alternative='less'",
        },
        "summary": summary,
        "control_matching": control_diagnostics,
        "per_prompt": per_prompt,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Done] results saved to: {out_path}")
    print_summary(summary)
    return result


def summarize_results(per_prompt: List[Dict]) -> Dict:
    target_clean = np.array([x["delta_target_from_clean"] for x in per_prompt], dtype=np.float64)
    target_recon = np.array([x["delta_target_from_recon"] for x in per_prompt], dtype=np.float64)
    control_clean = np.array([x["delta_control_from_clean_mean"] for x in per_prompt], dtype=np.float64)
    control_recon = np.array([x["delta_control_from_recon_mean"] for x in per_prompt], dtype=np.float64)
    effect_vs_control = target_recon - control_recon

    try:
        stat, p_value = wilcoxon(
            target_recon,
            control_recon,
            alternative="less",
            zero_method="wilcox",
        )
        stat = float(stat)
        p_value = float(p_value)
    except Exception:
        stat, p_value = None, None

    by_pred_dir = {}
    for d in DIR_LABELS:
        xs = [x for x in per_prompt if x["pred_dir_clean"] == d]
        if not xs:
            continue
        by_pred_dir[d] = {
            "n": len(xs),
            "target_feature": int(DIRECTION_FEATURES[d]),
            "median_delta_target_from_recon": float(np.median([x["delta_target_from_recon"] for x in xs])),
            "mean_delta_target_from_recon": float(np.mean([x["delta_target_from_recon"] for x in xs])),
            "median_control_from_recon": float(np.median([x["delta_control_from_recon_mean"] for x in xs])),
            "target_flip_rate": float(np.mean([x["target_flip"] for x in xs])),
            "control_flip_rate_mean": float(np.mean([x["control_flip_rate"] for x in xs])),
        }

    return {
        "median_delta_target_from_clean": float(np.median(target_clean)),
        "mean_delta_target_from_clean": float(np.mean(target_clean)),
        "median_delta_control_from_clean": float(np.median(control_clean)),
        "mean_delta_control_from_clean": float(np.mean(control_clean)),
        "median_delta_target_from_recon": float(np.median(target_recon)),
        "mean_delta_target_from_recon": float(np.mean(target_recon)),
        "median_delta_control_from_recon": float(np.median(control_recon)),
        "mean_delta_control_from_recon": float(np.mean(control_recon)),
        "median_effect_vs_control_from_recon": float(np.median(effect_vs_control)),
        "mean_effect_vs_control_from_recon": float(np.mean(effect_vs_control)),
        "target_flip_rate": float(np.mean([x["target_flip"] for x in per_prompt])),
        "control_flip_rate_mean": float(np.mean([x["control_flip_rate"] for x in per_prompt])),
        "wilcoxon_stat_target_vs_control_recon": stat,
        "wilcoxon_p_target_less_than_control_recon": p_value,
        "by_predicted_direction": by_pred_dir,
    }


def print_summary(summary: Dict) -> None:
    print("\n[Summary]")
    print(f"  target Δlogit(pred) vs clean median: {summary['median_delta_target_from_clean']:+.4f}")
    print(f"  control Δlogit(pred) vs clean median: {summary['median_delta_control_from_clean']:+.4f}")
    print(f"  target Δlogit(pred) vs recon median: {summary['median_delta_target_from_recon']:+.4f}")
    print(f"  control Δlogit(pred) vs recon median: {summary['median_delta_control_from_recon']:+.4f}")
    print(f"  target-control effect median: {summary['median_effect_vs_control_from_recon']:+.4f}")
    print(f"  target flip rate: {summary['target_flip_rate']*100:.1f}%")
    print(f"  control flip rate mean: {summary['control_flip_rate_mean']*100:.1f}%")
    print(f"  Wilcoxon p(target < control, recon baseline): "
          f"{summary['wilcoxon_p_target_less_than_control_recon']}")


def parse_args():
    parser = argparse.ArgumentParser(description="L26 top-1 cDLA direction-feature ablation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "holdout"],
        help="train=balanced_city_pairs.jsonl; holdout=holdout_city_pairs.jsonl (city-disjoint holdout set).",
    )
    parser.add_argument(
        "--n_controls",
        type=int,
        default=1,
        help="number of nearest matched controls per direction; default 1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N samples (smoke test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
