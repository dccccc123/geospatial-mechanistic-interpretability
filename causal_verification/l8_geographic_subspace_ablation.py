#!/usr/bin/env python3
"""
L8 geographic-subspace ablation.

This is the direct causal test for the 21 L8 Moran's-I-selected geographic
features.

Question:
  If we remove the L8 geographic feature subspace at the target-city (c_B)
  position, does the model's own predicted-direction logit drop?

Design:
  For each prompt:
    1. Run clean forward and record the model-predicted direction d_pred.
    2. At L8 c_B position, encode the residual with the L8 SAE.
    3. Decode without ablation and write it back as the reconstruction baseline.
    4. Zero the 21 geographic features, decode, write back, and continue forward.
    5. Repeat with 21 matched non-geographic control features.

Main metric:
  Delta logit = logit_ablation(d_pred) - logit_reconstruction(d_pred)

The reconstruction baseline removes the generic effect of SAE encode/decode,
leaving the additional effect of removing the feature subspace.

Run:
  /opt/miniconda3/envs/pytorch_env/bin/python \
      causal_verification/l8_geographic_subspace_ablation.py
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


ROOT = Path(__file__).resolve().parent.parent
CITY_FEAT_L8 = ROOT / "feature_discovery" / "results" / "city_features_L8.json"
L8_SAE_PATH = ROOT / "sae" / "final_sae" / "layer_8"
RESULT_DIR = ROOT / "causal_verification" / "results"


def resolve_paths(dataset: str):
    """Return (input jsonl, c_B activation cache npy, output json) based on --dataset.

    train keeps the original paths; holdout uses *_holdout suffixes to avoid
    overwriting training-set results.
    """
    if dataset == "holdout":
        return (
            ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl",
            ROOT / "sae" / "activations" / "cb_sae_acts_l8_holdout.npy",
            RESULT_DIR / "l8_geographic_subspace_ablation_holdout.json",
        )
    return (
        ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl",
        ROOT / "sae" / "activations" / "cb_sae_acts_l8.npy",
        RESULT_DIR / "l8_geographic_subspace_ablation.json",
    )

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 8
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}


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


def load_geographic_features() -> List[int]:
    data = json.loads(CITY_FEAT_L8.read_text(encoding="utf-8"))
    return [int(x["feature_id"]) for x in data["significant_features"]]


def load_l8_decoder_norms() -> np.ndarray:
    raw = load_file(str(L8_SAE_PATH / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec"))
    if w_dec is None:
        raise KeyError("Cannot find W_dec / sae.W_dec in L8 SAE weights")
    return torch.linalg.norm(w_dec.float().cpu(), dim=1).numpy()


def load_l8_w_dec_cpu() -> torch.Tensor:
    raw = load_file(str(L8_SAE_PATH / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec"))
    if w_dec is None:
        raise KeyError("Cannot find W_dec / sae.W_dec in L8 SAE weights")
    return w_dec.float().cpu()


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def summarize_metrics(
    fid: int,
    mean_act: np.ndarray,
    firing_freq: np.ndarray,
    decoder_norm: np.ndarray,
    max_abs_dla: np.ndarray | None = None,
) -> Dict:
    item = {
        "mean_task_act": float(mean_act[fid]),
        "task_firing_freq": float(firing_freq[fid]),
        "decoder_norm": float(decoder_norm[fid]),
    }
    if max_abs_dla is not None:
        item["max_abs_direct_logit_contribution"] = float(max_abs_dla[fid])
    return item


def compute_or_load_cb_sae_acts(model, tokenizer, sae: SAE, rows: List[Dict], cb_acts_path: Path) -> np.ndarray:
    """Cache L8 SAE activations at the task prompt c_B position: [N, 65536]."""
    if cb_acts_path.exists():
        acts = np.load(str(cb_acts_path))
        if acts.shape[0] == len(rows):
            print(f"[Cache] loading L8 c_B SAE activations: {cb_acts_path}, shape={acts.shape}")
            return acts
        print(f"[Cache] old L8 c_B activation cache shape={acts.shape}, rows={len(rows)}, recomputing.")

    all_acts = []
    print("[Cache] computing L8 c_B SAE activations for low-DLA matched controls...")
    for row in tqdm(rows, desc="[extract L8 c_B SAE acts]", unit="prompt"):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        cb_pos = get_city_b_last_pos(inputs.input_ids, tokenizer, row["City_B"])
        if cb_pos < 0:
            raise RuntimeError(f"Cannot find c_B token for prompt: {row['prompt']}")

        captured = {}

        def hook(_module, _input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
            else:
                hidden = output.clone()
            hidden[:, :, OUTLIER_DIMS] = 0.0
            captured["h_cb"] = hidden[:, cb_pos, :].detach()
            return None

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                _ = model(**inputs)
        finally:
            handle.remove()

        with torch.inference_mode():
            feats = sae.encode(captured["h_cb"].to(device="cuda", dtype=torch.bfloat16))
        all_acts.append(feats.squeeze(0).detach().float().cpu().numpy())

    acts = np.stack(all_acts, axis=0).astype(np.float32)
    cb_acts_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cb_acts_path), acts)
    print(f"[Cache] saved L8 c_B SAE activations: {cb_acts_path}, shape={acts.shape}")
    return acts


def compute_max_abs_direct_logit_contribution(mean_task_act: np.ndarray, model) -> np.ndarray:
    """Direct-output proxy used only to keep controls low-contribution."""
    w_dec = load_l8_w_dec_cpu()
    vals = []
    for d in DIR_LABELS:
        u_d = model.get_output_embeddings().weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
        vals.append(mean_task_act * (w_dec @ u_d).numpy())
    return np.max(np.abs(np.stack(vals, axis=0)), axis=0).astype(np.float32)


def build_matched_controls(
    geo_fids: List[int],
    cb_task_acts: np.ndarray,
    model,
    low_dla_quantile: float,
) -> Tuple[List[int], Dict]:
    """
    Build one matched non-geographic control feature for each geographic feature.

    Candidate controls:
      - same L8 SAE dictionary
      - nontrivial at the task c_B position: zero_ratio <= 0.95
      - not among the Moran's-I-selected geographic features
      - low direct contribution to the four direction logits

    Matching dimensions:
      - log1p(mean activation at the task c_B position)
      - firing frequency at the task c_B position
      - log decoder-vector norm
    """
    decoder_norm = load_l8_decoder_norms()
    mean_act = cb_task_acts.mean(axis=0)
    firing_freq = (cb_task_acts > 0).mean(axis=0)
    zero_ratio = (cb_task_acts == 0).mean(axis=0)
    nontrivial = zero_ratio <= 0.95
    match_space = np.stack([
        standardize(np.log1p(mean_act)),
        standardize(firing_freq),
        standardize(np.log(decoder_norm + 1e-12)),
    ], axis=1)

    geo_set = set(int(x) for x in geo_fids)
    finite = np.isfinite(match_space).all(axis=1)
    base_candidate_mask = nontrivial & finite
    for fid in geo_set:
        base_candidate_mask[fid] = False

    max_abs_dla = compute_max_abs_direct_logit_contribution(mean_act, model)
    base_candidate_ids = np.where(base_candidate_mask)[0].astype(int)
    if len(base_candidate_ids) < len(geo_fids):
        raise RuntimeError(
            f"Base control pool too small: candidates={len(base_candidate_ids)}, "
            f"needed={len(geo_fids)}."
        )
    dla_cutoff = float(np.quantile(max_abs_dla[base_candidate_ids], low_dla_quantile))
    low_dla = max_abs_dla <= dla_cutoff
    candidate_mask = base_candidate_mask & low_dla
    candidates = np.where(candidate_mask)[0].astype(int)
    if len(candidates) < len(geo_fids):
        raise RuntimeError(
            f"Low-DLA control pool too small: candidates={len(candidates)}, "
            f"needed={len(geo_fids)}. Increase --low_dla_quantile."
        )

    used = set()
    control_fids = []
    per_feature = []
    for fid in geo_fids:
        available = np.array([c for c in candidates if int(c) not in used], dtype=np.int64)
        if len(available) == 0:
            raise RuntimeError("No available matched-control features remain")
        distances = np.linalg.norm(match_space[available] - match_space[fid][None, :], axis=1)
        best_pos = int(np.argmin(distances))
        ctrl = int(available[best_pos])
        used.add(ctrl)
        control_fids.append(ctrl)
        per_feature.append({
            "geo_feature": int(fid),
            "control_feature": ctrl,
            "match_distance": float(distances[best_pos]),
            "geo_metrics": summarize_metrics(fid, mean_act, firing_freq, decoder_norm, max_abs_dla),
            "control_metrics": summarize_metrics(ctrl, mean_act, firing_freq, decoder_norm, max_abs_dla),
        })

    diagnostics = {
        "matching": "greedy_1_to_1_nearest_neighbor",
        "candidate_rule": (
            "task c_B zero_ratio <= 0.95, not Moran-selected geographic features, "
            f"and max_abs_direct_logit_contribution <= q{low_dla_quantile:.2f}"
        ),
        "matched_dimensions": ["log1p_mean_task_activation", "task_firing_frequency", "log_decoder_norm"],
        "low_dla_quantile": float(low_dla_quantile),
        "low_dla_cutoff": dla_cutoff,
        "n_candidates": int(len(candidates)),
        "geo_features": [int(x) for x in geo_fids],
        "control_features": [int(x) for x in control_fids],
        "per_feature": per_feature,
    }
    return control_fids, diagnostics


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID not found in prompt")
    return int(positions[-1].item())


def get_city_b_last_pos(input_ids: torch.Tensor, tokenizer, city_b: str) -> int:
    prompt_ids = input_ids[0].tolist()
    candidate_id_lists = [
        tokenizer(" " + city_b, add_special_tokens=False).input_ids,
        tokenizer(city_b, add_special_tokens=False).input_ids,
    ]

    for city_ids in candidate_id_lists:
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
    city_lower = city_b.lower()
    last_pos = -1
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("▁Ġ").lower()
        if clean and city_lower.endswith(clean):
            last_pos = i
    return last_pos


def direction_logits(logits: torch.Tensor) -> Dict[str, float]:
    return {d: float(logits[DIR_TOKEN_IDS[d]].item()) for d in DIR_LABELS}


def model_predict_from_logits(logits: torch.Tensor) -> str:
    score_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    score_sn = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(score_ew) >= abs(score_sn):
        return "East" if score_ew > 0 else "West"
    return "South" if score_sn > 0 else "North"


def make_l8_ablation_hook(sae: SAE, cb_pos: int, ablate_fids: List[int] | None):
    fid_tensor = None
    if ablate_fids is not None:
        fid_tensor = torch.tensor(ablate_fids, dtype=torch.long, device="cuda")

    def hook(_module, _input, output):
        if cb_pos < 0:
            return output
        if isinstance(output, tuple):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None

        hidden_for_sae = hidden.clone()
        hidden_for_sae[:, :, OUTLIER_DIMS] = 0.0
        h_cb = hidden_for_sae[:, cb_pos, :].to(device="cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            feats = sae.encode(h_cb)
            if fid_tensor is not None:
                feats[:, fid_tensor] = 0.0
            h_new = sae.decode(feats)

        hidden[:, cb_pos, :] = h_new.to(dtype=hidden.dtype)
        if rest is not None:
            return (hidden, *rest)
        return hidden

    return hook


def run_clean(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def run_l8_replacement(
    model,
    tokenizer,
    sae: SAE,
    prompt: str,
    city_b: str,
    ablate_fids: List[int] | None,
) -> torch.Tensor | None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    cb_pos = get_city_b_last_pos(inputs.input_ids, tokenizer, city_b)
    if cb_pos < 0:
        return None

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(
        make_l8_ablation_hook(sae, cb_pos, ablate_fids)
    )
    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def load_model_and_sae():
    print("[Model] loading tokenizer/model/L8 SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    sae = SAE.load_from_disk(str(L8_SAE_PATH), device="cuda")
    sae.eval()
    print(f"[Model] done, device={next(model.parameters()).device}")
    return model, tokenizer, sae


def run_experiment(args) -> Dict:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data_path, cb_acts_path, out_path = resolve_paths(args.dataset)
    all_rows = load_rows(data_path)
    rows = all_rows
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    geo_fids = load_geographic_features()
    model, tokenizer, sae = load_model_and_sae()
    cb_task_acts = compute_or_load_cb_sae_acts(model, tokenizer, sae, all_rows, cb_acts_path)
    ctrl_fids, control_diagnostics = build_matched_controls(
        geo_fids,
        cb_task_acts=cb_task_acts,
        model=model,
        low_dla_quantile=args.low_dla_quantile,
    )
    print(f"[Data] prompts={len(rows)}")
    print(f"[Feature] geographic={len(geo_fids)}, low-DLA matched_control={len(ctrl_fids)}")

    records = []
    skipped = []
    for idx, row in enumerate(tqdm(rows, desc="[L8 geographic ablation]", unit="prompt")):
        prompt = row["prompt"]
        clean_logits = run_clean(model, tokenizer, prompt)
        pred_dir = model_predict_from_logits(clean_logits)

        recon_logits = run_l8_replacement(
            model, tokenizer, sae, prompt, row["City_B"], ablate_fids=None
        )
        geo_logits = run_l8_replacement(
            model, tokenizer, sae, prompt, row["City_B"], ablate_fids=geo_fids
        )
        ctrl_logits = run_l8_replacement(
            model, tokenizer, sae, prompt, row["City_B"], ablate_fids=ctrl_fids
        )
        if recon_logits is None or geo_logits is None or ctrl_logits is None:
            skipped.append({"index": idx, **row, "reason": "city_b_token_not_found"})
            continue

        pred_id = DIR_TOKEN_IDS[pred_dir]
        clean_logit = float(clean_logits[pred_id].item())
        recon_logit = float(recon_logits[pred_id].item())
        geo_logit = float(geo_logits[pred_id].item())
        ctrl_logit = float(ctrl_logits[pred_id].item())

        records.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "prompt": prompt,
            "pred_dir_clean": pred_dir,
            "clean_direction_logits": direction_logits(clean_logits),
            "recon_direction_logits": direction_logits(recon_logits),
            "geo_ablation_direction_logits": direction_logits(geo_logits),
            "control_ablation_direction_logits": direction_logits(ctrl_logits),
            "logit_pred_clean": clean_logit,
            "logit_pred_recon": recon_logit,
            "logit_pred_geo_ablation": geo_logit,
            "logit_pred_control_ablation": ctrl_logit,
            "delta_geo_from_clean": float(geo_logit - clean_logit),
            "delta_control_from_clean": float(ctrl_logit - clean_logit),
            "delta_geo_from_recon": float(geo_logit - recon_logit),
            "delta_control_from_recon": float(ctrl_logit - recon_logit),
            "pred_dir_geo_after": model_predict_from_logits(geo_logits),
            "pred_dir_control_after": model_predict_from_logits(ctrl_logits),
            "geo_flip": bool(model_predict_from_logits(geo_logits) != pred_dir),
            "control_flip": bool(model_predict_from_logits(ctrl_logits) != pred_dir),
        })

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary = summarize(records)
    result = {
        "experiment": "l8_geographic_subspace_ablation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "intervention_position": "L8 target-city c_B token",
        "analysis_basis": "model_predicted_direction",
        "main_metric": "delta logit of clean predicted direction, reconstruction-controlled",
        "metric_formula": "logit_ablation(d_pred) - logit_reconstruction(d_pred)",
        "n_prompts_total": len(rows),
        "n_prompts_used": len(records),
        "n_prompts_skipped": len(skipped),
        "geographic_features": [int(x) for x in geo_fids],
        "control_matching": control_diagnostics,
        "summary": summary,
        "per_prompt": records,
        "skipped": skipped,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"[Done] results saved to: {out_path}")
    return result


def summarize(records: List[Dict]) -> Dict:
    geo = np.array([r["delta_geo_from_recon"] for r in records], dtype=np.float64)
    ctrl = np.array([r["delta_control_from_recon"] for r in records], dtype=np.float64)
    diff = geo - ctrl

    try:
        stat, p_value = wilcoxon(geo, ctrl, alternative="less", zero_method="wilcox")
        stat = float(stat)
        p_value = float(p_value)
    except Exception:
        stat, p_value = None, None

    by_pred = {}
    for d in DIR_LABELS:
        xs = [r for r in records if r["pred_dir_clean"] == d]
        if not xs:
            continue
        by_pred[d] = {
            "n": len(xs),
            "median_delta_geo_from_recon": float(np.median([x["delta_geo_from_recon"] for x in xs])),
            "median_delta_control_from_recon": float(np.median([x["delta_control_from_recon"] for x in xs])),
            "mean_delta_geo_from_recon": float(np.mean([x["delta_geo_from_recon"] for x in xs])),
            "mean_delta_control_from_recon": float(np.mean([x["delta_control_from_recon"] for x in xs])),
            "geo_flip_rate": float(np.mean([x["geo_flip"] for x in xs])),
            "control_flip_rate": float(np.mean([x["control_flip"] for x in xs])),
        }

    return {
        "median_delta_geo_from_recon": float(np.median(geo)),
        "median_delta_control_from_recon": float(np.median(ctrl)),
        "median_geo_minus_control_delta": float(np.median(diff)),
        "mean_delta_geo_from_recon": float(np.mean(geo)),
        "mean_delta_control_from_recon": float(np.mean(ctrl)),
        "mean_geo_minus_control_delta": float(np.mean(diff)),
        "geo_negative_delta_rate": float(np.mean(geo < 0)),
        "control_negative_delta_rate": float(np.mean(ctrl < 0)),
        "geo_flip_rate": float(np.mean([r["geo_flip"] for r in records])),
        "control_flip_rate": float(np.mean([r["control_flip"] for r in records])),
        "wilcoxon_stat_geo_less_than_control": stat,
        "wilcoxon_p_geo_less_than_control": p_value,
        "by_predicted_direction": by_pred,
    }


def print_summary(summary: Dict) -> None:
    print("\n[Summary]")
    print(f"  median geo Δlogit vs recon    : {summary['median_delta_geo_from_recon']:+.4f}")
    print(f"  median control Δlogit vs recon: {summary['median_delta_control_from_recon']:+.4f}")
    print(f"  median geo-control Δlogit     : {summary['median_geo_minus_control_delta']:+.4f}")
    print(f"  geo negative Δlogit rate      : {summary['geo_negative_delta_rate']*100:.1f}%")
    print(f"  ctrl negative Δlogit rate     : {summary['control_negative_delta_rate']*100:.1f}%")
    print(f"  geo flip rate                 : {summary['geo_flip_rate']*100:.1f}%")
    print(f"  ctrl flip rate                : {summary['control_flip_rate']*100:.1f}%")
    print(f"  Wilcoxon p(geo < control)     : {summary['wilcoxon_p_geo_less_than_control']}")


def parse_args():
    parser = argparse.ArgumentParser(description="L8 geographic-subspace ablation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "holdout"],
        help="train=balanced_city_pairs.jsonl; holdout=holdout_city_pairs.jsonl (city-disjoint holdout set).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    parser.add_argument(
        "--low_dla_quantile",
        type=float,
        default=0.25,
        help="Control features must be below this quantile of max absolute direct logit contribution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
