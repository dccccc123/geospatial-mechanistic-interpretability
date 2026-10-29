#!/usr/bin/env python3
"""
L8 geographic-subspace ablation at the c_A / c_B / joint positions.

Ablates the L8 geographic features at the reference city c_A, at the target
city c_B, and at both jointly, comparing the predicted-direction logit drop
(relative to the SAE reconstruction baseline) and flip rate across the three
interventions.
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
from safetensors.torch import load_file
from scipy.stats import wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
CITY_FEAT_L8 = ROOT / "feature_discovery" / "results" / "city_features_L8.json"
L8_SAE_PATH = ROOT / "sae" / "final_sae" / "layer_8"
OUT_PATH = ROOT / "causal_verification" / "results" / "l8_geographic_subspace_ablation_ca_joint.json"


def resolve_paths(dataset: str):
    """Return (input jsonl, c_B activation cache npy, output json) based on --dataset."""
    if dataset == "holdout":
        return (
            ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl",
            ROOT / "sae" / "activations" / "cb_sae_acts_l8_holdout.npy",
            ROOT / "causal_verification" / "results" / "l8_geographic_subspace_ablation_ca_joint_holdout.json",
        )
    return (
        ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl",
        ROOT / "sae" / "activations" / "cb_sae_acts_l8.npy",
        ROOT / "causal_verification" / "results" / "l8_geographic_subspace_ablation_ca_joint.json",
    )

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 8
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}


def load_rows(data_path):
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


def load_geographic_features():
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


def summarize_metrics(fid, mean_act, firing_freq, decoder_norm, max_abs_dla=None):
    item = {
        "mean_task_act": float(mean_act[fid]),
        "task_firing_freq": float(firing_freq[fid]),
        "decoder_norm": float(decoder_norm[fid]),
    }
    if max_abs_dla is not None:
        item["max_abs_direct_logit_contribution"] = float(max_abs_dla[fid])
    return item


def compute_max_abs_direct_logit_contribution(mean_task_act, model) -> np.ndarray:
    """Direct-output proxy used only to keep controls low-contribution."""
    w_dec = load_l8_w_dec_cpu()
    vals = []
    for d in DIR_LABELS:
        u_d = model.get_output_embeddings().weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
        vals.append(mean_task_act * (w_dec @ u_d).numpy())
    return np.max(np.abs(np.stack(vals, axis=0)), axis=0).astype(np.float32)


def compute_or_load_cb_sae_acts(model, tokenizer, sae, rows, cb_acts_path) -> np.ndarray:
    """Cache L8 c_B-position SAE activations [N, 65536] for matched controls."""
    if cb_acts_path.exists():
        acts = np.load(str(cb_acts_path))
        if acts.shape[0] == len(rows):
            print(f"[Cache] loading L8 c_B SAE activations: {cb_acts_path}, shape={acts.shape}")
            return acts
        print(f"[Cache] old L8 c_B activation cache shape={acts.shape}, rows={len(rows)}, recomputing.")

    all_acts = []
    print("[Cache] computing L8 c_B SAE activations for matched controls...")
    for row in tqdm(rows, desc="[extract L8 c_B SAE acts]", unit="prompt"):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        cb_pos = get_city_last_pos(inputs.input_ids, tokenizer, row["City_B"])
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


def build_matched_controls(geo_fids, cb_task_acts, model, low_dla_quantile):
    """Pair each geographic feature with a non-geographic control of similar activation/frequency/norm (low direction logit contribution)."""
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
        "low_dla_cutoff": float(dla_cutoff),
        "n_candidates": int(len(candidates)),
        "geo_features": [int(x) for x in geo_fids],
        "control_features": [int(x) for x in control_fids],
        "per_feature": per_feature,
    }
    return control_fids, diagnostics


def get_city_last_pos(input_ids: torch.Tensor, tokenizer, city: str) -> int:
    prompt_ids = input_ids[0].tolist()
    candidate_id_lists = [
        tokenizer(" " + city, add_special_tokens=False).input_ids,
        tokenizer(city, add_special_tokens=False).input_ids,
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
    city_lower = city.lower()
    last_pos = -1
    for i, tok in enumerate(tokens):
        clean = tok.lstrip("▁Ġ").lower()
        if clean and city_lower.endswith(clean):
            last_pos = i
    return last_pos


def direction_logits(logits: torch.Tensor):
    return {d: float(logits[DIR_TOKEN_IDS[d]].item()) for d in DIR_LABELS}


def model_predict_from_logits(logits: torch.Tensor) -> str:
    s_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    s_ns = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


def make_l8_hook(sae: SAE, positions, ablate_fids):
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

        with torch.inference_mode():
            for pos in positions:
                h = hidden_for_sae[:, pos, :].to(device="cuda", dtype=torch.bfloat16)
                feats = sae.encode(h)
                if fid_tensor is not None:
                    feats[:, fid_tensor] = 0.0
                h_new = sae.decode(feats)
                hidden[:, pos, :] = h_new.to(dtype=hidden.dtype)

        if rest is not None:
            return (hidden, *rest)
        return hidden

    return hook


def run_clean(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def run_l8_replacement(model, tokenizer, sae, prompt: str, positions, ablate_fids):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(
        make_l8_hook(sae, positions, ablate_fids)
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
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    sae = SAE.load_from_disk(str(L8_SAE_PATH), device="cuda")
    sae.eval()
    return model, tokenizer, sae


def one_sample_wilcoxon_less(x):
    try:
        stat, p = wilcoxon(x, alternative="less", zero_method="wilcox")
        return float(stat), float(p)
    except Exception:
        return None, None


def run_experiment(args):
    data_path, cb_acts_path, out_path = resolve_paths(args.dataset)
    all_rows = load_rows(data_path)
    rows = all_rows
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    geo_fids = load_geographic_features()
    print(f"[Feature] geographic={len(geo_fids)}")

    model, tokenizer, sae = load_model_and_sae()

    # matched controls: pair each geographic feature with a non-geographic control of similar activation/frequency/norm
    cb_task_acts = compute_or_load_cb_sae_acts(model, tokenizer, sae, all_rows, cb_acts_path)
    ctrl_fids, control_diagnostics = build_matched_controls(
        geo_fids, cb_task_acts=cb_task_acts, model=model,
        low_dla_quantile=args.low_dla_quantile,
    )
    print(f"[Control] geographic={len(geo_fids)}, matched_control={len(ctrl_fids)}")

    records = []
    skipped = []
    for idx, row in enumerate(tqdm(rows, desc="[L8 c_A/c_B/joint ablation]", unit="prompt")):
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        ca_pos = get_city_last_pos(inputs.input_ids, tokenizer, row["City_A"])
        cb_pos = get_city_last_pos(inputs.input_ids, tokenizer, row["City_B"])
        if ca_pos < 0 or cb_pos < 0:
            skipped.append({"index": idx, "city_a": row["City_A"], "city_b": row["City_B"],
                            "reason": "city_token_not_found"})
            continue

        clean = run_clean(model, tokenizer, prompt)
        pred = model_predict_from_logits(clean)
        pred_id = DIR_TOKEN_IDS[pred]

        recon_A = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos], None)
        geo_A = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos], geo_fids)
        ctrl_A = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos], ctrl_fids)
        recon_B = run_l8_replacement(model, tokenizer, sae, prompt, [cb_pos], None)
        geo_B = run_l8_replacement(model, tokenizer, sae, prompt, [cb_pos], geo_fids)
        ctrl_B = run_l8_replacement(model, tokenizer, sae, prompt, [cb_pos], ctrl_fids)
        recon_AB = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos, cb_pos], None)
        geo_AB = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos, cb_pos], geo_fids)
        ctrl_AB = run_l8_replacement(model, tokenizer, sae, prompt, [ca_pos, cb_pos], ctrl_fids)

        records.append({
            "index": idx,
            "city_a": row["City_A"],
            "city_b": row["City_B"],
            "ground_truth": row["Direction"],
            "pred_dir_clean": pred,
            "clean_direction_logits": direction_logits(clean),
            "recon_A_direction_logits": direction_logits(recon_A),
            "geo_A_direction_logits": direction_logits(geo_A),
            "ctrl_A_direction_logits": direction_logits(ctrl_A),
            "recon_B_direction_logits": direction_logits(recon_B),
            "geo_B_direction_logits": direction_logits(geo_B),
            "ctrl_B_direction_logits": direction_logits(ctrl_B),
            "recon_AB_direction_logits": direction_logits(recon_AB),
            "geo_AB_direction_logits": direction_logits(geo_AB),
            "ctrl_AB_direction_logits": direction_logits(ctrl_AB),
            "delta_geo_A_from_recon": float(geo_A[pred_id].item() - recon_A[pred_id].item()),
            "delta_ctrl_A_from_recon": float(ctrl_A[pred_id].item() - recon_A[pred_id].item()),
            "delta_geo_B_from_recon": float(geo_B[pred_id].item() - recon_B[pred_id].item()),
            "delta_ctrl_B_from_recon": float(ctrl_B[pred_id].item() - recon_B[pred_id].item()),
            "delta_geo_AB_from_recon": float(geo_AB[pred_id].item() - recon_AB[pred_id].item()),
            "delta_ctrl_AB_from_recon": float(ctrl_AB[pred_id].item() - recon_AB[pred_id].item()),
            "flip_A": model_predict_from_logits(geo_A) != pred,
            "flip_B": model_predict_from_logits(geo_B) != pred,
            "flip_AB": model_predict_from_logits(geo_AB) != pred,
        })

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    def block(name, geo_delta, ctrl_delta, flips):
        stat_g, p_g = one_sample_wilcoxon_less(geo_delta)
        try:
            stat_gc, p_gc = wilcoxon(geo_delta, ctrl_delta, alternative="less", zero_method="wilcox")
            stat_gc, p_gc = float(stat_gc), float(p_gc)
        except Exception:
            stat_gc, p_gc = None, None
        return {
            "n": int(len(geo_delta)),
            "median_delta_geo_from_recon": float(np.median(geo_delta)),
            "median_delta_ctrl_from_recon": float(np.median(ctrl_delta)),
            "mean_delta_geo_from_recon": float(np.mean(geo_delta)),
            "mean_delta_ctrl_from_recon": float(np.mean(ctrl_delta)),
            "median_geo_minus_ctrl": float(np.median(geo_delta - ctrl_delta)),
            "flip_rate": float(np.mean(flips)),
            "wilcoxon_geo_less_than_0_stat": stat_g,
            "wilcoxon_geo_less_than_0_p": p_g,
            "wilcoxon_geo_less_than_ctrl_stat": stat_gc,
            "wilcoxon_geo_less_than_ctrl_p": p_gc,
        }

    dA = np.array([r["delta_geo_A_from_recon"] for r in records], dtype=np.float64)
    dCA = np.array([r["delta_ctrl_A_from_recon"] for r in records], dtype=np.float64)
    dB = np.array([r["delta_geo_B_from_recon"] for r in records], dtype=np.float64)
    dCB = np.array([r["delta_ctrl_B_from_recon"] for r in records], dtype=np.float64)
    dAB = np.array([r["delta_geo_AB_from_recon"] for r in records], dtype=np.float64)
    dCAB = np.array([r["delta_ctrl_AB_from_recon"] for r in records], dtype=np.float64)

    # joint vs single: paired Wilcoxon for whether joint is more negative
    try:
        stat_jb, p_jb = wilcoxon(dAB, dB, alternative="less", zero_method="wilcox")
        stat_jb, p_jb = float(stat_jb), float(p_jb)
    except Exception:
        stat_jb, p_jb = None, None
    try:
        stat_ja, p_ja = wilcoxon(dAB, dA, alternative="less", zero_method="wilcox")
        stat_ja, p_ja = float(stat_ja), float(p_ja)
    except Exception:
        stat_ja, p_ja = None, None

    summary = {
        "c_A": block("c_A", dA, dCA, [r["flip_A"] for r in records]),
        "c_B": block("c_B", dB, dCB, [r["flip_B"] for r in records]),
        "c_A_c_B_joint": block("c_A+c_B", dAB, dCAB, [r["flip_AB"] for r in records]),
        "wilcoxon_joint_less_than_c_B": {"stat": stat_jb, "p": p_jb},
        "wilcoxon_joint_less_than_c_A": {"stat": stat_ja, "p": p_ja},
    }

    result = {
        "experiment": "l8_geographic_subspace_ablation_ca_joint",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "analysis_basis": "model_predicted_direction",
        "geographic_features": [int(x) for x in geo_fids],
        "control_matching": control_diagnostics,
        "n_prompts_total": len(rows),
        "n_prompts_used": len(records),
        "n_prompts_skipped": len(skipped),
        "summary": summary,
        "per_prompt": records,
        "skipped": skipped,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary]")
    for key in ("c_A", "c_B", "c_A_c_B_joint"):
        b = summary[key]
        print(f"  {key:12s}: geo median Δ={b['median_delta_geo_from_recon']:+.3f}  "
              f"ctrl median Δ={b['median_delta_ctrl_from_recon']:+.3f}  "
              f"flip={b['flip_rate']*100:.1f}%  p(geo<ctrl)={b['wilcoxon_geo_less_than_ctrl_p']}")
    print(f"  joint vs c_B: p={summary['wilcoxon_joint_less_than_c_B']['p']}")
    print(f"  joint vs c_A: p={summary['wilcoxon_joint_less_than_c_A']['p']}")
    print(f"[Done] results saved to: {out_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="L8 c_A / c_B / joint geographic-subspace ablation")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    parser.add_argument("--dataset", type=str, default="train", choices=["train", "holdout"],
                        help="train=balanced_city_pairs.jsonl; holdout=holdout_city_pairs.jsonl (city-disjoint holdout set)")
    parser.add_argument(
        "--low_dla_quantile",
        type=float,
        default=0.25,
        help="Control features must be below this quantile of max absolute direct logit contribution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
