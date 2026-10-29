#!/usr/bin/env python3
"""
Volume-matched control ablation for the high-activation L26 features #18707 / #38915.

These top-2 features have near-zero decoder alignment with any direction token but
large, nearly constant activation. This script matches each to control features
with similar activation, firing frequency and decoder norm but negligible
direction alignment, then ablates target vs controls on all prompts to test
whether their large logit drop reflects direction-specific support or a generic
volume effect.
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
ACT_PATH = ROOT / "sae" / "activations" / "the_sae_acts_l26.npy"
SAE_PATH = ROOT / "sae" / "final_sae" / "layer_26"
OUT_PATH = ROOT / "causal_verification" / "results" / "l26_high_activation_control_ablation.json"

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# high-activation target features and number of matched controls
TARGETS = [18707, 38915]
N_CONTROLS = 2
# excluded: the targets themselves + the four true top-1 direction features
EXCLUDE = set([18707, 38915, 47636, 56725, 10801, 26800])
# non-directional threshold: max|projection| below this is treated as non-directional
NON_DIRECTIONAL_THRESH = 0.10


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


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def compute_feature_metrics(acts: np.ndarray, model) -> dict:
    """mean_act / firing_freq / decoder_norm / max|direction projection|."""
    raw = load_file(str(SAE_PATH / "sae_weights.safetensors"))
    w_dec = raw.get("sae.W_dec", raw.get("W_dec")).float().cpu()

    mean_act = acts.mean(axis=0).astype(np.float64)
    firing_freq = (acts > 0).mean(axis=0).astype(np.float64)
    decoder_norm = torch.linalg.norm(w_dec, dim=1).numpy().astype(np.float64)

    w_u = {d: model.get_output_embeddings().weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
           for d in DIR_LABELS}
    proj = {d: (w_dec @ w_u[d]).numpy() for d in DIR_LABELS}
    max_abs_proj = np.max(np.abs(np.stack([proj[d] for d in DIR_LABELS], axis=0)), axis=0)

    return {
        "mean_act": mean_act,
        "firing_freq": firing_freq,
        "decoder_norm": decoder_norm,
        "proj": proj,
        "max_abs_proj": max_abs_proj,
    }


def select_controls(target_fid: int, metrics: dict, k: int) -> list:
    """Select k controls matched on activation/firing/norm and non-directional (small projection)."""
    match_space = np.stack([
        standardize(np.log1p(metrics["mean_act"])),
        standardize(metrics["firing_freq"]),
        standardize(np.log(metrics["decoder_norm"] + 1e-12)),
    ], axis=1)

    candidate_mask = (metrics["max_abs_proj"] < NON_DIRECTIONAL_THRESH)
    candidate_mask &= np.isfinite(match_space).all(axis=1)
    for fid in EXCLUDE:
        candidate_mask[fid] = False

    target_vec = match_space[target_fid]
    candidate_ids = np.where(candidate_mask)[0]
    dists = np.linalg.norm(match_space[candidate_ids] - target_vec[None, :], axis=1)
    order = np.argsort(dists)
    chosen = candidate_ids[order[:k]].astype(int).tolist()

    report = []
    for i, fid in enumerate(chosen):
        report.append({
            "feature_id": int(fid),
            "match_distance": float(dists[order[i]]),
            "mean_act": float(metrics["mean_act"][fid]),
            "firing_freq": float(metrics["firing_freq"][fid]),
            "decoder_norm": float(metrics["decoder_norm"][fid]),
            "max_abs_proj": float(metrics["max_abs_proj"][fid]),
        })
    return chosen, report


def run_experiment(args):
    rows = load_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    model, tokenizer, sae = load_model_and_sae()
    acts = np.load(str(ACT_PATH))
    if acts.shape[0] < len(rows):
        raise RuntimeError(f"activation cache has too few samples: acts={acts.shape}, rows={len(rows)}")

    metrics = compute_feature_metrics(acts, model)
    controls = {}
    control_diag = {}
    for t in TARGETS:
        ctrl, diag = select_controls(t, metrics, k=N_CONTROLS)
        controls[t] = ctrl
        control_diag[str(t)] = {
            "target": {
                "feature_id": t,
                "mean_act": float(metrics["mean_act"][t]),
                "firing_freq": float(metrics["firing_freq"][t]),
                "decoder_norm": float(metrics["decoder_norm"][t]),
                "max_abs_proj": float(metrics["max_abs_proj"][t]),
            },
            "controls": diag,
        }
        print(f"[Control] matched controls for #{t}: {ctrl}")

    all_fids = TARGETS + [c for t in TARGETS for c in controls[t]]

    per_prompt = []
    for idx, row in enumerate(tqdm(rows, desc="[high-activation control ablation]", unit="prompt")):
        prompt = row["prompt"]
        clean = run_clean(model, tokenizer, prompt)
        pred = model_predict_from_logits(clean)
        pred_id = DIR_TOKEN_IDS[pred]

        recon = run_replacement(model, tokenizer, sae, prompt, None)
        recon_logit = float(recon[pred_id].item())

        rec = {
            "index": idx,
            "pred_dir_clean": pred,
            "clean_direction_logits": direction_logits(clean),
            "deltas": {},
        }
        for fid in all_fids:
            ablated = run_replacement(model, tokenizer, sae, prompt, [fid])
            rec["deltas"][str(fid)] = float(ablated[pred_id].item()) - recon_logit
        per_prompt.append(rec)

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary = {}
    for t in TARGETS:
        t_delta = np.array([r["deltas"][str(t)] for r in per_prompt], dtype=np.float64)
        ctrl_deltas = np.array(
            [[r["deltas"][str(c)] for c in controls[t]] for r in per_prompt],
            dtype=np.float64,
        )
        ctrl_mean = ctrl_deltas.mean(axis=1)

        try:
            stat, p = wilcoxon(t_delta, ctrl_mean, zero_method="wilcox")  # two-sided
            stat, p = float(stat), float(p)
        except Exception:
            stat, p = None, None

        summary[str(t)] = {
            "target_median_delta": float(np.median(t_delta)),
            "target_mean_delta": float(np.mean(t_delta)),
            "target_flip_rate": float(np.mean(t_delta < 0)),
            "control_median_delta": float(np.median(ctrl_mean)),
            "control_mean_delta": float(np.mean(ctrl_mean)),
            "controls": [
                {"feature_id": c, "median_delta": float(np.median(ctrl_deltas[:, i]))}
                for i, c in enumerate(controls[t])
            ],
            "wilcoxon_target_vs_control_mean": {"stat": stat, "p": p},
        }

    result = {
        "experiment": "l26_high_activation_control_ablation",
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "targets": TARGETS,
        "n_controls_per_target": N_CONTROLS,
        "control_matching": control_diag,
        "summary": summary,
        "per_prompt": per_prompt,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary] median delta logit(pred) vs recon for targets vs volume-matched controls")
    for t in TARGETS:
        s = summary[str(t)]
        p = s["wilcoxon_target_vs_control_mean"]["p"]
        ps = f"{p:.2e}" if p is not None else "None"
        print(f"  #{t}: target median={s['target_median_delta']:+.3f}  "
              f"control median={s['control_median_delta']:+.3f}  "
              f"Wilcoxon p={ps}")
        for c in s["controls"]:
            print(f"       └ control #{c['feature_id']}: median={c['median_delta']:+.3f}")
    print(f"[Done] results saved to: {OUT_PATH}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="L26 high-activation control ablation")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
