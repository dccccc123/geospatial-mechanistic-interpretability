#!/usr/bin/env python3
"""
Layer-wise direction selectivity via the logit lens.

Projects the residual at the "the" position of each layer onto the four direction
tokens and computes selectivity (top logit minus second-top logit) and logit-lens
accuracy, showing where direction information accumulates across layers.
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

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
OUT_PATH = ROOT / "causal_verification" / "results" / "layer_direction_selectivity.json"

MODEL_NAME = "Qwen/Qwen3-8B"
N_LAYERS = 36
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279

DIR_LABELS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}


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


def get_the_pos(input_ids: torch.Tensor) -> int:
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise RuntimeError("THE_TOKEN_ID=279 not found")
    return int(positions[-1].item())


def model_predict_from_logits(logits: torch.Tensor) -> str:
    s_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    s_ns = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(s_ew) >= abs(s_ns):
        return "East" if s_ew > 0 else "West"
    return "South" if s_ns > 0 else "North"


def make_residual_capture_hook(the_pos: int, store: dict, layer_idx: int):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store[layer_idx] = hidden[:, the_pos, :].detach().clone()
        return None

    return hook


def load_model():
    print("[Model] loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def run_experiment(args):
    rows = load_rows()
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[Debug] running only the first {len(rows)} prompts")

    model, tokenizer = load_model()
    W_U = model.get_output_embeddings().weight.detach().float().cpu()  # [vocab, d_model]
    W_dir = torch.stack([W_U[DIR_TOKEN_IDS[d]] for d in DIR_LABELS], dim=0)  # [4, d_model]

    selectivities = {L: [] for L in range(N_LAYERS)}
    lens_acc = {L: [] for L in range(N_LAYERS)}

    for idx, row in enumerate(tqdm(rows, desc="[layer direction selectivity]", unit="prompt")):
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)

        store = {}
        handles = [
            model.model.layers[L].register_forward_hook(
                make_residual_capture_hook(the_pos, store, L)
            )
            for L in range(N_LAYERS)
        ]
        with torch.inference_mode():
            out = model(**inputs)
        for h in handles:
            h.remove()

        final_logits = out.logits[0, -1].detach().float().cpu()
        final_pred = model_predict_from_logits(final_logits)

        for L in range(N_LAYERS):
            residual = store[L].float().cpu()  # [1, d_model]
            logits = (W_dir @ residual.T).squeeze(1)  # [4]
            logits = {d: float(logits[i].item()) for i, d in enumerate(DIR_LABELS)}
            vals = [logits[d] for d in DIR_LABELS]
            top = max(vals)
            second = sorted(vals, reverse=True)[1]
            selectivities[L].append(top - second)

            # logit-lens direction prediction (axis rule)
            s_ew = logits["East"] - logits["West"]
            s_ns = logits["South"] - logits["North"]
            lens_pred = ("East" if s_ew > 0 else "West") if abs(s_ew) >= abs(s_ns) else ("South" if s_ns > 0 else "North")
            lens_acc[L].append(lens_pred == final_pred)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "n_prompts": len(rows),
        "per_layer": {},
    }
    for L in range(N_LAYERS):
        summary["per_layer"][str(L)] = {
            "median_selectivity": float(np.median(selectivities[L])),
            "logit_lens_accuracy": float(np.mean(lens_acc[L])),
        }

    result = {
        "experiment": "layer_direction_selectivity",
        "model_name": MODEL_NAME,
        "summary": summary,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[Summary] per-layer direction selectivity (median) and logit-lens accuracy")
    print(f"  {'Layer':>4} {'selectivity':>12} {'lens acc':>8}")
    for L in range(N_LAYERS):
        s = summary["per_layer"][str(L)]
        marker = ""
        if L in (0, 1):
            marker = "  <- lexical"
        elif L == 8:
            marker = "  <- geographic"
        elif L == 26:
            marker = "  <- direction"
        elif L == 34:
            marker = "  <- output"
        print(f"  {L:>4} {s['median_selectivity']:>12.3f} {s['logit_lens_accuracy']*100:>7.1f}%{marker}")
    print(f"[Done] results saved to: {OUT_PATH}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Layer-wise direction selectivity (logit lens)")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N prompts for smoke test")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
