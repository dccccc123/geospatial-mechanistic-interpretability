#!/usr/bin/env python3
"""
Inspect decoder-direction projection and per-direction activation for six L26 features.

For each feature this prints two tables: the direct logit projection
proj_d = W_dec[fid] . W_U[token_d] onto each direction token, and the mean
activation act_d on prompts whose ground-truth direction is d. This diagnoses
which direction a feature's decoder actually points toward (e.g. whether #18707
is East, North, or a mixture).

Pure CPU; reads cached activations, does not load the GPU model.
"""
import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
SAE_DIR = ROOT / "sae" / "final_sae" / "layer_26"
ACTS = ROOT / "sae" / "activations" / "the_sae_acts_l26.npy"
PAIRS = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"

DIR_TOK = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
DIRS = ["East", "West", "South", "North"]

FEATS = {
    "East_top1": 47636,
    "East/North_top2": 18707,
    "West_top1": 56725,
    "West/South_top2": 38915,
    "South_top1": 10801,
    "North_top1": 26800,
}


def main():
    acts = np.load(str(ACTS))                                   # [180, 65536]
    dirs = [json.loads(l)["Direction"] for l in open(PAIRS) if l.strip()]
    dir_idx = {d: [i for i, x in enumerate(dirs) if x == d] for d in DIRS}

    raw = load_file(str(SAE_DIR / "sae_weights.safetensors"))
    W_dec = raw.get("sae.W_dec", raw.get("W_dec")).float()

    print("[Model] Loading lm_head (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, local_files_only=True,
    )
    W_U = model.get_output_embeddings().weight.detach().float()
    del model

    print("\n=== Decoder direction projection  W_dec[fid] . W_U[direction_token] ===")
    header = f"{'feature':>18} | " + " | ".join(f"{d:>8}" for d in DIRS)
    print(header)
    print("-" * len(header))
    for name, fid in FEATS.items():
        proj = {d: float(W_dec[fid] @ W_U[DIR_TOK[d]]) for d in DIRS}
        row = f"{name:>18} (#{fid}) | " + " | ".join(f"{proj[d]:+8.2f}" for d in DIRS)
        print(row)

    print("\n=== Mean activation per ground-truth direction ===")
    print(f"{'feature':>18} | " + " | ".join(f"{d:>8}" for d in DIRS))
    print("-" * len(header))
    for name, fid in FEATS.items():
        act = {d: float(acts[dir_idx[d]][:, fid].mean()) for d in DIRS}
        row = f"{name:>18} (#{fid}) | " + " | ".join(f"{act[d]:8.2f}" for d in DIRS)
        print(row)


if __name__ == "__main__":
    main()
