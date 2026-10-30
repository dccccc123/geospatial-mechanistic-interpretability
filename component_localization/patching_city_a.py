import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# =============================================================================
# City_A mirror of patching.py.
#
# Original patching.py pairs clean/corrupt by SAME City_A + OPPOSITE direction,
# flipping only City_B. Under causal masking the City_A position is then
# identical across clean and corrupt, so patching it is a structural no-op
# (Metric == 1) and the design cannot probe City_A.
#
# This script mirrors that: it pairs by SAME City_B + OPPOSITE direction,
# flipping only City_A. Now the City_A position differs between clean and
# corrupt and becomes the meaningful intervention site (we expect a peak near
# L8, symmetric to City_B); the City_B stage becomes the no-op here. Everything
# else -- model loading, hook_mlp_out / hook_z patching, the normalised metric
# -- is identical to patching.py for direct comparability.
# =============================================================================


def load_model(model_name: str) -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        use_attn_result=False,
    )
    model.eval()
    return model


PROMPT_TEMPLATE = "Relative to {city_a}, {city_b} is to the"
STAGES = ["city_a", "city_b", "final"]
OPPOSITE_DIRECTION = {
    "east": "west",
    "west": "east",
    "north": "south",
    "south": "north",
}


@dataclass
class PairSample:
    clean_prompt: str
    corrupt_prompt: str
    clean_target_id: int
    corrupt_target_id: int
    clean_pos_city_a: int
    clean_pos_city_b: int
    clean_pos_final: int
    corrupt_pos_city_a: int
    corrupt_pos_city_b: int
    corrupt_pos_final: int


def compute_positions(model: HookedTransformer, city_a: str, city_b: str, prepend_bos: bool) -> dict:
    prompt = PROMPT_TEMPLATE.format(city_a=city_a, city_b=city_b)
    prefix_city_a = f"Relative to {city_a}"
    prefix_city_b = f"Relative to {city_a}, {city_b}"

    pos_city_a = model.to_tokens(prefix_city_a, prepend_bos=prepend_bos).shape[1] - 1
    pos_city_b = model.to_tokens(prefix_city_b, prepend_bos=prepend_bos).shape[1] - 1
    pos_final = model.to_tokens(prompt, prepend_bos=prepend_bos).shape[1] - 1
    return {
        "prompt": prompt,
        "pos_city_a": int(pos_city_a),
        "pos_city_b": int(pos_city_b),
        "pos_final": int(pos_final),
    }


def direction_to_token_id(model: HookedTransformer, direction: str) -> int:
    token_text = f" {direction}"
    token_ids = model.to_tokens(token_text, prepend_bos=False).squeeze(0)
    if token_ids.numel() != 1:
        raise ValueError(f"Direction token '{token_text}' is not a single token: {token_ids.tolist()}")
    return int(token_ids.item())


def normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    alias = {"e": "east", "w": "west", "n": "north", "s": "south"}
    d = alias.get(d, d)
    if d not in OPPOSITE_DIRECTION:
        raise ValueError(f"Unsupported direction: {direction}")
    return d


def batched(seq, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def gather_logit_diff(logits, final_positions, clean_target_ids, corrupt_target_ids):
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    clean_logits = logits[batch_idx, final_positions, clean_target_ids]
    corrupt_logits = logits[batch_idx, final_positions, corrupt_target_ids]
    return clean_logits - corrupt_logits


def topk_nodes(metric_mlp, metric_attn, topk: int):
    nodes = []
    n_layers = metric_mlp.shape[0]
    n_heads = metric_attn.shape[1]
    for layer in range(n_layers):
        nodes.append((f"MLP L{layer}", float(metric_mlp[layer].item())))
    for layer in range(n_layers):
        for head in range(n_heads):
            nodes.append((f"Head L{layer}H{head}", float(metric_attn[layer, head].item())))
    nodes.sort(key=lambda x: x[1])
    return nodes[:topk]


def save_patching_results(save_npz_path, save_meta_path, metric_mlp, metric_attn,
                          clean_diff, corrupt_diff, denom, n_layers, n_heads,
                          total_n, model_name, dataset_path):
    save_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta_path.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "stages": np.array(STAGES),
        "clean_diff": np.array([clean_diff], dtype=np.float32),
        "corrupt_diff": np.array([corrupt_diff], dtype=np.float32),
        "denominator": np.array([denom], dtype=np.float32),
        "n_layers": np.array([n_layers], dtype=np.int32),
        "n_heads": np.array([n_heads], dtype=np.int32),
        "num_pairs": np.array([total_n], dtype=np.int32),
    }
    for stage in STAGES:
        npz_payload[f"metric_mlp_{stage}"] = metric_mlp[stage].detach().cpu().numpy()
        npz_payload[f"metric_attn_{stage}"] = metric_attn[stage].detach().cpu().numpy()
    np.savez_compressed(save_npz_path, **npz_payload)
    meta = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "pairing": "same_city_b_opposite_direction (City_A mirror)",
        "stages": STAGES,
        "clean_diff": clean_diff,
        "corrupt_diff": corrupt_diff,
        "denominator": denom,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "num_pairs": total_n,
    }
    with save_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="City_A activation patching (same-City_B, opposite-direction pairs)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-path", type=str, default="data_process/dataset/balanced_city_pairs.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-results-npz", type=str,
                        default="component_localization/patching_results_cityA.npz")
    parser.add_argument("--save-results-meta", type=str,
                        default="component_localization/patching_results_cityA_meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[Info] Loading model: {args.model_name}")
    model = load_model(args.model_name)
    prepend_bos = model.cfg.default_prepend_bos
    device = model.cfg.device
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    print(f"[Info] Loading dataset: {dataset_path}")
    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            direction = item.get("Direction")
            if direction is None:
                continue
            try:
                item["_direction_norm"] = normalize_direction(direction)
                rows.append(item)
            except ValueError:
                continue
    if not rows:
        raise ValueError("No valid rows with four cardinal directions found in dataset.")

    # --- KEY DIFFERENCE: group by City_B (not City_A) ---
    rows_by_city_b_and_direction: dict = {}
    for r in rows:
        city_b = r["City_B"]
        direction = r["_direction_norm"]
        if city_b not in rows_by_city_b_and_direction:
            rows_by_city_b_and_direction[city_b] = {k: [] for k in OPPOSITE_DIRECTION}
        rows_by_city_b_and_direction[city_b][direction].append(r)

    token_id_by_direction = {d: direction_to_token_id(model, d) for d in OPPOSITE_DIRECTION}

    pairs = []
    skipped = 0
    for clean_row in rows:
        clean_city_b = clean_row["City_B"]
        clean_dir = clean_row["_direction_norm"]
        corrupt_dir = OPPOSITE_DIRECTION[clean_dir]
        # corrupt must share City_B but have the opposite direction (-> different City_A)
        corrupt_pool = rows_by_city_b_and_direction.get(clean_city_b, {}).get(corrupt_dir, [])
        if not corrupt_pool:
            skipped += 1
            continue
        corrupt_row = random.choice(corrupt_pool)

        clean_city_a = clean_row["City_A"]
        corrupt_city_b = clean_city_b              # SAME City_B
        corrupt_city_a = corrupt_row["City_A"]     # DIFFERENT City_A

        clean_meta = compute_positions(model, clean_city_a, clean_city_b, prepend_bos)
        corrupt_meta = compute_positions(model, corrupt_city_a, corrupt_city_b, prepend_bos)

        pairs.append(
            PairSample(
                clean_prompt=clean_meta["prompt"],
                corrupt_prompt=corrupt_meta["prompt"],
                clean_target_id=token_id_by_direction[clean_dir],
                corrupt_target_id=token_id_by_direction[corrupt_dir],
                clean_pos_city_a=clean_meta["pos_city_a"],
                clean_pos_city_b=clean_meta["pos_city_b"],
                clean_pos_final=clean_meta["pos_final"],
                corrupt_pos_city_a=corrupt_meta["pos_city_a"],
                corrupt_pos_city_b=corrupt_meta["pos_city_b"],
                corrupt_pos_final=corrupt_meta["pos_final"],
            )
        )

    if not pairs:
        raise ValueError("No clean/corrupt pairs generated (no same-City_B opposite-direction matches).")

    print(f"[Info] Total clean/corrupt pairs (same City_B, opposite direction): {len(pairs)}")
    if skipped > 0:
        print(f"[Info] Skipped rows without same-City_B opposite sample: {skipped}")
    print(f"[Info] Layers: {n_layers}, Heads: {n_heads}, Batch size: {args.batch_size}")

    cache_names = [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)] + [
        f"blocks.{l}.attn.hook_z" for l in range(n_layers)
    ]
    cache_name_set = set(cache_names)

    def cache_filter(name: str) -> bool:
        return name in cache_name_set

    diff_sum_clean = 0.0
    diff_sum_corrupt = 0.0
    total_n = 0
    patched_diff_sum_mlp = {stage: torch.zeros(n_layers, dtype=torch.float64, device="cpu") for stage in STAGES}
    patched_diff_sum_attn = {stage: torch.zeros((n_layers, n_heads), dtype=torch.float64, device="cpu") for stage in STAGES}

    with torch.inference_mode():
        total_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
        for batch_pairs in tqdm(batched(pairs, args.batch_size), total=total_batches,
                                desc="Patching Batches", unit="batch"):
            clean_prompts = [x.clean_prompt for x in batch_pairs]
            corrupt_prompts = [x.corrupt_prompt for x in batch_pairs]

            clean_tokens = model.to_tokens(clean_prompts, prepend_bos=prepend_bos)
            corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=prepend_bos)

            clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=cache_filter, return_type="logits")
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=cache_filter, return_type="logits")

            clean_final = torch.tensor([x.clean_pos_final for x in batch_pairs], device=device, dtype=torch.long)
            corrupt_final = torch.tensor([x.corrupt_pos_final for x in batch_pairs], device=device, dtype=torch.long)
            clean_tgt = torch.tensor([x.clean_target_id for x in batch_pairs], device=device, dtype=torch.long)
            corrupt_tgt = torch.tensor([x.corrupt_target_id for x in batch_pairs], device=device, dtype=torch.long)

            clean_diff_vec = gather_logit_diff(clean_logits, clean_final, clean_tgt, corrupt_tgt)
            corrupt_diff_vec = gather_logit_diff(corrupt_logits, corrupt_final, clean_tgt, corrupt_tgt)
            diff_sum_clean += float(clean_diff_vec.sum().item())
            diff_sum_corrupt += float(corrupt_diff_vec.sum().item())
            total_n += len(batch_pairs)

            clean_pos_by_stage = {
                "city_a": torch.tensor([x.clean_pos_city_a for x in batch_pairs], device=device, dtype=torch.long),
                "city_b": torch.tensor([x.clean_pos_city_b for x in batch_pairs], device=device, dtype=torch.long),
                "final": clean_final,
            }
            corrupt_pos_by_stage = {
                "city_a": torch.tensor([x.corrupt_pos_city_a for x in batch_pairs], device=device, dtype=torch.long),
                "city_b": torch.tensor([x.corrupt_pos_city_b for x in batch_pairs], device=device, dtype=torch.long),
                "final": corrupt_final,
            }
            batch_idx = torch.arange(len(batch_pairs), device=device)

            for stage in STAGES:
                clean_pos = clean_pos_by_stage[stage]
                corrupt_pos = corrupt_pos_by_stage[stage]
                for layer in range(n_layers):
                    mlp_name = f"blocks.{layer}.hook_mlp_out"
                    corrupt_mlp = corrupt_cache[mlp_name]

                    def mlp_hook_fn(act, hook, _corrupt=corrupt_mlp, _cpos=clean_pos, _kpos=corrupt_pos, _bidx=batch_idx):
                        out = act.clone()
                        out[_bidx, _cpos, :] = _corrupt[_bidx, _kpos, :]
                        return out

                    patched_logits = model.run_with_hooks(clean_tokens, return_type="logits",
                                                           fwd_hooks=[(mlp_name, mlp_hook_fn)])
                    patched_diff_vec = gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)
                    patched_diff_sum_mlp[stage][layer] += patched_diff_vec.sum().double().cpu()

                    z_name = f"blocks.{layer}.attn.hook_z"
                    corrupt_z = corrupt_cache[z_name]
                    for head in range(n_heads):
                        def z_hook_fn(act, hook, _corrupt=corrupt_z, _cpos=clean_pos, _kpos=corrupt_pos, _bidx=batch_idx, _head=head):
                            out = act.clone()
                            out[_bidx, _cpos, _head, :] = _corrupt[_bidx, _kpos, _head, :]
                            return out

                        patched_logits = model.run_with_hooks(clean_tokens, return_type="logits",
                                                              fwd_hooks=[(z_name, z_hook_fn)])
                        patched_diff_vec = gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)
                        patched_diff_sum_attn[stage][layer, head] += patched_diff_vec.sum().double().cpu()

    clean_diff = diff_sum_clean / total_n
    corrupt_diff = diff_sum_corrupt / total_n
    denom = clean_diff - corrupt_diff
    if abs(denom) < 1e-8:
        raise ValueError("Clean_Diff and Corrupt_Diff are too close; metric denominator is unstable.")

    print("\n===== Baseline =====")
    print(f"Clean_Diff   : {clean_diff:.6f}")
    print(f"Corrupt_Diff : {corrupt_diff:.6f}")
    print(f"Denominator  : {denom:.6f}")

    metric_mlp = {}
    metric_attn = {}
    for stage in STAGES:
        patched_avg_mlp = (patched_diff_sum_mlp[stage] / total_n).to(torch.float32)
        patched_avg_attn = (patched_diff_sum_attn[stage] / total_n).to(torch.float32)
        metric_mlp[stage] = (patched_avg_mlp - corrupt_diff) / denom
        metric_attn[stage] = (patched_avg_attn - corrupt_diff) / denom

    stage_title = {
        "city_a": "City_A position (key in this run: probeable)",
        "city_b": "City_B position (no-op here: same City_B)",
        "final": "the position at the end",
    }
    stage_topk = {"city_a": 10, "city_b": 5, "final": 10}

    print("\n===== Causal Node Leaderboard (lower metric = more causal) =====")
    for stage in STAGES:
        topk = stage_topk[stage]
        ranked = topk_nodes(metric_mlp[stage], metric_attn[stage], topk=topk)
        print(f"\n[{stage_title[stage]}]Top-{topk}")
        for rank, (name, value) in enumerate(ranked, start=1):
            print(f"{rank:02d}. {name:<14} Metric={value:.6f}")

    # quick per-layer MLP summary for the key City_A stage
    print("\n===== City_A stage: per-layer MLP Metric (low = causal) =====")
    for layer in range(n_layers):
        print(f"  L{layer:02d}: {float(metric_mlp['city_a'][layer].item()):.4f}")

    save_patching_results(
        save_npz_path=Path(args.save_results_npz),
        save_meta_path=Path(args.save_results_meta),
        metric_mlp=metric_mlp,
        metric_attn=metric_attn,
        clean_diff=clean_diff,
        corrupt_diff=corrupt_diff,
        denom=denom,
        n_layers=n_layers,
        n_heads=n_heads,
        total_n=total_n,
        model_name=args.model_name,
        dataset_path=str(dataset_path),
    )
    print(f"\n[Info] Saved -> {args.save_results_npz}")
    print(f"[Info] Saved -> {args.save_results_meta}")


if __name__ == "__main__":
    main()
