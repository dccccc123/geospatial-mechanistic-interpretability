"""
Path patching to verify the L8-MLP -> L26-H24 causal path.

Activation patching only shows that a node is important; path patching injects
the clean activation at the sender output during a corrupt run so that only one
path is updated, isolating that path's causal contribution.

Four experiments:
  Exp-1  baseline: patch the L8-MLP output alone
  Exp-2  baseline: patch L26-H24 alone
  Exp-3  main:     L8-MLP[pos] -> L26-H24 (direct head path)
  Exp-4  main:     L8-MLP[pos] -> Residual@L26 (via the residual stream)

Token positions follow patching.py: city_a / city_b / final.
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = "Relative to {city_a}, {city_b} is to the"

OPPOSITE_DIRECTION = {
    "east": "west",
    "west": "east",
    "north": "south",
    "south": "north",
}

# Note: clean/corrupt pairs share the same city_a (only city_b differs), so the
# city_a activation is identical in both runs. Patching city_a is therefore a
# no-op (metric ~ 1.0); the city_a column carries no information and is kept
# only for debugging.
STAGES = ["city_a", "city_b", "final"]
VALID_STAGES = ["city_b", "final"]   # city_a is a no-op (clean == corrupt)

# Default nodes of interest (from the activation-patching results)
DEFAULT_SENDER_LAYER = 8       # L8-MLP (geographic knowledge)
DEFAULT_RECEIVER_LAYER = 26    # L26 (spatial computation)
DEFAULT_RECEIVER_HEAD = 24     # L26-H24 (spatial computation head)


# ─────────────────────────────────────────────────────────────────────────────
# model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name: str) -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        use_attn_result=False,
    )
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# data structures
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# data helpers
# ─────────────────────────────────────────────────────────────────────────────
def normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    alias = {"e": "east", "w": "west", "n": "north", "s": "south"}
    d = alias.get(d, d)
    if d not in OPPOSITE_DIRECTION:
        raise ValueError(f"Unsupported direction: {direction}")
    return d


def compute_positions(
    model: HookedTransformer, city_a: str, city_b: str, prepend_bos: bool
) -> dict:
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
        raise ValueError(
            f"Direction token '{token_text}' is not a single token: {token_ids.tolist()}"
        )
    return int(token_ids.item())


def batched(seq, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────
def gather_logit_diff(
    logits: torch.Tensor,
    final_positions: torch.Tensor,
    clean_target_ids: torch.Tensor,
    corrupt_target_ids: torch.Tensor,
) -> torch.Tensor:
    """clean_token_logit - corrupt_token_logit."""
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    clean_logits = logits[batch_idx, final_positions, clean_target_ids]
    corrupt_logits = logits[batch_idx, final_positions, corrupt_target_ids]
    return clean_logits - corrupt_logits


def normalize_metric(patched_diff: float, corrupt_diff: float, denom: float) -> float:
    """Normalized metric: 0 = fully corrupted, 1 = restored / no effect."""
    return (patched_diff - corrupt_diff) / denom


# ─────────────────────────────────────────────────────────────────────────────
# dataset construction
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs(
    model: HookedTransformer,
    dataset_path: Path,
    prepend_bos: bool,
    seed: int = 42,
) -> list[PairSample]:
    """
    Build clean/corrupt pairs (same logic as patching.py):
      clean:   (City_A, City_B, direction_d)
      corrupt: (City_A, City_C, opposite(direction_d))
    """
    random.seed(seed)
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
        raise ValueError("No valid rows found in dataset.")

    rows_by_city_a_and_direction: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        city_a = r["City_A"]
        direction = r["_direction_norm"]
        if city_a not in rows_by_city_a_and_direction:
            rows_by_city_a_and_direction[city_a] = {k: [] for k in OPPOSITE_DIRECTION}
        rows_by_city_a_and_direction[city_a][direction].append(r)

    token_id_by_direction = {d: direction_to_token_id(model, d) for d in OPPOSITE_DIRECTION}

    pairs: list[PairSample] = []
    skipped = 0
    for clean_row in rows:
        clean_city_a = clean_row["City_A"]
        clean_dir = clean_row["_direction_norm"]
        corrupt_dir = OPPOSITE_DIRECTION[clean_dir]

        corrupt_pool = (
            rows_by_city_a_and_direction.get(clean_city_a, {}).get(corrupt_dir, [])
        )
        if not corrupt_pool:
            skipped += 1
            continue

        corrupt_row = random.choice(corrupt_pool)
        corrupt_city_b = corrupt_row["City_B"]

        clean_meta = compute_positions(model, clean_city_a, clean_row["City_B"], prepend_bos)
        corrupt_meta = compute_positions(model, clean_city_a, corrupt_city_b, prepend_bos)

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

    print(f"[Data] Total valid pairs: {len(pairs)}  |  Skipped (no opposite sample): {skipped}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# baseline forward
# ─────────────────────────────────────────────────────────────────────────────
def run_with_cache_filtered(
    model: HookedTransformer,
    tokens: torch.Tensor,
    names_to_cache: list[str],
) -> tuple[torch.Tensor, dict]:
    cache_name_set = set(names_to_cache)
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name in cache_name_set,
        return_type="logits",
    )
    return logits, cache


# ─────────────────────────────────────────────────────────────────────────────
# four experiment functions
# ─────────────────────────────────────────────────────────────────────────────

def exp1_patch_sender_mlp(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_cache: dict,
    clean_pos: torch.Tensor,
    corrupt_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    clean_final: torch.Tensor,
    clean_tgt: torch.Tensor,
    corrupt_tgt: torch.Tensor,
    sender_layer: int,
) -> torch.Tensor:
    """
    Exp-1: activation-patch the L{sender_layer}-MLP output (reproduces patching.py).
    Replace the sender MLP output with the corrupt activation in a clean run.
    """
    mlp_name = f"blocks.{sender_layer}.hook_mlp_out"
    corrupt_mlp = corrupt_cache[mlp_name]

    def hook_fn(act, hook):
        out = act.clone()
        out[batch_idx, clean_pos, :] = corrupt_mlp[batch_idx, corrupt_pos, :]
        return out

    patched_logits = model.run_with_hooks(
        clean_tokens,
        return_type="logits",
        fwd_hooks=[(mlp_name, hook_fn)],
    )
    return gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)


def exp2_patch_receiver_head(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_cache: dict,
    clean_pos: torch.Tensor,
    corrupt_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    clean_final: torch.Tensor,
    clean_tgt: torch.Tensor,
    corrupt_tgt: torch.Tensor,
    receiver_layer: int,
    receiver_head: int,
) -> torch.Tensor:
    """
    Exp-2: activation-patch L{receiver_layer}-H{receiver_head} (reproduces patching.py).
    """
    z_name = f"blocks.{receiver_layer}.attn.hook_z"
    corrupt_z = corrupt_cache[z_name]

    def hook_fn(act, hook):
        out = act.clone()
        out[batch_idx, clean_pos, receiver_head, :] = (
            corrupt_z[batch_idx, corrupt_pos, receiver_head, :]
        )
        return out

    patched_logits = model.run_with_hooks(
        clean_tokens,
        return_type="logits",
        fwd_hooks=[(z_name, hook_fn)],
    )
    return gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)


def exp3_path_patch_mlp_to_head(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    clean_cache: dict,
    corrupt_cache: dict,
    clean_pos: torch.Tensor,
    corrupt_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    clean_final: torch.Tensor,
    clean_tgt: torch.Tensor,
    corrupt_tgt: torch.Tensor,
    sender_layer: int,
    receiver_layer: int,
    receiver_head: int,
) -> torch.Tensor:
    """
    Exp-3: path-patch L{sender_layer}-MLP -> L{receiver_layer}-H{receiver_head}.

    Two-stage algorithm: Stage A runs on corrupt tokens, injecting the clean
    sender-MLP output and caching the receiver head's z; Stage B runs on clean
    tokens, injecting that z at the receiver head. metric ~ 0 means the sender's
    information flows mainly through H{receiver_head}; metric ~ 1 means the path
    is weak and information travels elsewhere.
    """
    sender_mlp_name = f"blocks.{sender_layer}.hook_mlp_out"
    receiver_z_name = f"blocks.{receiver_layer}.attn.hook_z"
    clean_mlp = clean_cache[sender_mlp_name]

    # ── Stage A ──────────────────────────────────────────────────────────────
    stage_a_z_store: dict[str, torch.Tensor] = {}

    def sender_hook_a(act, hook):
        out = act.clone()
        out[batch_idx, corrupt_pos, :] = clean_mlp[batch_idx, clean_pos, :]
        return out

    def receiver_capture_hook(act, hook):
        stage_a_z_store["z"] = act.clone()
        return act

    model.run_with_hooks(
        corrupt_tokens,
        return_type=None,
        fwd_hooks=[
            (sender_mlp_name, sender_hook_a),
            (receiver_z_name, receiver_capture_hook),
        ],
    )

    stage_a_z = stage_a_z_store["z"]  # [B, seq, n_heads, d_head]

    # ── Stage B ──────────────────────────────────────────────────────────────
    def receiver_inject_hook(act, hook):
        out = act.clone()
        out[batch_idx, clean_pos, receiver_head, :] = (
            stage_a_z[batch_idx, corrupt_pos, receiver_head, :]
        )
        return out

    patched_logits = model.run_with_hooks(
        clean_tokens,
        return_type="logits",
        fwd_hooks=[(receiver_z_name, receiver_inject_hook)],
    )
    return gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)


def exp4_path_patch_mlp_to_resid(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    clean_cache: dict,
    corrupt_cache: dict,
    clean_pos: torch.Tensor,
    corrupt_pos: torch.Tensor,
    batch_idx: torch.Tensor,
    clean_final: torch.Tensor,
    clean_tgt: torch.Tensor,
    corrupt_tgt: torch.Tensor,
    sender_layer: int,
    receiver_layer: int,
) -> torch.Tensor:
    """
    Exp-4: path-patch L{sender_layer}-MLP -> the L{receiver_layer} residual stream.

    Two-stage algorithm: Stage A runs on corrupt tokens with the clean sender-MLP
    output injected, caching the receiver layer's residual-stream input; Stage B
    runs on clean tokens with that residual injected. Unlike Exp-3 (head-level),
    this replaces the whole layer input, so the two together reveal whether the
    information flows directly through H{receiver_head} or via the residual stream.
    """
    sender_mlp_name = f"blocks.{sender_layer}.hook_mlp_out"
    resid_pre_name = f"blocks.{receiver_layer}.hook_resid_pre"
    clean_mlp = clean_cache[sender_mlp_name]

    # ── Stage A ──────────────────────────────────────────────────────────────
    stage_a_resid_store: dict[str, torch.Tensor] = {}

    def sender_hook_a(act, hook):
        out = act.clone()
        out[batch_idx, corrupt_pos, :] = clean_mlp[batch_idx, clean_pos, :]
        return out

    def resid_capture_hook(act, hook):
        stage_a_resid_store["resid"] = act.clone()
        return act

    model.run_with_hooks(
        corrupt_tokens,
        return_type=None,
        fwd_hooks=[
            (sender_mlp_name, sender_hook_a),
            (resid_pre_name, resid_capture_hook),
        ],
    )

    stage_a_resid = stage_a_resid_store["resid"]  # [B, seq, d_model]

    # ── Stage B ──────────────────────────────────────────────────────────────
    def resid_inject_hook(act, hook):
        out = act.clone()
        out[batch_idx, clean_pos, :] = stage_a_resid[batch_idx, corrupt_pos, :]
        return out

    patched_logits = model.run_with_hooks(
        clean_tokens,
        return_type="logits",
        fwd_hooks=[(resid_pre_name, resid_inject_hook)],
    )
    return gather_logit_diff(patched_logits, clean_final, clean_tgt, corrupt_tgt)


# ─────────────────────────────────────────────────────────────────────────────
# results printing and saving
# ─────────────────────────────────────────────────────────────────────────────
def print_results(
    results: dict,
    clean_diff: float,
    corrupt_diff: float,
    sender_layer: int,
    receiver_layer: int,
    receiver_head: int,
) -> None:
    exp_labels = {
        "exp1_patch_sender":      f"Exp-1  Baseline : Patch L{sender_layer}-MLP        (activation patch)",
        "exp2_patch_receiver":    f"Exp-2  Baseline : Patch L{receiver_layer}-H{receiver_head}       (activation patch)",
        "exp3_path_mlp_to_head":  f"Exp-3  Path Patch: L{sender_layer}-MLP → L{receiver_layer}-H{receiver_head}",
        "exp4_path_mlp_to_resid": f"Exp-4  Path Patch: L{sender_layer}-MLP → Resid@L{receiver_layer}",
    }

    print("\n" + "=" * 82)
    print("  PATH PATCHING RESULTS       metric: 0.0 = fully corrupted  |  1.0 = clean/no effect")
    print(f"  Clean_Diff  : {clean_diff:+.4f}    Corrupt_Diff : {corrupt_diff:+.4f}")
    print("=" * 82)
    print(f"  {'Experiment':<52}  {'city_a*':>8}  {'city_b':>8}  {'final':>8}")
    print(f"  {'':52}  {'[invalid]':>8}")
    print("  " + "-" * 78)

    for exp_name, label in exp_labels.items():
        if exp_name not in results:
            continue
        row_parts = []
        for stage in STAGES:
            val = results[exp_name].get(stage, {}).get("metric", float("nan"))
            # city_a column: clean == corrupt, patch is a no-op
            if stage == "city_a":
                row_parts.append(f"{'('+f'{val:.2f}'+')':>8}")
            else:
                row_parts.append(f"{val:>8.4f}")
        print(f"  {label:<52}  {'  '.join(row_parts)}")

    print("=" * 82)
    print("  * city_a column: clean/corrupt share the same city_a, activations are identical, patch is a no-op,")
    print("    metric=1.0 is a mathematical necessity with no mechanistic meaning; ignore this column.")
    print("  valid columns: city_b (city_b token position), final (end-of-sequence position)")
    print("\n  ╔══ path contribution interpretation (valid columns only) ══════════════════════════════════════╗")
    print(f"  ║  Exp-3@final low -> L{sender_layer}-MLP signal flows mostly via H{receiver_head} to the output    ║")
    print(f"  ║  Exp-4@final low -> L{sender_layer}-MLP signal is injected into the residual stream at L{receiver_layer} ║")
    print(f"  ║  Exp-3@final ~= Exp-2@final -> path is intact, H{receiver_head} is the main consumer      ║")
    print(f"  ║  Exp-3@city_b low -> H{receiver_head} reads the L{sender_layer}-MLP encoding at city_b  ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝\n")


def save_results(
    save_path: Path,
    results: dict,
    clean_diff: float,
    corrupt_diff: float,
    denom: float,
    total_n: int,
    model_name: str,
    dataset_path: str,
    sender_layer: int,
    receiver_layer: int,
    receiver_head: int,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model_name": model_name,
        "dataset_path": dataset_path,
        "total_pairs": total_n,
        "sender_layer": sender_layer,
        "receiver_layer": receiver_layer,
        "receiver_head": receiver_head,
        "clean_diff": clean_diff,
        "corrupt_diff": corrupt_diff,
        "denominator": denom,
        "stages": STAGES,
        "experiments": results,
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Info] Results saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Path Patching: verify the L{sender}-MLP -> L{receiver}-H{head} causal path"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data_process/dataset/balanced_city_pairs.jsonl",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="path patching runs several forwards per step and uses more memory; <=8 recommended"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sender-layer", type=int, default=DEFAULT_SENDER_LAYER,
        help=f"information sender node: MLP layer (default {DEFAULT_SENDER_LAYER})"
    )
    parser.add_argument(
        "--receiver-layer", type=int, default=DEFAULT_RECEIVER_LAYER,
        help=f"information receiver layer (default {DEFAULT_RECEIVER_LAYER})"
    )
    parser.add_argument(
        "--receiver-head", type=int, default=DEFAULT_RECEIVER_HEAD,
        help=f"information receiver attention head index (default {DEFAULT_RECEIVER_HEAD})"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="component_localization/path_patching_results.json",
    )
    parser.add_argument(
        "--skip-exp3", action="store_true",
        help="skip Exp-3 (path patch to Head) to save memory"
    )
    parser.add_argument(
        "--skip-exp4", action="store_true",
        help="skip Exp-4 (path patch to ResidPre) to save memory"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"[Info] Loading model : {args.model_name}")
    model = load_model(args.model_name)
    prepend_bos = model.cfg.default_prepend_bos
    device = model.cfg.device
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    print(f"[Info] n_layers={n_layers}, n_heads={n_heads}, device={device}")

    # ── build dataset ─────────────────────────────────────────────────────────
    print(f"[Info] Building pairs : {dataset_path}")
    pairs = build_pairs(model, dataset_path, prepend_bos, seed=args.seed)

    sender_layer = args.sender_layer
    receiver_layer = args.receiver_layer
    receiver_head = args.receiver_head

    # hook names
    sender_mlp_name = f"blocks.{sender_layer}.hook_mlp_out"
    receiver_z_name = f"blocks.{receiver_layer}.attn.hook_z"
    receiver_resid_pre_name = f"blocks.{receiver_layer}.hook_resid_pre"
    names_to_cache = [sender_mlp_name, receiver_z_name, receiver_resid_pre_name]

    # ── determine experiment list ─────────────────────────────────────────────
    exp_names = ["exp1_patch_sender", "exp2_patch_receiver"]
    if not args.skip_exp3:
        exp_names.append("exp3_path_mlp_to_head")
    if not args.skip_exp4:
        exp_names.append("exp4_path_mlp_to_resid")

    # ── init accumulators ─────────────────────────────────────────────────────
    diff_sum_clean = 0.0
    diff_sum_corrupt = 0.0
    total_n = 0
    sums: dict[str, dict[str, float]] = {
        exp: {stage: 0.0 for stage in STAGES} for exp in exp_names
    }

    # ── main loop ─────────────────────────────────────────────────────────────
    total_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    print(
        f"\n[Info] Experiments to run : {exp_names}"
        f"\n[Info] Sender  : L{sender_layer}-MLP"
        f"\n[Info] Receiver: L{receiver_layer}-H{receiver_head}"
        f"\n[Info] Pairs   : {len(pairs)}  |  Batch size : {args.batch_size}\n"
    )

    with torch.inference_mode():
        for batch_pairs in tqdm(
            batched(pairs, args.batch_size),
            total=total_batches,
            desc="Path Patching",
            unit="batch",
        ):
            bsz = len(batch_pairs)

            clean_prompts = [x.clean_prompt for x in batch_pairs]
            corrupt_prompts = [x.corrupt_prompt for x in batch_pairs]
            clean_tokens = model.to_tokens(clean_prompts, prepend_bos=prepend_bos)
            corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=prepend_bos)

            # baseline forward (with activation cache)
            clean_logits, clean_cache = run_with_cache_filtered(
                model, clean_tokens, names_to_cache
            )
            corrupt_logits, corrupt_cache = run_with_cache_filtered(
                model, corrupt_tokens, names_to_cache
            )

            # position tensors
            clean_final = torch.tensor(
                [x.clean_pos_final for x in batch_pairs], device=device, dtype=torch.long
            )
            corrupt_final = torch.tensor(
                [x.corrupt_pos_final for x in batch_pairs], device=device, dtype=torch.long
            )
            clean_tgt = torch.tensor(
                [x.clean_target_id for x in batch_pairs], device=device, dtype=torch.long
            )
            corrupt_tgt = torch.tensor(
                [x.corrupt_target_id for x in batch_pairs], device=device, dtype=torch.long
            )
            batch_idx = torch.arange(bsz, device=device, dtype=torch.long)

            clean_pos_by_stage = {
                "city_a": torch.tensor([x.clean_pos_city_a for x in batch_pairs], device=device, dtype=torch.long),
                "city_b": torch.tensor([x.clean_pos_city_b for x in batch_pairs], device=device, dtype=torch.long),
                "final":  clean_final,
            }
            corrupt_pos_by_stage = {
                "city_a": torch.tensor([x.corrupt_pos_city_a for x in batch_pairs], device=device, dtype=torch.long),
                "city_b": torch.tensor([x.corrupt_pos_city_b for x in batch_pairs], device=device, dtype=torch.long),
                "final":  corrupt_final,
            }

            # accumulate baseline logit diffs
            clean_diff_vec = gather_logit_diff(clean_logits, clean_final, clean_tgt, corrupt_tgt)
            corrupt_diff_vec = gather_logit_diff(corrupt_logits, corrupt_final, clean_tgt, corrupt_tgt)
            diff_sum_clean += float(clean_diff_vec.sum().item())
            diff_sum_corrupt += float(corrupt_diff_vec.sum().item())
            total_n += bsz

            # run each experiment per stage
            for stage in STAGES:
                clean_pos = clean_pos_by_stage[stage]
                corrupt_pos = corrupt_pos_by_stage[stage]

                # Exp-1
                d = exp1_patch_sender_mlp(
                    model, clean_tokens, corrupt_cache,
                    clean_pos, corrupt_pos, batch_idx,
                    clean_final, clean_tgt, corrupt_tgt,
                    sender_layer,
                )
                sums["exp1_patch_sender"][stage] += float(d.sum().item())

                # Exp-2
                d = exp2_patch_receiver_head(
                    model, clean_tokens, corrupt_cache,
                    clean_pos, corrupt_pos, batch_idx,
                    clean_final, clean_tgt, corrupt_tgt,
                    receiver_layer, receiver_head,
                )
                sums["exp2_patch_receiver"][stage] += float(d.sum().item())

                # Exp-3
                if not args.skip_exp3:
                    d = exp3_path_patch_mlp_to_head(
                        model, clean_tokens, corrupt_tokens,
                        clean_cache, corrupt_cache,
                        clean_pos, corrupt_pos, batch_idx,
                        clean_final, clean_tgt, corrupt_tgt,
                        sender_layer, receiver_layer, receiver_head,
                    )
                    sums["exp3_path_mlp_to_head"][stage] += float(d.sum().item())

                # Exp-4
                if not args.skip_exp4:
                    d = exp4_path_patch_mlp_to_resid(
                        model, clean_tokens, corrupt_tokens,
                        clean_cache, corrupt_cache,
                        clean_pos, corrupt_pos, batch_idx,
                        clean_final, clean_tgt, corrupt_tgt,
                        sender_layer, receiver_layer,
                    )
                    sums["exp4_path_mlp_to_resid"][stage] += float(d.sum().item())

    # ── normalize ─────────────────────────────────────────────────────────────
    clean_diff = diff_sum_clean / total_n
    corrupt_diff = diff_sum_corrupt / total_n
    denom = clean_diff - corrupt_diff
    if abs(denom) < 1e-8:
        raise ValueError("Denominator too small (clean_diff ≈ corrupt_diff).")

    print(
        f"\n[Baseline] Clean_Diff={clean_diff:+.6f}  "
        f"Corrupt_Diff={corrupt_diff:+.6f}  Denom={denom:.6f}"
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for exp_name in exp_names:
        results[exp_name] = {}
        for stage in STAGES:
            mean_diff = sums[exp_name][stage] / total_n
            metric = normalize_metric(mean_diff, corrupt_diff, denom)
            results[exp_name][stage] = {"mean_diff": round(mean_diff, 6), "metric": round(metric, 6)}

    # ── print & save ──────────────────────────────────────────────────────────
    print_results(results, clean_diff, corrupt_diff, sender_layer, receiver_layer, receiver_head)

    save_results(
        save_path=Path(args.save_results),
        results=results,
        clean_diff=clean_diff,
        corrupt_diff=corrupt_diff,
        denom=denom,
        total_n=total_n,
        model_name=args.model_name,
        dataset_path=str(dataset_path),
        sender_layer=sender_layer,
        receiver_layer=receiver_layer,
        receiver_head=receiver_head,
    )


if __name__ == "__main__":
    main()
