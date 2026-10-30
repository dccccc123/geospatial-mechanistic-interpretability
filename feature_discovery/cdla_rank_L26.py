"""
Rank all L26 features by contrastive DLA, with top-1 ranking-stability bootstrap.

cDLA(f, d) = mean_act_d[f] * (W_dec[f] . W_U[d])
             - mean_{d' != d}[ mean_act_{d'}[f] * (W_dec[f] . W_U[d']) ].

For each direction this reports the top-1 / top-2 feature ids and their cDLA gap,
then bootstraps the direction's prompts (with replacement) to estimate the
stability of the top-1 ranking.
"""
import os, json, gc
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent
SAE_DIR = ROOT / "sae" / "final_sae" / "layer_26"
THE_ACTS = ROOT / "sae" / "activations" / "the_sae_acts_l26.npy"
PAIRS = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
OUT = ROOT / "feature_discovery" / "results" / "cdla_rank_stability.json"
MODEL = "Qwen/Qwen3-8B"
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
TARGET_LAYER = 26
OUTLIER_DIMS = [2276, 233]
DIR_TOK = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}
DIRS = ["East", "West", "South", "North"]
EXPECT = {"North": 26800, "South": 10801, "West": 56725, "East": 47636}
MIN_ACTIVE = 0.10
B_BOOT = 1000
SEED = 42


def compute_or_load_the_sae_acts():
    """Load cached L26 the-position SAE activations [N, d_sae]; generate if missing."""
    if THE_ACTS.exists():
        acts = np.load(str(THE_ACTS))
        n_rows = sum(1 for l in open(PAIRS) if l.strip())
        if acts.shape[0] >= n_rows:
            print(f"[Cache] loading L26 the SAE activations: {THE_ACTS}, shape={acts.shape}")
            return acts.astype(np.float32)
        print(f"[Cache] stale cache ({acts.shape[0]} < {n_rows}); regenerating.")

    print("[Cache] computing L26 the SAE activations (forward + encode) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    sae = SAE.load_from_disk(str(SAE_DIR), device="cuda")
    sae.eval()

    rows = [json.loads(l) for l in open(PAIRS) if l.strip()]
    all_acts = []
    for row in tqdm(rows, desc="[extract L26 the SAE acts]", unit="prompt"):
        prompt = PROMPT_TPL.format(city_a=row["City_A"], city_b=row["City_B"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        positions = (inputs.input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise RuntimeError(f"THE_TOKEN_ID={THE_TOKEN_ID} not found in prompt: {prompt}")
        the_pos = int(positions[-1].item())
        captured = {}

        def hook(_m, _i, output):
            hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
            hidden[:, :, OUTLIER_DIMS] = 0.0
            captured["h"] = hidden[:, the_pos, :].detach()
            return None

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(**inputs)
        finally:
            handle.remove()

        with torch.inference_mode():
            feats = sae.encode(captured["h"].to(device="cuda", dtype=torch.bfloat16))
        all_acts.append(feats.squeeze(0).detach().float().cpu().numpy())

    acts = np.stack(all_acts, axis=0).astype(np.float32)
    THE_ACTS.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(THE_ACTS), acts)
    print(f"[Cache] saved L26 the SAE activations: {THE_ACTS}, shape={acts.shape}")

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()
    return acts


def main():
    acts = compute_or_load_the_sae_acts()                      # [180, 65536]
    dirs = [json.loads(l)["Direction"] for l in open(PAIRS) if l.strip()]
    dir_idx = {d: [i for i, x in enumerate(dirs) if x == d] for d in DIRS}

    raw = load_file(str(SAE_DIR / "sae_weights.safetensors"))
    W_dec = raw.get("sae.W_dec", raw.get("W_dec")).float()     # [d_sae, d_model]

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu",
                                                 trust_remote_code=True, local_files_only=True)
    W_U = model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]
    u = {d: W_U[DIR_TOK[d]] for d in DIRS}
    del model

    proj = {d: (W_dec @ u[d]).numpy() for d in DIRS}           # [d_sae]
    means = {d: acts[dir_idx[d]].mean(0) for d in DIRS}
    dla = {d: means[d] * proj[d] for d in DIRS}                # [d_sae]

    stability = {}
    for d in DIRS:
        others = np.stack([dla[o] for o in DIRS if o != d]).mean(0)
        cdla = dla[d] - others
        active_ratio = (acts[dir_idx[d]] > 0).mean(0)
        order = np.argsort(-cdla)
        top1, top2 = int(order[0]), int(order[1])
        gap = float(cdla[top1] - cdla[top2])
        exp = EXPECT[d]

        print(f"\n=== {d}  (expected top1 = #{exp}) ===")
        print(f"  top-1 = #{top1} (cDLA={cdla[top1]:.3f})")
        print(f"  top-2 = #{top2} (cDLA={cdla[top2]:.3f})")
        print(f"  gap(top1 - top2) = {gap:+.3f}")
        print("  pure cDLA (no gate) top5:", [(int(f), round(float(cdla[f]), 3)) for f in order[:5]])
        masked = np.where(active_ratio >= MIN_ACTIVE, cdla, -np.inf)
        print("  cDLA + 10% gate top5:", [(int(f), round(float(cdla[f]), 3)) for f in np.argsort(-masked)[:5]])
        rank_raw = int(np.where(order == exp)[0][0]) + 1
        print(f"  #{exp} pure-cDLA rank={rank_raw}; cDLA={cdla[exp]:.3f}; active ratio={active_ratio[exp]:.2f}")

        stability[d] = bootstrap_stability(acts[dir_idx[d]], proj[d], others, exp, top2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(stability, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


def bootstrap_stability(acts_dir, proj_d, others_full, exp, top2):
    """Within-direction bootstrap: resample the direction's prompts to estimate top-1 ranking stability."""
    n = acts_dir.shape[0]
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(n, [1.0 / n] * n, size=B_BOOT).astype(np.float32)  # [B, n]
    mean_boot = (counts @ acts_dir) / n                       # [B, d_sae]
    cdla_boot = mean_boot * proj_d - others_full              # [B, d_sae]

    rank_exp = (cdla_boot > cdla_boot[:, exp][:, None]).sum(axis=1) + 1
    overtaken = float((cdla_boot[:, top2] > cdla_boot[:, exp]).mean())

    p_top1 = float((rank_exp == 1).mean())
    p_le2 = float((rank_exp <= 2).mean())
    median_rank = float(np.median(rank_exp))
    mean_rank = float(rank_exp.mean())

    print(f"  [bootstrap B={B_BOOT}] #{exp} ranking stability: P(#1)={p_top1*100:.1f}%, "
          f"P(<=2)={p_le2*100:.1f}%, median rank={median_rank:.2f}, mean rank={mean_rank:.2f}; "
          f"overtaken by #{top2}: {overtaken*100:.1f}%")

    return {
        "top1_feature": int(exp),
        "top2_feature": int(top2),
        "n_prompts": int(n),
        "n_bootstrap": int(B_BOOT),
        "seed": int(SEED),
        "p_rank_1": p_top1,
        "p_rank_le2": p_le2,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "p_overtaken_by_top2": overtaken,
        "rank_distribution": [int((rank_exp == k).sum()) for k in range(1, 11)],
    }


if __name__ == "__main__":
    main()
