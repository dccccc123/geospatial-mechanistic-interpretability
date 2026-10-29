#!/usr/bin/env python3
"""
Diagnose a newly trained L26 SAE against the reference SAE.

1. Reconstruction quality (explained variance, cosine similarity, MSE, L0).
2. Direction-feature classification: for the new SAE's cDLA top-k features,
   label each as CLEAN (decoder aligned with its own direction token and
   activation direction-selective), ARTIFACT (decoder near-orthogonal to all
   four directions with constant high activation), or AMBIGUOUS. The reference
   SAE's six known features are classified with the same criteria as a check.
3. Optional (--ablate) reconstruction ablation of the top-1 cDLA feature and the
   best CLEAN candidate, reporting delta logit(pred) and flip rate.

Generates a cache of raw L26 "the" hidden states once, then reuses it.
"""
import argparse
import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent

# ── constants (same as the reference script) ──
MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 26
PROMPT_TPL = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID = 279
OUTLIER_DIMS = [2276, 233]
DIRS = ["East", "West", "South", "North"]
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}

# Reference SAE's known features (calibration): 4 clean top-1 + 2 top-2 artifacts
REF_FEATS = {
    "East_top1": 47636,
    "West_top1": 56725,
    "South_top1": 10801,
    "North_top1": 26800,
    "East/North_top2_artifact": 18707,
    "West/South_top2_artifact": 38915,
}

# ── diagnosis thresholds (tunable) ──
MIN_ACTIVE = 0.05       # dead-feature gate: exclude features below this per-direction active ratio
PROJ_OWN_MIN = 0.1      # CLEAN: min projection onto the feature's own direction
PROJ_MARGIN_MIN = 0.1   # CLEAN: margin of proj_own over the other directions' max proj
ACT_SEL_RATIO = 1.5     # CLEAN: min ratio act_own / max(act_other)
ARTIFACT_MAX_PROJ = 0.5   # ARTIFACT: decoder |proj| below this for all four directions
ARTIFACT_ACT_FLOOR = 0.3  # ARTIFACT: min ratio of other-direction mean activation to own activation


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose a newly trained L26 SAE")
    p.add_argument("--new-sae", type=str, default=None,
                   help="New SAE directory (default sae/final_sae_seed1234_v2/layer_26)")
    p.add_argument("--orig-sae", type=str, default=None,
                   help="Reference SAE directory (default sae/final_sae/layer_26)")
    p.add_argument("--tag", type=str, default="seed1234_v2",
                   help="Suffix tag for cache/output files, distinguishing different seeds")
    p.add_argument("--dataset", type=str, default="train", choices=["train", "holdout"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Run only the first N prompts (smoke test)")
    p.add_argument("--ablate", action="store_true", help="Ablate the top-1 and best CLEAN features")
    p.add_argument("--no-save-acts", action="store_true", help="Do not save activation caches")
    return p.parse_args()


def resolve_paths(args):
    if args.new_sae is None:
        args.new_sae = str(ROOT / "sae" / "final_sae_seed1234_v2" / "layer_26")
    if args.orig_sae is None:
        args.orig_sae = str(ROOT / "sae" / "final_sae" / "layer_26")
    if args.dataset == "holdout":
        pairs = ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl"
    else:
        pairs = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
    hidden_npy = ROOT / "sae" / "activations" / f"the_l26_hidden_{args.dataset}.npy"
    acts_new_npy = ROOT / "sae" / "activations" / f"the_sae_acts_l26_{args.tag}.npy"
    out = ROOT / "feature_discovery" / "results" / f"diagnose_seed_sae_{args.tag}.json"
    return Path(args.new_sae), Path(args.orig_sae), pairs, hidden_npy, acts_new_npy, out


def load_rows(data_path: Path) -> List[Dict]:
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
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
        raise RuntimeError("THE_TOKEN_ID=279 not found; check the prompt/tokenizer.")
    return int(positions[-1].item())


def model_predict_from_logits(logits: torch.Tensor) -> str:
    ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    sn = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(ew) >= abs(sn):
        return "East" if ew > 0 else "West"
    return "South" if sn > 0 else "North"


def dir_mean(acts: np.ndarray, idx: List[int]) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(acts.shape[1], dtype=np.float32)
    return acts[idx].mean(0).astype(np.float32)


def dir_active_ratio(acts: np.ndarray, idx: List[int]) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(acts.shape[1], dtype=np.float32)
    return (acts[idx] > 0).mean(0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
def load_model_and_saes(new_sae: Path, orig_sae: Path):
    print("[Model] Loading tokenizer / model / two SAEs ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    sae_new = SAE.load_from_disk(str(new_sae), device="cuda")
    sae_new.eval()
    sae_orig = SAE.load_from_disk(str(orig_sae), device="cuda")
    sae_orig.eval()
    u = {
        d: model.get_output_embeddings().weight[DIR_TOKEN_IDS[d]].detach().float().cpu()
        for d in DIRS
    }
    print(f"[Model] Done. new_sae={new_sae}, orig_sae={orig_sae}")
    return model, tokenizer, sae_new, sae_orig, u


def compute_hidden_and_acts(model, tokenizer, rows, sae_new, hidden_npy, acts_new_npy, args):
    """L26 "the" position: raw hidden (outlier dims zeroed) + new-SAE encoded features."""
    if hidden_npy.exists() and acts_new_npy.exists():
        X = np.load(str(hidden_npy)).astype(np.float32)
        acts_new = np.load(str(acts_new_npy)).astype(np.float32)
        if X.shape[0] >= len(rows) and X.ndim == 2 and acts_new.ndim == 2:
            print(f"[Cache] Reusing hidden={X.shape} / acts_new={acts_new.shape}")
            return X, acts_new
        print(f"[Cache] Old cache is malformed or too small (hidden={X.shape}, acts_new={acts_new.shape}); recomputing.")

    Xs, acts_new_list = [], []
    for row in tqdm(rows, desc="[forward + encode new SAE]", unit="prompt"):
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        the_pos = get_the_pos(inputs.input_ids)
        captured = {}

        def hook(_m, _i, output):
            hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
            hidden[:, :, OUTLIER_DIMS] = 0.0
            captured["h"] = hidden[:, the_pos, :].detach()
            return None

        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                _ = model(**inputs)
        finally:
            handle.remove()

        vec = captured["h"].to(device="cuda", dtype=torch.bfloat16)
        Xs.append(captured["h"].float().cpu().squeeze(0))
        with torch.inference_mode():
            acts_new_list.append(sae_new.encode(vec).squeeze(0).float().cpu().numpy())

    X = np.stack(Xs).astype(np.float32)
    acts_new = np.stack(acts_new_list).astype(np.float32)

    if not args.no_save_acts:
        hidden_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(hidden_npy), X)
        np.save(str(acts_new_npy), acts_new)
        print(f"[Cache] Saved hidden={X.shape} -> {hidden_npy}")
        print(f"[Cache] Saved acts_new={acts_new.shape} -> {acts_new_npy}")

    return X, acts_new


def encode_acts(sae, X: np.ndarray) -> np.ndarray:
    """Encode raw hidden with a given SAE, returning [N, d_sae]."""
    acts = []
    Xt = torch.from_numpy(X)
    for i in range(X.shape[0]):
        vec = Xt[i].to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        with torch.inference_mode():
            acts.append(sae.encode(vec).squeeze(0).float().cpu().numpy())
    return np.stack(acts).astype(np.float32)


def recon_metrics(X: np.ndarray, sae: SAE, name: str) -> Dict:
    """Reconstruction quality: EV / cossim / MSE / L0 of encode->decode vs raw hidden."""
    xhats, l0s = [], []
    Xt = torch.from_numpy(X)
    for i in tqdm(range(X.shape[0]), desc=f"[recon {name}]", unit="prompt", leave=False):
        vec = Xt[i].to(device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        with torch.inference_mode():
            feats = sae.encode(vec)
            l0s.append(int((feats > 0).sum().item()))
            xhats.append(sae.decode(feats).squeeze(0).float().cpu())
    Xh = torch.stack(xhats)
    per_ev = (1.0 - ((Xt - Xh) ** 2).sum(1) / ((Xt) ** 2).sum(1)).numpy()
    return {
        "explained_variance_mean": float(per_ev.mean()),
        "explained_variance_median": float(np.median(per_ev)),
        "mse": float(((Xt - Xh) ** 2).mean().item()),
        "cossim_mean": float(F.cosine_similarity(Xt, Xh, dim=1).mean().item()),
        "l0_mean": float(np.mean(l0s)),
        "l0_median": float(np.median(l0s)),
    }


def load_w_dec(sae_dir: Path) -> torch.Tensor:
    raw = load_file(str(sae_dir / "sae_weights.safetensors"))
    w = raw.get("sae.W_dec", raw.get("W_dec"))
    if w is None:
        raise KeyError(f"No W_dec found in {sae_dir} weights")
    return w.float().cpu()


def classify(proj_vec: Dict[str, float], act_vec: Dict[str, float], d: str) -> str:
    proj_own = proj_vec[d]
    proj_other_max = max(proj_vec[o] for o in DIRS if o != d)
    act_own = act_vec[d]
    act_other_max = max(act_vec[o] for o in DIRS if o != d)
    act_other_mean = sum(act_vec[o] for o in DIRS if o != d) / 3.0
    max_abs_proj = max(abs(proj_vec[dd]) for dd in DIRS)

    is_clean = (
        proj_own > PROJ_OWN_MIN
        and proj_own - proj_other_max > PROJ_MARGIN_MIN
        and act_own > MIN_ACTIVE
        and act_own > ACT_SEL_RATIO * max(act_other_max, 1e-9)
    )
    is_artifact = (
        max_abs_proj < ARTIFACT_MAX_PROJ
        and act_own > MIN_ACTIVE
        and act_other_mean > ARTIFACT_ACT_FLOOR * act_own
    )
    if is_clean:
        return "CLEAN"
    if is_artifact:
        return "ARTIFACT"
    return "AMBIGUOUS"


def _fmt_vec(v: Dict[str, float]) -> str:
    return " ".join(f"{v[d]:+8.2f}" for d in DIRS)


def diagnose_direction(acts: np.ndarray, w_dec: torch.Tensor, u: Dict[str, torch.Tensor],
                       dir_idx: Dict[str, List[int]], top_k: int) -> Dict:
    proj = {d: (w_dec @ u[d]).numpy() for d in DIRS}
    means = {d: dir_mean(acts, dir_idx[d]) for d in DIRS}
    dla = {d: means[d] * proj[d] for d in DIRS}

    result = {}
    for d in DIRS:
        others = np.stack([dla[o] for o in DIRS if o != d]).mean(0)
        cdla = dla[d] - others
        active_ratio = dir_active_ratio(acts, dir_idx[d])
        masked = np.where(active_ratio >= MIN_ACTIVE, cdla, -np.inf)
        order = np.argsort(-masked)

        print(f"\n=== {d}: cDLA top-{top_k} feature diagnosis (new SAE) ===")
        print(f"{'rank':>4} {'fid':>7} {'cDLA':>7} {'proj(E/W/S/N)':>34} {'act(E/W/S/N)':>30}  label")
        top_rows, clean_hits = [], []
        for r in range(top_k):
            f = int(order[r])
            pv = {dd: float(proj[dd][f]) for dd in DIRS}
            av = {dd: float(means[dd][f]) for dd in DIRS}
            lab = classify(pv, av, d)
            if lab == "CLEAN":
                clean_hits.append((r + 1, f, float(cdla[f])))
            print(f"{r+1:>4} #{f:>6} {cdla[f]:+7.3f} {_fmt_vec(pv):>34} {_fmt_vec(av):>30}  {lab}")
            top_rows.append({"rank": r + 1, "feature": f, "cdla": float(cdla[f]),
                             "proj": pv, "act": av, "label": lab})
        if clean_hits:
            best = clean_hits[0]
            print(f"  >> {d} found {len(clean_hits)} CLEAN direction features in the top-{top_k}, "
                  f"best = #{best[1]} (rank={best[0]}, cDLA={best[2]:.3f})")
        else:
            print(f"  >> {d} has no CLEAN direction feature in the top-{top_k} (possibly weak / lower-ranked)")
        result[d] = {
            "top_k": top_k,
            "top1_feature": int(order[0]),
            "top1_cdla": float(cdla[order[0]]),
            "n_clean_in_topk": len(clean_hits),
            "best_clean": ({"rank": clean_hits[0][0], "feature": clean_hits[0][1],
                            "cdla": clean_hits[0][2]} if clean_hits else None),
            "rows": top_rows,
        }
    return result


def calibration(acts_orig: np.ndarray, w_dec_orig: torch.Tensor, u: Dict[str, torch.Tensor],
                dir_idx: Dict[str, List[int]]):
    """Calibrate using the reference SAE's six known features with the same criteria."""
    print("\n=== Calibration: reference SAE known features (same proj/act/label criteria) ===")
    proj = {d: (w_dec_orig @ u[d]).numpy() for d in DIRS}
    means = {d: dir_mean(acts_orig, dir_idx[d]) for d in DIRS}
    print(f"{'feature':>24} {'fid':>7} {'proj(E/W/S/N)':>34} {'act(E/W/S/N)':>30}  label")
    out = {}
    for name, f in REF_FEATS.items():
        pv = {dd: float(proj[dd][f]) for dd in DIRS}
        av = {dd: float(means[dd][f]) for dd in DIRS}
        dom = max(DIRS, key=lambda dd: pv[dd])
        lab = classify(pv, av, dom)
        print(f"{name:>24} #{f:>6} {_fmt_vec(pv):>34} {_fmt_vec(av):>30}  {lab}")
        out[name] = {"feature": f, "proj": pv, "act": av, "label": lab,
                     "dominant_dir": dom}
    return out


# ─────────────────────────────────────────────────────────────
def make_l26_sae_hook(sae, the_pos, ablate_feature):
    def hook(_m, _i, output):
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


def run_l26_replacement(model, tokenizer, sae, prompt, ablate_feature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    the_pos = get_the_pos(inputs.input_ids)
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(
        make_l26_sae_hook(sae, the_pos, ablate_feature))
    try:
        with torch.inference_mode():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def run_forward(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    return out.logits[0, -1].detach().float().cpu()


def ablate_new_sae(model, tokenizer, sae_new, rows, diag, limit):
    print("\n=== Ablation: top-1 cDLA vs best CLEAN (new SAE, reconstruction + recon baseline) ===")
    per_dir = {
        d: {
            "top1": diag[d]["top1_feature"],
            "best_clean": (diag[d]["best_clean"]["feature"] if diag[d]["best_clean"] else None),
        }
        for d in DIRS
    }
    for d in DIRS:
        print(f"  {d}: top1=#{per_dir[d]['top1']}, best_clean=#{per_dir[d]['best_clean']}")

    recs = []
    for idx, row in enumerate(tqdm(rows[:limit] if limit else rows,
                                   desc="[ablation]", unit="prompt")):
        prompt = row["prompt"]
        clean = run_forward(model, tokenizer, prompt)
        pred = model_predict_from_logits(clean)
        recon = run_l26_replacement(model, tokenizer, sae_new, prompt, None)
        rec = {"pred_dir": pred,
               "recon_logit_pred": float(recon[DIR_TOKEN_IDS[pred]].item())}
        for tag, fid in [("top1", per_dir[pred]["top1"]),
                         ("best_clean", per_dir[pred]["best_clean"])]:
            if fid is None:
                rec[f"delta_{tag}"] = None
                rec[f"flip_{tag}"] = None
                continue
            out = run_l26_replacement(model, tokenizer, sae_new, prompt, fid)
            rec[f"delta_{tag}"] = float(out[DIR_TOKEN_IDS[pred]].item()
                                        - recon[DIR_TOKEN_IDS[pred]].item())
            rec[f"flip_{tag}"] = bool(model_predict_from_logits(out) != pred)
        recs.append(rec)

    summary = {}
    for tag in ["top1", "best_clean"]:
        deltas = np.array([r[f"delta_{tag}"] for r in recs
                           if r[f"delta_{tag}"] is not None], dtype=np.float64)
        flips = np.array([r[f"flip_{tag}"] for r in recs
                          if r[f"flip_{tag}"] is not None], dtype=np.float64)
        summary[tag] = {
            "median_delta_from_recon": float(np.median(deltas)) if len(deltas) else None,
            "mean_delta_from_recon": float(np.mean(deltas)) if len(deltas) else None,
            "flip_rate": float(flips.mean()) if len(flips) else None,
            "n": int(len(deltas)),
        }
        if summary[tag]["n"]:
            print(f"  [{tag}] n={summary[tag]['n']} median Δlogit(pred) vs recon = "
                  f"{summary[tag]['median_delta_from_recon']:+.3f}  flip={summary[tag]['flip_rate']*100:.1f}%")
        else:
            print(f"  [{tag}] no samples")
    return summary, recs


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    new_sae, orig_sae, pairs, hidden_npy, acts_new_npy, out = resolve_paths(args)
    if not new_sae.exists():
        raise FileNotFoundError(f"New SAE directory does not exist: {new_sae} (check --new-sae)")
    if not orig_sae.exists():
        raise FileNotFoundError(f"Reference SAE directory does not exist: {orig_sae}")

    rows = load_rows(pairs)
    if args.limit:
        rows = rows[: args.limit]
    dirs = [r["Direction"] for r in rows]
    dir_idx = {d: [i for i, x in enumerate(dirs) if x == d] for d in DIRS}
    empty = [d for d in DIRS if len(dir_idx[d]) == 0]
    if empty:
        print(f"[Warning] directions {empty} have no prompts in the current {len(rows)} samples (--limit too small), "
              f"so their diagnosis tables are meaningless; use a larger --limit or run the full set.")

    model, tokenizer, sae_new, sae_orig, u = load_model_and_saes(new_sae, orig_sae)

    # New SAE hidden + features
    X, acts_new = compute_hidden_and_acts(model, tokenizer, rows, sae_new,
                                          hidden_npy, acts_new_npy, args)

    # Reference SAE features (same hidden), self-checked against the legacy cache
    acts_orig = encode_acts(sae_orig, X)
    legacy = ROOT / "sae" / "activations" / \
        f"the_sae_acts_l26{'_holdout' if args.dataset == 'holdout' else ''}.npy"
    if legacy.exists():
        ref = np.load(str(legacy)).astype(np.float32)
        if ref.shape[0] >= len(rows):
            diff = float(np.abs(ref[: len(rows)] - acts_orig).max())
            print(f"[Self-check] recomputed reference SAE features vs existing cache max|diff| = {diff:.3e} (should be ~0)")

    # 1) Reconstruction-quality comparison
    print("\n=== 1) Reconstruction-quality comparison (on the same raw hidden) ===")
    recon_orig = recon_metrics(X, sae_orig, "orig")
    recon_new = recon_metrics(X, sae_new, "new")
    for name, m in [("orig", recon_orig), ("new", recon_new)]:
        print(f"  [{name}] EV={m['explained_variance_mean']:.4f} "
              f"cossim={m['cossim_mean']:.4f} MSE={m['mse']:.4f} L0={m['l0_mean']:.1f}")

    # 2) Direction-feature diagnosis (new SAE) + calibration (reference SAE)
    w_dec_new = load_w_dec(new_sae)
    diag = diagnose_direction(acts_new, w_dec_new, u, dir_idx, args.top_k)

    w_dec_orig = load_w_dec(orig_sae)
    cal = calibration(acts_orig, w_dec_orig, u, dir_idx)

    # 3) Optional ablation
    abl_summary = None
    if args.ablate:
        abl_summary, _ = ablate_new_sae(model, tokenizer, sae_new, rows, diag, args.limit)

    result = {
        "new_sae": str(new_sae),
        "orig_sae": str(orig_sae),
        "dataset": args.dataset,
        "n_prompts": len(rows),
        "reconstruction": {"orig": recon_orig, "new": recon_new},
        "diagnosis": diag,
        "calibration": cal,
        "ablation": abl_summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out}")

    del model, sae_new, sae_orig
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
