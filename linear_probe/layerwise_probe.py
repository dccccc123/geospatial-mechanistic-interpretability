"""
Layerwise linear probes for direction classification, with two label sets.

Trains a 5-fold logistic-regression probe on the residual at the "the" position
of every layer, using either the ground-truth direction or the model's own
prediction as labels. Caches per-layer activations and model predictions, and
reports per-layer accuracy (also produces baseline_preds.npy, used by other
analyses).

Usage: python linear_probe/layerwise_probe.py
"""

import os, json, warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────
# paths
# ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data_process/dataset/balanced_city_pairs.jsonl"
CACHE_DIR  = Path(__file__).resolve().parent / "activations"
RESULT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR    = Path(__file__).resolve().parent / "figures"

for d in [CACHE_DIR, RESULT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen3-8B"
N_LAYERS      = 36
OUTLIER_DIMS  = [2276, 233]
PROMPT_TPL    = "Relative to {city_a}, {city_b} is to the"
THE_TOKEN_ID  = 279        # " the"
DIR_LABELS    = ["East", "West", "South", "North"]
LABEL2IDX     = {d: i for i, d in enumerate(DIR_LABELS)}
IDX2LABEL     = {i: d for i, d in enumerate(DIR_LABELS)}

# token ids for axis-based logit-difference prediction
DIR_TOKEN_IDS = {"East": 10984, "West": 9710, "South": 9806, "North": 10200}


# ─────────────────────────────────────────────────────────────
# Step 1: load data
# ─────────────────────────────────────────────────────────────
def load_data():
    data    = [json.loads(l) for l in DATA_PATH.open()]
    prompts = [PROMPT_TPL.format(city_a=d["City_A"], city_b=d["City_B"]) for d in data]
    labels_gt = [LABEL2IDX[d["Direction"]] for d in data]

    dist = defaultdict(int)
    for d in data:
        dist[d["Direction"]] += 1
    print(f"[Data] {len(data)} prompts, direction distribution: {dict(dist)}")
    return prompts, labels_gt, data


# ─────────────────────────────────────────────────────────────
# Step 2: extract per-layer activations + model predictions (cached)
# ─────────────────────────────────────────────────────────────
def _get_the_pos(input_ids):
    positions = (input_ids[0] == THE_TOKEN_ID).nonzero(as_tuple=True)[0]
    return positions[-1].item()


def _model_predict(logits):
    """Axis-based prediction from the four direction logits."""
    score_ew = logits[DIR_TOKEN_IDS["East"]].item() - logits[DIR_TOKEN_IDS["West"]].item()
    score_sn = logits[DIR_TOKEN_IDS["South"]].item() - logits[DIR_TOKEN_IDS["North"]].item()
    if abs(score_ew) >= abs(score_sn):
        return LABEL2IDX["East"] if score_ew > 0 else LABEL2IDX["West"]
    else:
        return LABEL2IDX["South"] if score_sn > 0 else LABEL2IDX["North"]


def extract_all(prompts):
    """
    Extract L0..L35 activations and the final model prediction, with caching.

    Returns:
      layer_acts   : {layer_idx: np.ndarray [N, 4096]}
      labels_model : np.ndarray [N]
    """
    missing_layers  = [
        i for i in range(N_LAYERS)
        if not (CACHE_DIR / f"layer_{i:02d}_the_acts.npy").exists()
    ]
    pred_cache_path = CACHE_DIR / "baseline_preds.npy"
    need_preds      = not pred_cache_path.exists()

    if not missing_layers and not need_preds:
        print("[Cache] All activations and predictions are cached; loading directly.")
    else:
        print(f"[Model] {len(missing_layers)} layer activations missing"
              f"{'+ model predictions' if need_preds else ''}; loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        print(f"[Model] Loaded, device: {next(model.parameters()).device}")

        _run_extraction(model, tokenizer, prompts, missing_layers, need_preds, pred_cache_path)

        del model
        torch.cuda.empty_cache()
        print("[Model] GPU memory released.")

    # load cache
    print("[Load] Loading activations and predictions from cache...")
    layer_acts = {
        i: np.load(str(CACHE_DIR / f"layer_{i:02d}_the_acts.npy"))
        for i in range(N_LAYERS)
    }
    labels_model = np.load(str(pred_cache_path))
    print(f"[Load] per-layer shape={layer_acts[0].shape}, predictions shape={labels_model.shape}")
    return layer_acts, labels_model


def _run_extraction(model, tokenizer, prompts, missing_layers, need_preds, pred_cache_path):
    missing_set  = set(missing_layers)
    acts_buffer  = {i: [] for i in missing_set}
    preds_buffer = [] if need_preds else None

    for prompt in tqdm(prompts, desc="[extract activations + predictions]"):
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        the_pos = _get_the_pos(inputs.input_ids)

        handles     = []
        layer_cache = {}

        for layer_idx in missing_set:
            def make_hook(lidx):
                def hook_fn(module, input, output):
                    h = output.clone().float()
                    h[:, :, OUTLIER_DIMS] = 0.0
                    layer_cache[lidx] = h[0, the_pos, :].cpu().numpy()
                return hook_fn
            handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            handles.append(handle)

        with torch.no_grad():
            out = model(**inputs)

        for handle in handles:
            handle.remove()

        for layer_idx in missing_set:
            acts_buffer[layer_idx].append(layer_cache[layer_idx])

        if need_preds:
            preds_buffer.append(_model_predict(out.logits[0, -1]))

    # save activation caches
    for layer_idx in missing_set:
        arr  = np.stack(acts_buffer[layer_idx], axis=0)
        path = CACHE_DIR / f"layer_{layer_idx:02d}_the_acts.npy"
        np.save(str(path), arr)
        print(f"  [Cache] Layer {layer_idx:2d} -> {path.name}")

    # save prediction cache
    if need_preds:
        np.save(str(pred_cache_path), np.array(preds_buffer))
        print(f"  [Cache] model predictions -> {pred_cache_path.name}")


# ─────────────────────────────────────────────────────────────
# Step 3: train per-layer linear probes (5-fold CV)
# ─────────────────────────────────────────────────────────────
def train_probes(layer_acts, labels, desc="[layerwise probe]"):
    labels_arr = np.array(labels)
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results    = {}

    for layer_idx in tqdm(range(N_LAYERS), desc=f"  {desc}"):
        X         = layer_acts[layer_idx]
        fold_accs = []

        for train_idx, val_idx in cv.split(X, labels_arr):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = labels_arr[train_idx], labels_arr[val_idx]

            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_val  = scaler.transform(X_val)

            clf = LogisticRegression(
                max_iter=1000, C=1.0,
                multi_class="multinomial", solver="lbfgs"
            )
            clf.fit(X_tr, y_tr)
            fold_accs.append(accuracy_score(y_val, clf.predict(X_val)))

        results[layer_idx] = {
            "mean":      round(float(np.mean(fold_accs)), 4),
            "std":       round(float(np.std(fold_accs)),  4),
            "fold_accs": [round(a, 4) for a in fold_accs],
        }

    return results


# ─────────────────────────────────────────────────────────────
# Step 4: print comparison
# ─────────────────────────────────────────────────────────────
def print_results(res_gt, res_model, labels_gt, labels_model):
    model_acc = float(np.mean(np.array(labels_gt) == np.array(labels_model)))
    print(f"\n  Final model prediction accuracy (reference baseline): {model_acc*100:.1f}%")
    print(f"  Four-class random baseline: 25.0%\n")

    best_gt    = max(res_gt,    key=lambda i: res_gt[i]["mean"])
    best_model = max(res_model, key=lambda i: res_model[i]["mean"])

    print(f"  {'Layer':>7}  {'GT Acc':>14}  {'Model Acc':>14}")
    print(f"  {'─'*46}")
    for i in range(N_LAYERS):
        note = ""
        if i == 8:          note += " ← L8"
        if i == 26:         note += " ← L26"
        if i == best_gt:    note += " [GT max]"
        if i == best_model: note += " [Model max]"
        print(f"  Layer {i:2d}:  "
              f"{res_gt[i]['mean']*100:5.1f}%±{res_gt[i]['std']*100:.1f}%   "
              f"{res_model[i]['mean']*100:5.1f}%±{res_model[i]['std']*100:.1f}%"
              f"{note}")

    print(f"\n  GT labels    max: Layer {best_gt}  {res_gt[best_gt]['mean']*100:.1f}%")
    print(f"  Model labels max: Layer {best_model}  {res_model[best_model]['mean']*100:.1f}%")
    print(f"\n  L8   GT={res_gt[8]['mean']*100:.1f}%   Model={res_model[8]['mean']*100:.1f}%")
    print(f"  L26  GT={res_gt[26]['mean']*100:.1f}%   Model={res_model[26]['mean']*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# Step 5: visualize (dual-label curves)
# ─────────────────────────────────────────────────────────────
def plot_results(res_gt, res_model, model_acc):
    layers    = list(range(N_LAYERS))
    means_gt  = [res_gt[i]["mean"]    * 100 for i in layers]
    stds_gt   = [res_gt[i]["std"]     * 100 for i in layers]
    means_mod = [res_model[i]["mean"] * 100 for i in layers]
    stds_mod  = [res_model[i]["std"]  * 100 for i in layers]

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("#F8F8F8")
    ax.set_facecolor("#F0F0F0")

    # ground-truth label curve
    ax.plot(layers, means_gt, color="#2563EB", linewidth=2,
            zorder=3, label="Ground Truth Labels")
    ax.fill_between(layers,
                    [m - s for m, s in zip(means_gt, stds_gt)],
                    [m + s for m, s in zip(means_gt, stds_gt)],
                    alpha=0.15, color="#2563EB")

    # model-prediction label curve
    ax.plot(layers, means_mod, color="#D97706", linewidth=2,
            linestyle="--", zorder=3, label="Model Prediction Labels")
    ax.fill_between(layers,
                    [m - s for m, s in zip(means_mod, stds_mod)],
                    [m + s for m, s in zip(means_mod, stds_mod)],
                    alpha=0.15, color="#D97706")

    # baseline
    ax.axhline(25.0, color="gray", linestyle=":", linewidth=1,
               label="Random Baseline (25%)")
    ax.axhline(model_acc * 100, color="#6B7280", linestyle="-.",
               linewidth=1, label=f"Model Output Acc ({model_acc*100:.1f}%)")

    # mark L8 and L26
    for layer, color, label in [(8, "#DC2626", "L8"), (26, "#16A34A", "L26")]:
        ax.axvline(layer, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
        y_pos = max(means_gt[layer], means_mod[layer]) + 4
        ax.text(layer, y_pos, label, ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        "Layerwise Linear Probe Accuracy — Dual Label Strategy\n"
        "(Direction Classification, 5-fold CV)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_ylim(0, 110)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.4)

    save_path = FIG_DIR / "layerwise_probe_accuracy.png"
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Step 6: save results
# ─────────────────────────────────────────────────────────────
def save_results(res_gt, res_model, labels_gt, labels_model):
    model_acc  = float(np.mean(np.array(labels_gt) == np.array(labels_model)))
    best_gt    = max(res_gt,    key=lambda i: res_gt[i]["mean"])
    best_model = max(res_model, key=lambda i: res_model[i]["mean"])

    output = {
        "model":            MODEL_NAME,
        "n_layers":         N_LAYERS,
        "n_samples":        len(labels_gt),
        "cv_folds":         5,
        "random_baseline":  0.25,
        "model_output_acc": round(model_acc, 4),
        "ground_truth_labels": {
            "best_layer":    best_gt,
            "best_accuracy": res_gt[best_gt]["mean"],
            "l8_accuracy":   res_gt[8]["mean"],
            "l26_accuracy":  res_gt[26]["mean"],
            "layerwise":     {str(i): res_gt[i] for i in range(N_LAYERS)},
        },
        "model_pred_labels": {
            "best_layer":    best_model,
            "best_accuracy": res_model[best_model]["mean"],
            "l8_accuracy":   res_model[8]["mean"],
            "l26_accuracy":  res_model[26]["mean"],
            "layerwise":     {str(i): res_model[i] for i in range(N_LAYERS)},
        },
    }

    path = RESULT_DIR / "layerwise_probe.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[Saved] Result written to: {path}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Layerwise linear probe (dual-label strategy)")
    print("=" * 55)

    print(f"\n=== Step 1: load data ===")
    prompts, labels_gt, _ = load_data()

    print(f"\n=== Step 2: extract per-layer activations + model predictions ===")
    layer_acts, labels_model = extract_all(prompts)

    model_acc = float(np.mean(np.array(labels_gt) == labels_model))
    print(f"[Prediction] model output accuracy: {model_acc*100:.1f}%")

    print(f"\n=== Step 3a: train ground-truth-label probes ===")
    res_gt = train_probes(layer_acts, labels_gt, desc="GT-label probe")

    print(f"\n=== Step 3b: train model-prediction-label probes ===")
    res_model = train_probes(layer_acts, labels_model.tolist(), desc="Model-label probe")

    print(f"\n=== Step 4: summarize results ===")
    print_results(res_gt, res_model, labels_gt, labels_model.tolist())

    print(f"\n=== Step 5: plot ===")
    plot_results(res_gt, res_model, model_acc)

    print(f"\n=== Step 6: save results ===")
    save_results(res_gt, res_model, labels_gt, labels_model.tolist())

    print(f"\n{'='*55}")
    print(f"  Experiment complete")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()