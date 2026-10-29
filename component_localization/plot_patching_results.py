import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


STAGES = ["city_a", "city_b", "final"]
STAGE_LABELS = {
    "city_a": "City A",
    "city_b": "City B",
    "final": "Final Token",
}


def load_results(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Result file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    metric_mlp = {}
    metric_attn = {}
    for stage in STAGES:
        metric_mlp[stage] = data[f"metric_mlp_{stage}"]
        metric_attn[stage] = data[f"metric_attn_{stage}"]

    meta = {
        "clean_diff": float(data["clean_diff"][0]),
        "corrupt_diff": float(data["corrupt_diff"][0]),
        "denominator": float(data["denominator"][0]),
        "n_layers": int(data["n_layers"][0]),
        "n_heads": int(data["n_heads"][0]),
        "num_pairs": int(data["num_pairs"][0]),
    }
    return metric_mlp, metric_attn, meta


def draw_mlp_heatmap(metric_mlp: dict[str, np.ndarray], save_path: Path, title_suffix: str = "") -> None:
    n_layers = metric_mlp["city_a"].shape[0]
    mlp_stages = ["city_b", "final"]
    mlp_data = np.stack([metric_mlp[s] for s in mlp_stages], axis=1)

    plt.figure(figsize=(7, 10))
    ax = sns.heatmap(
        mlp_data,
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Recovery Metric (Lower is more causal)"},
    )
    ax.invert_yaxis()
    ax.set_title(f"MLP Causal Heatmap{title_suffix}", fontsize=14, pad=12)
    ax.set_xlabel("Intervention Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xticks(np.arange(len(mlp_stages)) + 0.5, [STAGE_LABELS[s] for s in mlp_stages], rotation=0)
    ax.set_yticks(np.arange(0, n_layers, 2) + 0.5, np.arange(0, n_layers, 2))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def draw_attn_final_heatmap(metric_attn: dict[str, np.ndarray], save_path: Path, title_suffix: str = "") -> None:
    attn_city_b = metric_attn["city_b"]
    attn_final = metric_attn["final"]
    n_layers, n_heads = attn_final.shape

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    stages = [("city_b", attn_city_b), ("final", attn_final)]

    for idx, (stage_name, stage_metric) in enumerate(stages):
        ax = sns.heatmap(
            stage_metric,
            cmap="vlag",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Recovery Metric (Lower is more causal)"} if idx == 1 else None,
            ax=axes[idx],
        )
        ax.invert_yaxis()
        ax.set_title(f"{STAGE_LABELS[stage_name]} Position", fontsize=13, pad=10)
        ax.set_xlabel("Head", fontsize=11)
        ax.set_xticks(np.arange(0, n_heads, 2) + 0.5, np.arange(0, n_heads, 2))
        if idx == 0:
            ax.set_ylabel("Layer", fontsize=11)
            ax.set_yticks(np.arange(0, n_layers, 2) + 0.5, np.arange(0, n_layers, 2))
        else:
            ax.set_ylabel("")

    fig.suptitle(f"Attention Head Causal Heatmap{title_suffix}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def print_npz_contents(npz_path: Path) -> None:
	if not npz_path.exists():
		raise FileNotFoundError(f"File not found: {npz_path}")

	# lift the print threshold so large arrays print fully.
	np.set_printoptions(threshold=np.inf, linewidth=200)

	with np.load(npz_path, allow_pickle=True) as data:
		keys = list(data.keys())
		print(f"File path: {npz_path}")
		print(f"Contains {len(keys)} keys: {keys}\n")

		for key in keys:
			value = data[key]
			print(f"=== {key} ===")
			print(f"dtype={value.dtype}, shape={value.shape}")
			print("Full contents:")
			print(value)
			print("\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heatmaps from saved activation-patching results")
    parser.add_argument(
        "--results-npz",
        type=str,
        default="spatial_computation_neurons/patching_results.npz",
        help="Path to saved patching npz file",
    )
    parser.add_argument(
        "--meta-json",
        type=str,
        default="spatial_computation_neurons/patching_results_meta.json",
        help="Optional metadata json path",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="spatial_computation_neurons",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    results_path = Path(args.results_npz)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print_npz_contents(results_path)

    metric_mlp, metric_attn, meta = load_results(results_path)

    title_suffix = f"\nN={meta['num_pairs']}, Clean={meta['clean_diff']:.4f}, Corrupt={meta['corrupt_diff']:.4f}"

    mlp_fig = save_dir / "heatmap_mlp.png"
    attn_fig = save_dir / "heatmap_attn.png"
    draw_mlp_heatmap(metric_mlp, mlp_fig, title_suffix=title_suffix)
    draw_attn_final_heatmap(metric_attn, attn_fig, title_suffix=title_suffix)

    meta_path = Path(args.meta_json)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            user_meta = json.load(f)
        print(f"[Info] Loaded metadata: model={user_meta.get('model_name')}, dataset={user_meta.get('dataset_path')}")

    print(f"[Info] Saved figure: {mlp_fig}")
    print(f"[Info] Saved figure: {attn_fig}")


if __name__ == "__main__":
    main()
