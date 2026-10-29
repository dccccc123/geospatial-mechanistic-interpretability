"""
Discover layer-8 SAE city features via spatial autocorrelation.

Encodes the layer-8 residual of each city token with the L8 SAE, computes global
Moran's I (with Benjamini-Hochberg FDR correction) and local Moran's I (LISA) to
identify geographically clustered features, and visualizes the top features on a
map of China.

Requires: esda, libpysal, geopandas.
"""

import gc
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError("geopandas is not installed; install it with: pip install geopandas") from exc

try:
    from esda.moran import Moran, Moran_Local
    from libpysal.weights import W
except ImportError as exc:
    raise ImportError("esda/libpysal is not installed; install them with: pip install esda libpysal") from exc

from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CITY_CSV_PATH = PROJECT_ROOT / "data_process" / "cleaned_data" / "china_cities_cleaned.csv"
SAE_PATH = PROJECT_ROOT / "sae" / "final_sae" / "layer_8"
ACT_CACHE_DIR = PROJECT_ROOT / "sae" / "activations"
RESULT_JSON_PATH = PROJECT_ROOT / "feature_discovery" / "results" / "city_features_L8.json"
FIG_DIR = PROJECT_ROOT / "feature_discovery" / "figures" / "city_features_L8"

ACT_CACHE_PATH = ACT_CACHE_DIR / "city_sae_acts.npy"
COORD_CACHE_PATH = ACT_CACHE_DIR / "city_coords.npy"
NAME_CACHE_PATH = ACT_CACHE_DIR / "city_names.npy"

MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYER = 8
OUTLIER_DIMS = [2276, 233]
PROMPT_TEMPLATES = [
    "{city} is a city in China",
    "{city} is located in China",
]


def _get_target_layer_module(model: HookedTransformer, layer_idx: int):
    # Prefer the explicit path: model.model.model.layers[layer]
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "blocks"):
        return model.blocks[layer_idx]
    raise RuntimeError("Cannot locate the target layer module on the current model object; check the transformer_lens version.")


def ensure_dirs() -> None:
    ACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_city_data() -> pd.DataFrame:
    print("[Step 1] Loading city data (with coordinates and population)...")
    df = pd.read_csv(CITY_CSV_PATH)
    required_cols = ["城市名称", "经度", "纬度", "人口"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"City data is missing columns: {missing_cols}")
    df = df[required_cols].dropna().reset_index(drop=True)
    print(f"[Info] Total cities: {len(df)}")
    return df


def load_qwen_model(model_name: str) -> Tuple[HookedTransformer, AutoTokenizer]:
    print("[Step 2] Loading Qwen3-8B model and tokenizer (bfloat16 + CUDA)...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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
    return model, tokenizer


def load_sae(sae_path: Path) -> SAE:
    print(f"[Step 2] Loading trained SAE: {sae_path}")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*model_from_pretrained_kwargs.*",
    )
    return SAE.load_from_disk(str(sae_path), device="cuda")


def _find_city_last_token_index(
    prompt: str,
    city_name: str,
    tokenizer: AutoTokenizer,
) -> Tuple[torch.Tensor, int, List[str]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    token_list = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    start_char = prompt.find(city_name)
    if start_char == -1:
        raise ValueError(f"City name not found in prompt: {city_name} | prompt={prompt}")
    end_char = start_char + len(city_name)

    city_token_indices: List[int] = []
    for i, (s, e) in enumerate(offsets):
        overlap = (s < end_char) and (e > start_char)
        if overlap:
            city_token_indices.append(i)

    if not city_token_indices:
        city_ids = tokenizer(city_name, add_special_tokens=False)["input_ids"]
        seq_ids = input_ids.tolist()
        n = len(city_ids)
        for i in range(0, len(seq_ids) - n + 1):
            if seq_ids[i : i + n] == city_ids:
                city_token_indices = list(range(i, i + n))
        if not city_token_indices:
            raise ValueError(f"Cannot locate the city token span: {city_name} | prompt={prompt}")

    last_idx = city_token_indices[-1]
    return input_ids.unsqueeze(0), last_idx, token_list


def _sae_encode(sae: SAE, hidden_state: torch.Tensor) -> torch.Tensor:
    if hasattr(sae, "encode"):
        return sae.encode(hidden_state)

    out = sae(hidden_state)
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "feature_acts"):
        return out.feature_acts
    if isinstance(out, tuple) and len(out) > 0:
        if isinstance(out[0], torch.Tensor):
            return out[0]
    raise RuntimeError("Cannot parse feature activations from the SAE forward output")


def compute_or_load_city_sae_activations(
    city_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_exists = ACT_CACHE_PATH.exists() and COORD_CACHE_PATH.exists() and NAME_CACHE_PATH.exists()
    if cache_exists:
        print("[Step 2] Activation cache found; loading directly and skipping model inference...")
        city_sae_acts = np.load(ACT_CACHE_PATH)
        city_coords = np.load(COORD_CACHE_PATH)
        city_names = np.load(NAME_CACHE_PATH, allow_pickle=True)
        print(f"[Info] Loaded cached activation matrix: {city_sae_acts.shape}")
        return city_sae_acts, city_coords, city_names

    model, tokenizer = load_qwen_model(MODEL_NAME)
    sae = load_sae(SAE_PATH)

    city_names = city_df["城市名称"].astype(str).to_numpy()
    city_coords = city_df[["经度", "纬度"]].to_numpy(dtype=np.float64)

    city_sae_acts: List[np.ndarray] = []
    debug_prompt_count = 0

    print("[Step 2] Extracting L8 city-token activations and encoding as SAE features (mean over two templates)...")
    for city in tqdm(city_names, desc="City inference", unit="city"):
        prompt_acts: List[np.ndarray] = []

        for template in PROMPT_TEMPLATES:
            prompt = template.format(city=city)
            input_ids, city_last_idx, token_list = _find_city_last_token_index(prompt, city, tokenizer)

            if debug_prompt_count < 3:
                print(f"[Debug] Prompt: {prompt}")
                print(f"[Debug] Token list: {token_list}")
                print(f"[Debug] City last-token index: {city_last_idx}")
                debug_prompt_count += 1

            captured: Dict[str, torch.Tensor] = {}

            def layer_hook(_module, _input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden.clone()
                    hidden[:, :, OUTLIER_DIMS] = 0.0
                    captured["hidden"] = hidden[:, city_last_idx, :].detach()
                    return (hidden, *output[1:])

                hidden = output.clone()
                hidden[:, :, OUTLIER_DIMS] = 0.0
                captured["hidden"] = hidden[:, city_last_idx, :].detach()
                return hidden

            target_layer_module = _get_target_layer_module(model, TARGET_LAYER)
            hook_handle = target_layer_module.register_forward_hook(layer_hook)
            input_ids = input_ids.to("cuda")

            with torch.inference_mode():
                try:
                    _ = model(input_ids, return_type=None)
                except TypeError:
                    _ = model(input_ids)

            hook_handle.remove()

            if "hidden" not in captured:
                raise RuntimeError(f"Layer {TARGET_LAYER} output not captured, city={city}, prompt={prompt}")

            hidden_vec = captured["hidden"].to(device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                feat_acts = _sae_encode(sae, hidden_vec)

            prompt_acts.append(feat_acts.squeeze(0).detach().float().cpu().numpy())

        city_mean_act = np.mean(np.stack(prompt_acts, axis=0), axis=0)
        city_sae_acts.append(city_mean_act)

    city_sae_acts_arr = np.stack(city_sae_acts, axis=0).astype(np.float32)

    np.save(ACT_CACHE_PATH, city_sae_acts_arr)
    np.save(COORD_CACHE_PATH, city_coords)
    np.save(NAME_CACHE_PATH, city_names)
    print(f"[Info] Saved cache: {ACT_CACHE_PATH}")
    print(f"[Info] Saved cache: {COORD_CACHE_PATH}")
    print(f"[Info] Saved cache: {NAME_CACHE_PATH}")

    del model
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    print("[Info] Activation extraction done; model memory released.")

    return city_sae_acts_arr, city_coords, city_names


def filter_nontrivial_features(city_sae_acts: np.ndarray) -> np.ndarray:
    print("[Step 3] Filtering nontrivial features (dropping features with >95% zero city activation)...")
    zero_ratio = (city_sae_acts == 0).mean(axis=0)
    keep_mask = zero_ratio <= 0.95
    keep_ids = np.where(keep_mask)[0]
    print(f"[Info] Nontrivial features: {len(keep_ids)} / {city_sae_acts.shape[1]}")
    return keep_ids


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns adjusted p-values (monotonized to [0, 1])."""
    p = np.asarray(p_values, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)  # non-finite -> not significant
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = sorted_p * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = adjusted
    return out


def build_spatial_weights(city_coords: np.ndarray, k: int = 8):
    print("[Step 4.1] Building the KNN spatial-weights matrix (haversine + row normalization)...")
    # city_coords: [lon, lat] -> radians for haversine.
    lon_rad = np.radians(city_coords[:, 0]).astype(np.float64)
    lat_rad = np.radians(city_coords[:, 1]).astype(np.float64)

    # Build the pairwise haversine distance matrix directly to avoid libpysal-version issues with KNN.from_array.
    dlon = lon_rad[:, None] - lon_rad[None, :]
    dlat = lat_rad[:, None] - lat_rad[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

    n = dist.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < number of cities; got k={k}, n={n}")

    np.fill_diagonal(dist, np.inf)

    neighbors: Dict[int, List[int]] = {}
    weights: Dict[int, List[float]] = {}
    for i in range(n):
        nn_idx = np.argpartition(dist[i], k)[:k]
        nn_sorted = nn_idx[np.argsort(dist[i, nn_idx])]
        neighbors[i] = nn_sorted.tolist()
        weights[i] = [1.0] * len(nn_sorted)

    w = W(neighbors=neighbors, weights=weights)

    w.transform = "R"
    print("[Info] Spatial-weights matrix built.")
    return w


def compute_global_morans_i(
    city_sae_acts: np.ndarray,
    nontrivial_ids: np.ndarray,
    w,
) -> List[Dict[str, float]]:
    print("[Step 4.2] Computing global Moran's I (999 permutations)...")
    results: List[Dict[str, float]] = []

    for feat_id in tqdm(nontrivial_ids, desc="Moran's I", unit="feat"):
        x = city_sae_acts[:, feat_id].astype(np.float64)
        moran = Moran(x, w, permutations=999)
        results.append(
            {
                "feature_id": int(feat_id),
                "morans_i": float(moran.I),
                "p_value": float(moran.p_sim),
            }
        )

    return results


def _describe_hotspot_geo(city_coords: np.ndarray, hotspot_indices: np.ndarray) -> str:
    if len(hotspot_indices) == 0:
        return "No significant HH hot-spot cities"

    lon = city_coords[hotspot_indices, 0]
    lat = city_coords[hotspot_indices, 1]
    return (
        f"Longitude range [{lon.min():.2f}, {lon.max():.2f}], "
        f"Latitude range [{lat.min():.2f}, {lat.max():.2f}], "
        f"center approx (lon {lon.mean():.2f}, lat {lat.mean():.2f})"
    )


def compute_local_morans_i_top_features(
    city_sae_acts: np.ndarray,
    city_names: np.ndarray,
    city_coords: np.ndarray,
    top_features: List[Dict[str, float]],
    w,
) -> Dict[int, Dict[str, object]]:
    print("[Step 4.4] Computing local Moran's I (LISA) for the top-20 significant features...")
    local_results: Dict[int, Dict[str, object]] = {}

    for item in top_features:
        feat_id = int(item["feature_id"])
        x = city_sae_acts[:, feat_id].astype(np.float64)
        local = Moran_Local(x, w, permutations=999)

        local_p_adj = benjamini_hochberg(np.asarray(local.p_sim, dtype=np.float64))
        hotspot_mask = (local.q == 1) & (local_p_adj < 0.05)
        hotspot_indices = np.where(hotspot_mask)[0]
        hotspot_cities = city_names[hotspot_indices].tolist()
        geo_desc = _describe_hotspot_geo(city_coords, hotspot_indices)

        print(f"[LISA] Feature {feat_id} HH hot-spot cities: {hotspot_cities}")
        print(f"[LISA] Feature {feat_id} geographic distribution: {geo_desc}")

        local_results[feat_id] = {
            "hotspot_indices": hotspot_indices,
            "hotspot_cities": hotspot_cities,
            "geo_desc": geo_desc,
        }

    return local_results


def _load_china_basemap():
    """Load a China province-level basemap (incl. Taiwan/HK/Macau) from Aliyun DataV.GeoAtlas."""
    url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
    try:
        china = gpd.read_file(url)
        if china.empty:
            print("[Warning] DataV basemap is empty; the map will show only scatter points.")
            return None
        return china
    except Exception as exc:
        print(f"[Warning] Cannot load the DataV basemap ({exc}); the map will show only scatter points.")
        return None


def visualize_top_features(
    city_sae_acts: np.ndarray,
    city_names: np.ndarray,
    city_coords: np.ndarray,
    top10_features: List[Dict[str, float]],
    local_results: Dict[int, Dict[str, object]],
) -> None:
    print("[Step 5] Generating geographic visualizations for the top-10 significant features...")
    china_gdf = _load_china_basemap()

    lon = city_coords[:, 0]
    lat = city_coords[:, 1]

    for item in top10_features:
        feat_id = int(item["feature_id"])
        morans_i = float(item["morans_i"])
        p_value = float(item["p_value"])
        acts = city_sae_acts[:, feat_id]

        hotspot_indices = local_results.get(feat_id, {}).get(
            "hotspot_indices", np.array([], dtype=int)
        )
        hotspot_set = set(hotspot_indices.tolist())

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor("#F8F8F8")
        ax.set_facecolor("#EEF2F7")

        # ── basemap ──
        if china_gdf is not None:
            china_gdf.boundary.plot(ax=ax, color="#999999", linewidth=0.8)

        # ── shared colormap range ──
        vmin = acts.min()
        vmax = acts.max()

        # ── non-hotspot cities: small, low-alpha dots ──
        non_hotspot_mask = np.array(
            [i not in hotspot_set for i in range(len(city_names))]
        )
        ax.scatter(
            lon[non_hotspot_mask],
            lat[non_hotspot_mask],
            c=acts[non_hotspot_mask],
            cmap="YlOrRd",
            s=18,
            alpha=0.45,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )

        # ── hotspot cities: large dots + black edge ──
        if len(hotspot_indices) > 0:
            sc = ax.scatter(
                lon[hotspot_indices],
                lat[hotspot_indices],
                c=acts[hotspot_indices],
                cmap="YlOrRd",
                s=110,
                alpha=1.0,
                edgecolors="black",
                linewidths=0.9,
                vmin=vmin,
                vmax=vmax,
                zorder=4,
            )
        else:
            sc = ax.scatter(
                lon, lat,
                c=acts,
                cmap="YlOrRd",
                s=18,
                alpha=0.45,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
                zorder=2,
            )

        # ── hotspot labels: top-8 by activation, adjustText avoids overlap ──
        if len(hotspot_indices) > 0:
            top_n = min(8, len(hotspot_indices))
            sorted_idx = hotspot_indices[
                np.argsort(acts[hotspot_indices])[::-1][:top_n]
            ]

            texts = []
            for idx in sorted_idx:
                t = ax.text(
                    lon[idx],
                    lat[idx],
                    str(city_names[idx]),
                    fontsize=7.5,
                    fontweight="bold",
                    color="#1a1a1a",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="#aaaaaa",
                        alpha=0.85,
                        linewidth=0.5,
                    ),
                    zorder=5,
                )
                texts.append(t)

            adjust_text(
                texts,
                x=lon[sorted_idx],
                y=lat[sorted_idx],
                ax=ax,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    lw=0.6,
                ),
                expand=(1.8, 1.8),
                force_text=(0.5, 0.5),
                force_points=(0.3, 0.3),
                lim=300,
            )

        # ── title ──
        ax.set_title(
            f"Feature {feat_id}  |  Moran's I = {morans_i:.4f}  |  p = {p_value:.4f}"
            f"\nHH Hotspots: {len(hotspot_indices)} cities",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_xlim(72, 136)
        ax.set_ylim(17, 54)

        # ── Colorbar ──────────────────────────────────────────
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("SAE Feature Activation", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # ── legend ──
        legend_elements = [
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor="#dddddd",
                markersize=6,
                label="Other cities",
            ),
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor="#d73027",
                markeredgecolor="black",
                markersize=10,
                label="HH Hotspot (Top-8 labeled)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=8,
            framealpha=0.85,
        )

        save_path = FIG_DIR / f"feature_{feat_id}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved: {save_path}")


def save_results_json(
    significant_features: List[Dict[str, float]],
    local_results: Dict[int, Dict[str, object]],
    total_nontrivial: int,
) -> None:
    print("[Step 6] Saving result JSON...")
    out_features = []
    for item in significant_features:
        feat_id = int(item["feature_id"])
        out_features.append(
            {
                "feature_id": feat_id,
                "morans_i": float(item["morans_i"]),
                "p_value": float(item["p_value"]),
                "p_adj": float(item["p_adj"]),
                "hotspot_cities": local_results.get(feat_id, {}).get("hotspot_cities", []),
            }
        )

    payload = {
        "significant_features": out_features,
        "total_significant": len(significant_features),
        "total_nontrivial": int(total_nontrivial),
    }

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Info] Result saved to: {RESULT_JSON_PATH}")


def main() -> None:
    ensure_dirs()
    city_df = load_city_data()

    city_sae_acts, city_coords, city_names = compute_or_load_city_sae_activations(city_df)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nontrivial_ids = filter_nontrivial_features(city_sae_acts)

    w = build_spatial_weights(city_coords, k=8)
    global_results = compute_global_morans_i(city_sae_acts, nontrivial_ids, w)

    # Multiple-testing correction: Benjamini-Hochberg FDR over the tested nontrivial features
    p_adj = benjamini_hochberg(np.array([x["p_value"] for x in global_results]))
    for item, padj in zip(global_results, p_adj):
        item["p_adj"] = float(padj)

    significant = [x for x in global_results if x["p_adj"] < 0.05 and x["morans_i"] > 0]
    significant.sort(key=lambda x: x["morans_i"], reverse=True)

    n_uncorrected = sum(1 for x in global_results if x["p_value"] < 0.05 and x["morans_i"] > 0)
    print(f"[Step 4.3] Uncorrected (p<0.05 and I>0) features: {n_uncorrected}")
    print(f"[Step 4.3] FDR(BH)-significant features: {len(significant)}")
    top20 = significant[:20]
    print("[Step 4.3] Top 20 significant features (by descending Moran's I):")
    for rank, item in enumerate(top20, start=1):
        print(
            f"  {rank:02d}. feature={item['feature_id']}, "
            f"I={item['morans_i']:.6f}, p={item['p_value']:.6f}"
        )

    local_results = compute_local_morans_i_top_features(
        city_sae_acts=city_sae_acts,
        city_names=city_names,
        city_coords=city_coords,
        top_features=top20,
        w=w,
    )

    top10 = significant[:10]
    visualize_top_features(
        city_sae_acts=city_sae_acts,
        city_names=city_names,
        city_coords=city_coords,
        top10_features=top10,
        local_results=local_results,
    )

    save_results_json(
        significant_features=significant,
        local_results=local_results,
        total_nontrivial=len(nontrivial_ids),
    )

    print("[Done] L8 SAE city-feature spatial-autocorrelation analysis complete.")


if __name__ == "__main__":
    main()
