# User Guide

This guide summarizes the inputs, outputs, and command-line options of the main scripts. Every script also prints a usage summary with `--help`.

## Common options

- `--dataset train|holdout` — select the discovery set or the city-disjoint held-out set (available on the ablation scripts).
- `--limit N` — run only the first `N` prompts (useful for smoke tests).
- `--seed N` / `--output_dir PATH` — training seed and output directory (in `sae/train.py`).

## data_process

| Script | Input | Output | Purpose |
|---|---|---|---|
| `main.py` | GeoNames city records | `data_process/dataset/balanced_city_pairs.jsonl` | Two-subword city filtering and stratified pair sampling |
| `build_holdout.py` | GeoNames city records | `data_process/dataset/holdout_city_pairs.jsonl` | City-disjoint held-out set |

## sae

| Script | Input | Output | Purpose |
|---|---|---|---|
| `data_builder.py` | geography-dense text + WikiText-103 | packed tokenized dataset | Build the 1:4 SAE training corpus |
| `train.py` | tokenized dataset | `sae/final_sae/layer_8/`, `layer_26/` | Train the two BatchTopK SAEs (k=32, d_sae=65536) |

## component_localization

| Script | Input | Output | Purpose |
|---|---|---|---|
| `patching.py` | the 180 prompts | `patching_results.npz` | Activation patching (metric `m`) |
| `patching_city_a.py` | the 180 prompts | `patching_results_cityA.npz` | c_A-position patching |
| `path_patching.py` | the 180 prompts | `path_patching_results.json` | Edge-level path patching |
| `plot_patching_results.py` | `patching_results*.npz` | `heatmap_mlp.png`, `heatmap_attn.png` | Exploratory patching heatmaps |

## feature_discovery

| Script | Input | Output | Purpose |
|---|---|---|---|
| `city_feature_L8.py` | L8 SAE + city coordinates | `results/city_features_L8.json` | Moran's I + BH-FDR geographic features |
| `cdla_rank_L26.py` | L26 SAE + the 180 prompts | `results/` cDLA ranking | cDLA direction features + bootstrap stability |
| `inspect_direction_features.py` | L26 SAE | printed proj/act table | Diagnose direction-feature selectivity |
| `diagnose_seed_sae.py` | a new SAE | `results/diagnose_seed_sae_*.json` | Seed-robustness diagnostic (reconstruction + cDLA) |

## causal_verification

| Script | Input | Output | Purpose |
|---|---|---|---|
| `l8_geographic_subspace_ablation_ca_joint.py` | L8 SAE + prompts | `results/l8_..._ca_joint.json` | Geographic ablation at c_A / c_B / joint (with matched controls) |
| `l26_direction_feature_ablation.py` | L26 SAE + prompts | `results/l26_direction_feature_ablation.json` | Direction-feature ablation |
| `head_ablation_delta_logit.py` | model + prompts | `results/head_ablation_delta_logit.json` | Routing-head ablation |
| `l8_geographic_to_l26_direction_mediation.py` | L8+L26 SAE + prompts | `results/l8_..._mediation.json` | Upstream mediation (freeze heads) |
| `l8_geographic_bypass_layerscan.py` | L8+L26 SAE + prompts | `results/l8_..._bypass_layerscan.json` | Bypass layer scan |
| `cluster_bootstrap.py` | ablation JSONs | bootstrap p-values | Pair-level cluster bootstrap |
| `mediation_cluster_bootstrap.py` | mediation JSONs | bootstrap p-values | Cluster bootstrap for the mediation effects |
| `correct_incorrect_split.py` | ablation JSONs | correct/incorrect effects | Correct/incorrect prediction split |
| `ablation_margin_and_dla.py` | ablation JSONs | flip rate / margin / cDLA | Flip rate, winning margin, cDLA vs ablation |

## circuit_tracing

| Script | Input | Output | Purpose |
|---|---|---|---|
| `head_to_direction_feature_activation.py` | L26 SAE + prompts | `results/head_to_direction_feature_activation.json` | Head -> direction-feature activation change |
| `circuit.py` | SAE activations + attention | circuit edge weights | Circuit assembly |
| `circuit_grid.py` | circuit data | `results/circuit_perprompt.json` | Per-prompt circuit |

## Expected outputs

- **Table V** ablation effects: geographic (joint) −1.25, direction −0.50, Head 24 −0.25, Head 25 −0.31 (median ∆logit, control 0.00).
- **Mediation**: 60% of the geographic→direction effect is mediated by the routing heads.
- **Accuracy**: 56.7% on the discovery set (axis-based rule).
