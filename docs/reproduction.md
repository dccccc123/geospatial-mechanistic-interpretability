# Reproduction Guide

This guide explains how to reproduce the main results of the paper, following the four-stage analysis framework.

## Prerequisites

1. Python 3.12 and the dependencies listed in `requirements.txt`.
2. A single NVIDIA GPU with at least 48 GB of VRAM (required to train the SAEs; the analysis and ablation scripts run on the GPU as well).
3. The Qwen3-8B model, which is downloaded automatically from Hugging Face (`Qwen/Qwen3-8B`).
4. The trained SAE weights, downloaded from Hugging Face
   ([dc-321/qwen3-geospatial-sae-weights](https://huggingface.co/dc-321/qwen3-geospatial-sae-weights)) and placed at:
   - `sae/final_sae/layer_8/sae_weights.safetensors` and `cfg.json`
   - `sae/final_sae/layer_26/sae_weights.safetensors` and `cfg.json`

## Dataset

The discovery set (`data_process/dataset/balanced_city_pairs.jsonl`, 180 directed prompts) and the city-disjoint held-out set (`data_process/dataset/holdout_city_pairs.jsonl`) are included in this repository. They can be rebuilt from GeoNames with `data_process/main.py` and `data_process/build_holdout.py`.

## Stage 1 — Component localization

```bash
python component_localization/patching.py              # activation patching (MLP + attention heads)
python component_localization/patching_city_a.py       # c_A-position patching
python component_localization/plot_patching_results.py # Fig. 3 activation-patching heatmap
```

Outputs: the normalized patching score `m` per layer and token position, the critical layers (layer 8 for the city positions and layer 26 for the final token), and the routing heads Head 24/25 (Table II).

## Stage 2 — Feature discovery

```bash
python feature_discovery/city_feature_L8.py   # layer-8 geographic features (Moran's I + BH-FDR + LISA)
python feature_discovery/cdla_rank_L26.py     # layer-26 direction features (cDLA ranking + bootstrap)
```

Outputs: the 21 FDR-significant geographic features and the four top-1 cDLA direction features (`#47636`/`#56725`/`#10801`/`#26800`).

## Stage 3 — Causal verification

```bash
python causal_verification/l8_geographic_subspace_ablation_ca_joint.py  # geographic ablation at c_A / c_B / joint
python causal_verification/l26_direction_feature_ablation.py            # direction-feature ablation
python causal_verification/head_ablation_delta_logit.py                 # routing-head ablation
python causal_verification/l8_geographic_to_l26_direction_mediation.py  # upstream mediation (geographic -> heads -> direction)
python causal_verification/l8_geographic_bypass_layerscan.py            # bypass layer scan (layers 9-25)
python causal_verification/cluster_bootstrap.py                         # pair-level cluster bootstrap
```

Outputs: the ablation effects (Table V) and the upstream mediation fraction (60% through the heads).

## Stage 4 — Circuit attribution

```bash
python circuit_tracing/verify_head24_attention.py             # Head 24 attention summaries (cached JSON)
python circuit_tracing/trace_city_a_path.py                   # Head 25 attention summaries (cached JSON)
python circuit_tracing/head_to_direction_feature_activation.py  # head -> direction-feature activation change (Sec. 4.4)
python circuit_tracing/circuit_grid.py                          # Fig. 5 circuit map (per-prompt edge weights)
python circuit_tracing/circuit.py                               # four-direction overview diagrams (supplementary)
```

Outputs: the Fig. 5 circuit map (`circuit_tracing/figures/circuit_south.pdf`) and per-prompt edge weights (`circuit_tracing/results/circuit_perprompt.json`). The overview script `circuit.py` reads the cached attention summaries produced by the first two scripts above (`head24_attention.json`, `city_a_attn_path.json`).

## Supplementary analyses

```bash
python causal_verification/prediction_rule_comparison.py   # flat-argmax vs axis-based prediction rule
python causal_verification/correct_incorrect_split.py      # correct/incorrect prediction split
python causal_verification/ablation_margin_and_dla.py      # flip rate, winning margin, cDLA vs ablation
python causal_verification/mediation_cluster_bootstrap.py  # cluster bootstrap for the mediation effects
python causal_verification/layer_direction_selectivity.py  # logit-lens direction selectivity
python feature_discovery/diagnose_seed_sae.py              # seed-1234 SAE diagnostic (robustness)
```

Each script prints a usage summary when run with `--help`, and its docstring lists the expected inputs and outputs.
