# Mechanistic Interpretability of Geographic Direction Reasoning in Large Language Models

This repository contains the code for the paper:

> **Mechanistic Interpretability of Geographic Direction Reasoning in Large Language Models: A Case Study on Qwen3-8B**

## Overview

We present a circuit-level causal analysis of how the Qwen3-8B language model computes the cardinal direction between two Chinese cities. Using a four-stage framework — (1) **component localization** via activation patching, (2) **feature discovery** with sparse autoencoders (SAEs), (3) **causal verification** via ablation, and (4) **circuit attribution** — we identify a circuit that routes layer-8 geographic features through two routing attention heads (Head 24 and Head 25) to a layer-26 direction feature, which then drives the final direction prediction.

## Repository structure

```
.
├── data_process/           # City filtering and direction-pair dataset construction
├── prompt_validation/      # Prompt-format validation (Table I)
├── component_localization/ # Stage 1: activation patching / path patching
├── sae/                    # SAE corpus building and BatchTopK SAE training
├── feature_discovery/      # Stage 2: geographic features (Moran's I) and direction features (cDLA)
├── causal_verification/    # Stage 3: ablation, mediation, cluster bootstrap
├── circuit_tracing/        # Stage 4: circuit assembly and visualization
├── linear_probe/           # Layerwise linear probes (supplementary)
├── paper_plot/             # Plotting scripts for the paper figures
├── docs/                   # Reproduction guide and user guide
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT license
```

## Installation

**Prerequisites**

- Python 3.12
- A single NVIDIA GPU with at least 48 GB of VRAM (e.g., an RTX 4090). Training one SAE takes roughly 6 hours on this hardware.

**Install dependencies**

```bash
pip install -r requirements.txt
```

The key dependencies are `torch`, `transformers`, `sae-lens` (v6.39.0), `transformer-lens`, `numpy`, `scipy`, `matplotlib`, `esda`/`libpysal` (spatial autocorrelation), `safetensors`, and `tqdm`. The `sae-lens` version is pinned because the saved SAE weights use its threshold-gated inference format.

## Data and model weights

- **City pairs** — `data_process/dataset/balanced_city_pairs.jsonl` (discovery set, 180 directed prompts) and `holdout_city_pairs.jsonl` (city-disjoint held-out set) are included in this repository.
- **Model** — the Qwen3-8B checkpoint is downloaded automatically from Hugging Face (`Qwen/Qwen3-8B`).
- **SAE weights** — the trained layer-8 and layer-26 SAEs (~2.1 GB each) are too large for GitHub and are hosted on Hugging Face at [dc-321/qwen3-geospatial-sae-weights](https://huggingface.co/dc-321/qwen3-geospatial-sae-weights). Download them and place the files under `sae/final_sae/layer_8/` and `sae/final_sae/layer_26/`.

## Reproduction

A step-by-step reproduction guide covering all four stages is provided in [`docs/reproduction.md`](docs/reproduction.md). In brief:

1. **Component localization** — `component_localization/patching.py` produces the activation-patching heatmap (Fig. 3).
2. **Feature discovery** — `feature_discovery/city_feature_L8.py` discovers layer-8 geographic features (Moran's I with a Benjamini-Hochberg correction), and `feature_discovery/cdla_rank_L26.py` ranks layer-26 direction features by contrastive Direct Logit Attribution (cDLA).
3. **Causal verification** — `causal_verification/l8_geographic_subspace_ablation_ca_joint.py`, `l26_direction_feature_ablation.py`, and `head_ablation_delta_logit.py` run the main ablations; the mediation and bootstrap scripts run the pathway and robustness analyses.
4. **Circuit attribution** — `circuit_tracing/circuit.py` assembles the circuit map (Fig. 5).

Each script prints a usage summary when run with `--help`, and its docstring lists the expected inputs and outputs. A consolidated user guide is available in [`docs/user_guide.md`](docs/user_guide.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code or dataset in your research, please cite the paper:

```
@article{han2026geographic,
  title   = {Mechanistic Interpretability of Geographic Direction Reasoning in Large Language Models: A Case Study on Qwen3-8B},
  author  = {Han, Hongxin and Ding, Can and Fang, Shiteng and Li, Jiasheng and Tong, Deyu},
  journal = {Computers \& Geosciences},
  year    = {2026},
}
```

*(BibTeX entry to be updated with the final volume/DOI upon publication.)*
