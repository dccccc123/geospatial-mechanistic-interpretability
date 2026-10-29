"""
Sparse autoencoder (SAE) training pipeline for Qwen3-8B.

Trains a BatchTopK SAE on selected transformer layers with the sae_lens
streaming runner. The model is loaded through its native HuggingFace structure
(AutoModelForCausalLM) for compatibility, and activations are streamed online to
fit within 48 GB of VRAM. Layers are trained sequentially with GPU memory cleanup
between them; --smoke_test runs a short check for environment/OOM issues.

Directory layout (relative to the project root):
  data_process/dataset/my_parquet_data/  <- tokenized training corpus (input)
  sae/checkpoints/                       <- mid-training checkpoints
  sae/final_sae/                         <- final saved SAE weights
"""

import argparse
import gc
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)
import sae_lens.training.activations_store
sae_lens.training.activations_store.validate_pretokenized_dataset_tokenizer = lambda *args, **kwargs: None

# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to the project root)
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET_PATH = _PROJECT_ROOT / "data_process" / "dataset" / "my_parquet_data"
_DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "sae" / "checkpoints"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "sae" / "final_sae"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SAETrainConfig:
    # ---------- paths ----------
    dataset_path: Path = _DEFAULT_DATASET_PATH
    checkpoint_dir: Path = _DEFAULT_CHECKPOINT_DIR
    output_dir: Path = _DEFAULT_OUTPUT_DIR

    # ---------- model and target layers ----------
    model_name: str = "Qwen/Qwen3-8B"
    target_layers: List[int] = field(default_factory=lambda: [8, 26])
    context_size: int = 256
    
    # ---------- SAE architecture (BatchTopK) ----------
    d_in: int = 4096
    expansion_factor: int = 16
    k: int = 32
    
    # ---------- training / hardware (48 GB VRAM) ----------
    lr: float = 1e-4
    train_batch_size_tokens: int = 4096
    store_batch_size_prompts: int = 32  # lower this to avoid OOM
    n_batches_in_buffer: int = 64
    seed: int = 42  # random seed
    
    # ---------- run-mode control ----------
    smoke_test: bool = False
    total_tokens: int = 80_000_000  # total training tokens


# ─────────────────────────────────────────────────────────────────────────────
# Core class: SAE trainer
# ─────────────────────────────────────────────────────────────────────────────
class QwenSAETrainer:
    """Wraps SAE configuration and streaming training logic."""

    def __init__(self, config: SAETrainConfig):
        self.cfg = config
        
        # apply smoke-test overrides
        if self.cfg.smoke_test:
            print("[Trainer] running in Smoke Test mode")
            self.cfg.total_tokens = 5_000_000
            self._log_to_wandb = False
            self._n_checkpoints = 0
        else:
            self._log_to_wandb = True
            self._n_checkpoints = 2

        # ensure output directories exist
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_sae_config(self) -> BatchTopKTrainingSAEConfig:
        """Build the BatchTopK architecture config."""
        return BatchTopKTrainingSAEConfig(
            k=self.cfg.k,
            d_in=self.cfg.d_in,
            d_sae=self.cfg.d_in * self.cfg.expansion_factor,
            normalize_activations="expected_average_only_in",
        )

    def _build_runner_config(self, target_layer: int) -> LanguageModelSAERunnerConfig:
        """Build the global runner config for a given layer."""

        # use HuggingFace's naming convention
        hook_point = f"model.layers.{target_layer}"
        layer_ckpt_dir = self.cfg.checkpoint_dir / f"layer_{target_layer}"
        layer_ckpt_dir.mkdir(exist_ok=True)

        return LanguageModelSAERunnerConfig(
            # --- model & hook ---
            model_name=self.cfg.model_name,
            model_class_name="AutoModelForCausalLM",  # required for Qwen 3
            hook_name=hook_point,
            
            # --- dataset ---
            dataset_path=str(self.cfg.dataset_path),
            is_dataset_tokenized=True,
            context_size=self.cfg.context_size,
            training_tokens=self.cfg.total_tokens,
            
            # --- SAE architecture ---
            sae=self._build_sae_config(),
            
            # --- hardware / memory ---
            device="cuda",
            dtype="bfloat16",
            model_from_pretrained_kwargs={"torch_dtype": "bfloat16"},
            use_cached_activations=False,  # online streaming
            store_batch_size_prompts=self.cfg.store_batch_size_prompts,
            n_batches_in_buffer=self.cfg.n_batches_in_buffer,
            
            # --- optimizer & logging ---
            lr=self.cfg.lr,
            train_batch_size_tokens=self.cfg.train_batch_size_tokens,
            seed=self.cfg.seed,
            lr_scheduler_name="constant",
            lr_warm_up_steps=self.cfg.total_tokens // (self.cfg.train_batch_size_tokens * 10), # 10% warmup
            
            logger=LoggingConfig(
                log_to_wandb=self._log_to_wandb,
                wandb_project="qwen3-geo-sae" if self._log_to_wandb else None,
            ),
            checkpoint_path=str(layer_ckpt_dir),
            n_checkpoints=self._n_checkpoints,
        )

    def train_layer(self, layer: int):
        """Train the SAE for a single layer."""
        print(f"\n{'='*50}")
        print(f"[Trainer] preparing SAE training for Layer {layer}...")
        print(f"{'='*50}")

        runner_cfg = self._build_runner_config(layer)

        # SAELens loads the model when the runner is instantiated
        runner = LanguageModelSAETrainingRunner(runner_cfg)

        # ── register hook that zeroes outlier dimensions ────────────────────
        OUTLIER_DIMS = [2276, 233]

        def zero_outlier_hook(module, input, output):
            output[:, :, OUTLIER_DIMS] = 0.0
            return output

        hook_handle = runner.model.model.model.layers[layer].register_forward_hook(zero_outlier_hook)
        print(f"[Trainer] registered outlier-zeroing hook (dims={OUTLIER_DIMS}) on Layer {layer}")
        # ──────────────────────────────────────────────────────

        print(f"[Trainer] starting forward pass and streaming SAE training (target tokens: {self.cfg.total_tokens})...")
        trained_sae = runner.run()

        # remove the hook after training so it does not affect later layers
        hook_handle.remove()
        print(f"[Trainer] removed the hook for Layer {layer}")

        # save weights
        save_path = self.cfg.output_dir / f"layer_{layer}"
        trained_sae.save_inference_model(str(save_path))
        print(f"[Trainer] Layer {layer} training done! Saved to {save_path}")

    def run_all(self):
        """Train all target layers, managing GPU memory between them."""
        for layer in self.cfg.target_layers:
            self.train_layer(layer)
            
            # force GC and GPU cache clear after each layer to avoid OOM
            print(f"[Trainer] cleaning GPU memory for Layer {layer}...")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Trainer] GPU memory cleaned.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a spatial-geographic SAE for Qwen3-8B")
    
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 26], help="Target layers to train (default: 8 26)")
    parser.add_argument("--smoke_test", action="store_true", help="Enable smoke test (low tokens, no wandb)")
    parser.add_argument("--tokens", type=int, default=80_000_000, help="Total tokens for a full training run")
    parser.add_argument("--batch_prompts", type=int, default=32, help="Prompts per forward pass (lower if OOM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default sae/final_sae)")

    args = parser.parse_args()

    # initialize config
    config = SAETrainConfig(
        target_layers=args.layers,
        smoke_test=args.smoke_test,
        total_tokens=args.tokens,
        store_batch_size_prompts=args.batch_prompts,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR,
    )

    # start training
    trainer = QwenSAETrainer(config)
    trainer.run_all()