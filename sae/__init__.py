# sae/__init__.py
# Sparse autoencoder (SAE) training utilities for extracting internal
# representations of geographic relationships from Qwen3-8B.


from .data_builder import GeoCorpusBuilder, PackedDataset

__all__ = [
    "GeoCorpusBuilder",
    "PackedDataset",
]
