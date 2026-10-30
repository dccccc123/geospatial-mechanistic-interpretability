#!/usr/bin/env python3
"""
Download and prepare WikiText-103 for the SAE training corpus.

Downloads the space-tokenized WikiText-103 training split (Salesforce/wikitext,
config "wikitext-103-v1") from Hugging Face, keeps paragraphs longer than 50
characters, and writes them one paragraph per line to
``data_process/dataset/wikitext_103.txt``.

Run:
    python data_process/download_wikitext.py

The resulting file is a required input for ``sae/data_builder.py``.
"""
import os
from pathlib import Path

# Optional mirror for users in mainland China (uncomment if HF is unreachable):
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import datasets

ROOT = Path(__file__).resolve().parent
TARGET_FILE = ROOT / "dataset" / "wikitext_103.txt"
MIN_CHARS = 50  # drop short fragments (same filter used to produce the paper's file)


def main():
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("[Prepare] downloading WikiText-103 from Hugging Face ...")
    dataset = datasets.load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-v1",
        split="train",
    )

    valid_texts = [
        item["text"].strip() for item in dataset
        if len(item["text"].strip()) > MIN_CHARS
    ]

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        for line in valid_texts:
            f.write(line.replace("\n", " ") + "\n")

    print(f"[Done] saved {len(valid_texts)} paragraphs to {TARGET_FILE}")


if __name__ == "__main__":
    main()
