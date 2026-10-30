"""
Build a multi-source geographic training corpus for the SAE.

Generates natural-language descriptions of absolute city coordinates (from
cities15000.txt) and relative direction/distance (from balanced_city_pairs.jsonl),
mixes them with WikiText-103 at a 2:8 geographic:generic ratio, packs the texts
into fixed-length (max_length=256) token blocks separated by EOS, and writes them
as a HuggingFace Dataset, a Parquet file, or a JSONL file.

Inputs (relative to the project root):
  data_process/dataset/wikitext_103.txt
  data_process/dataset/balanced_city_pairs.jsonl
  data_process/row_data/cities15000.txt
Default output:
  data_process/dataset/packed_corpus.jsonl
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

# ── optional dependency (falls back to writing jsonl if absent) ──────────────
try:
    from datasets import Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Paths (relative to the project root)
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WIKI_TOKENS   = _PROJECT_ROOT / "data_process" / "dataset" / "wikitext_103.txt"
_CITY_PAIRS    = _PROJECT_ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
_CITIES15K     = _PROJECT_ROOT / "data_process" / "row_data"  / "cities15000.txt"
_DEFAULT_OUT   = _PROJECT_ROOT / "data_process" / "dataset" / "packed_corpus.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorpusConfig:
    # ---------- paths ----------
    wiki_path:        Path = _WIKI_TOKENS
    city_pairs_path:  Path = _CITY_PAIRS
    cities15k_path:   Path = _CITIES15K
    output_path:      Path = _DEFAULT_OUT

    # ---------- sampling hyperparameters ----------
    geo_ratio:        float = 0.2      # fraction of geographic text in the mix
    max_length:       int   = 256      # tokens per packed block (incl. EOS)
    seed:             int   = 42

    # ---------- model / tokenizer ----------
    model_name:       str   = "Qwen/Qwen3-8B"
    # without a tokenizer, approximate max_length by characters (1 token ~ 3 chars)
    use_tokenizer:    bool  = True
    # max wiki lines to read (0 = all; limit for large corpora)
    wiki_max_lines:   int   = 0

    # ---------- multilingual templates ----------
    use_chinese_templates: bool = False  # if True, also generate Chinese descriptions


# ─────────────────────────────────────────────────────────────────────────────
# cities15000.txt column indices (GeoNames TSV, 19 columns)
# see: https://download.geonames.org/export/dump/readme.txt
# ─────────────────────────────────────────────────────────────────────────────
_COL_GEONAMEID    = 0
_COL_NAME         = 1   # primary name (English)
_COL_ASCIINAME    = 2   # ASCII name
_COL_ALTERNAMES   = 3   # comma-separated multilingual aliases
_COL_LAT          = 4
_COL_LON          = 5


def _parse_cities15k(path: Path) -> List[dict]:
    """Parse GeoNames cities15000.txt into records with name/alt_names/lat/lon."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:          # skip blank or malformed lines
                continue
            name = parts[_COL_NAME].strip()
            alt_raw = parts[_COL_ALTERNAMES].strip()
            alt_names = [a.strip() for a in alt_raw.split(",") if a.strip()] if alt_raw else []
            try:
                lat = float(parts[_COL_LAT])
                lon = float(parts[_COL_LON])
            except ValueError:
                continue
            records.append({"name": name, "alt_names": alt_names, "lat": lat, "lon": lon})
    return records


def _city_abs_templates(rec: dict, use_chinese: bool = False) -> List[str]:
    """Generate natural-language sentences for one city record (optionally Chinese)."""
    name = rec["name"]
    lat  = round(rec["lat"], 4)
    lon  = round(rec["lon"], 4)
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    lat_abs = abs(lat)
    lon_abs = abs(lon)

    texts = [
        f"The exact coordinates of {name} are latitude {lat} and longitude {lon}.",
        f"{name} is located at {lat_abs}°{lat_dir}, {lon_abs}°{lon_dir}.",
        f"Geographically, {name} lies at ({lat}, {lon}) on the world map.",
    ]
    # optionally add a sentence with a random alias for variety
    if rec["alt_names"]:
        alt = random.choice(rec["alt_names"])
        texts.append(f"{alt} (also known as {name}) has coordinates {lat}°, {lon}°.")

    if use_chinese:
        texts.append(f"{name}的地理坐标为北纬（南纬）{lat_abs}度，东经（西经）{lon_abs}度。")
        texts.append(f"{name}位于经度{lon}、纬度{lat}的位置。")

    return texts


def _parse_city_pairs(path: Path) -> List[dict]:
    """Read balanced_city_pairs.jsonl (one JSON object per line)."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _city_rel_templates(rec: dict, use_chinese: bool = False) -> List[str]:
    """Generate relative direction/distance sentences for a city-pair record.

    ``rec`` must contain the fields City_A, City_B, Direction and Distance.
    """
    a    = rec.get("City_A", "Unknown")
    b    = rec.get("City_B", "Unknown")
    dirn = rec.get("Direction", "unknown direction")
    dist = rec.get("Distance", None)

    dist_str = f"{round(float(dist), 1)} km" if dist is not None else "an unknown distance"

    texts = [
        f"{b} is located to the {dirn} of {a}, approximately {dist_str} away.",
        f"Traveling from {a} to {b}, you head {dirn} for about {dist_str}.",
        f"The city of {b} lies {dist_str} to the {dirn} of {a}.",
    ]

    # add a precise sentence using the bearing
    bearing = rec.get("Bearing", None)
    if bearing is not None:
        texts.append(
            f"The bearing from {a} to {b} is approximately {round(float(bearing), 1)}°, "
            f"a distance of {dist_str}."
        )

    if use_chinese:
        texts.append(f"{b}位于{a}的{dirn}方向，距离约{dist_str}。")

    return texts


def _read_wiki_lines(path: Path, max_lines: int = 0) -> List[str]:
    """Read WikiText-103, one pre-tokenized line per paragraph.

    ``max_lines=0`` reads everything, otherwise only the first ``max_lines`` lines.
    """
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_lines > 0 and i >= max_lines:
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Packing utilities
# ─────────────────────────────────────────────────────────────────────────────

class _TokenPacker:
    """Concatenate texts and split into fixed-length token blocks, EOS-separated."""

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        # Qwen's EOS token is "<|endoftext|>" (id may vary by version); resolve it dynamically
        self.eos_id: int = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("Tokenizer did not return eos_token_id; check the model config.")

    def pack(self, texts: List[str]) -> List[List[int]]:
        """Pack all texts into a token stream and chunk it into blocks of exactly ``max_length`` tokens."""
        # tokenize in batches to avoid very long sequences
        token_stream: List[int] = []
        batch_size = 512  # texts per batch (heuristic, adjustable)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # return_attention_mask=False to save memory
            enc = self.tokenizer(
                batch,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            for ids in enc["input_ids"]:
                token_stream.extend(ids)
                token_stream.append(self.eos_id)  # document separator

        # chunk into blocks
        chunks: List[List[int]] = []
        for start in range(0, len(token_stream) - self.max_length + 1, self.max_length):
            chunks.append(token_stream[start : start + self.max_length])
        return chunks


class _CharPacker:
    """Tokenizer-free fallback packer (approximates one token as 4 characters)."""

    CHARS_PER_BLOCK = 256 * 4  # heuristic

    def pack(self, texts: List[str]) -> List[List[str]]:
        joined = " <EOS> ".join(texts)
        blocks = []
        for i in range(0, len(joined) - self.CHARS_PER_BLOCK + 1, self.CHARS_PER_BLOCK):
            blocks.append([joined[i : i + self.CHARS_PER_BLOCK]])
        return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Public interface: GeoCorpusBuilder
# ─────────────────────────────────────────────────────────────────────────────

class GeoCorpusBuilder:
    """Build the mixed geographic corpus.

    Usage:
        cfg = CorpusConfig(use_tokenizer=True)
        builder = GeoCorpusBuilder(cfg)
        builder.build_and_save()

    Output is a JSONL of ``{"input_ids": [int, ...]}`` lines, or a HuggingFace
    Dataset directory if available.
    """

    def __init__(self, config: CorpusConfig):
        self.cfg = config
        random.seed(config.seed)

        # initialize tokenizer if enabled
        self._tokenizer = None
        if config.use_tokenizer:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Please install transformers first: pip install transformers"
                )
            # set offline/mirror environment variables
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print(f"[DataBuilder] loading tokenizer: {config.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )

    # ──────────────────────────── text generation ─────────────────────────────

    def _gen_geo_texts(self) -> List[str]:
        """Generate and shuffle all geographic description texts (absolute + relative)."""
        texts: List[str] = []

        # absolute-coordinate descriptions
        print("[DataBuilder] parsing cities15000.txt ...")
        cities = _parse_cities15k(self.cfg.cities15k_path)
        for rec in cities:
            texts.extend(_city_abs_templates(rec, self.cfg.use_chinese_templates))

        # relative direction/distance descriptions
        print("[DataBuilder] parsing balanced_city_pairs.jsonl ...")
        pairs = _parse_city_pairs(self.cfg.city_pairs_path)
        for rec in pairs:
            texts.extend(_city_rel_templates(rec, self.cfg.use_chinese_templates))

        random.shuffle(texts)
        print(f"[DataBuilder] total geographic sentences: {len(texts)}")
        return texts

    def _gen_wiki_texts(self) -> List[str]:
        """Read WikiText-103 lines."""
        print("[DataBuilder] reading wiki 103 ...")
        lines = _read_wiki_lines(self.cfg.wiki_path, self.cfg.wiki_max_lines)
        print(f"[DataBuilder] Wiki text lines: {len(lines)}")
        return lines

    # ──────────────────────────── mixing / sampling ───────────────────────────

    def _mix_texts(self, geo: List[str], wiki: List[str]) -> List[str]:
        """Draw WikiText at a 1:4 geo:wiki ratio without replacement (when possible)."""
        # 1. total geo texts (no repeats)
        n_geo = len(geo)

        # 2. wiki count needed for the 2:8 (i.e. 1:4) ratio
        n_wiki = n_geo * 4

        # 3. resample with replacement if wiki is insufficient, else sample without replacement
        if len(wiki) < n_wiki:
            print("[Warning] Wiki data is less than 4x; resampling with replacement.")
            sampled_wiki = random.choices(wiki, k=n_wiki)
        else:
            sampled_wiki = random.sample(wiki, k=n_wiki)
            
        mixed = geo + sampled_wiki
        random.shuffle(mixed)
        
        print(f"[DataBuilder] mixed without repetition: geo={n_geo}, wiki={len(sampled_wiki)}, {len(mixed)} texts total")
        return mixed

    # ──────────────────────────── packing & saving ────────────────────────────────

    def build_and_save(
        self,
        output_path: Optional[Path] = None,
        save_as_hf: bool = False,
        save_as_parquet: bool = False,
    ) -> Path:
        """Build the mixed corpus and save it.

        Args:
            output_path: override the default output path in the config.
            save_as_hf: if True and HF is available, save as an HF Dataset directory.
            save_as_parquet: if True and Parquet is available, save as Parquet.

        Returns:
            The actual save path.
        """
        out = Path(output_path) if output_path else self.cfg.output_path
        out.parent.mkdir(parents=True, exist_ok=True)

        geo_texts  = self._gen_geo_texts()
        wiki_texts = self._gen_wiki_texts()
        mixed      = self._mix_texts(geo_texts, wiki_texts)

        if self._tokenizer is not None:
            packer = _TokenPacker(self._tokenizer, self.cfg.max_length)
            chunks = packer.pack(mixed)
            print(f"[DataBuilder] packed {len(chunks)} token blocks (block size={self.cfg.max_length})")

            if save_as_parquet and HF_AVAILABLE:
                parquet_dir = out.parent / "my_parquet_data"
                parquet_dir.mkdir(parents=True, exist_ok=True)
                parquet_out = parquet_dir / "packed_hf_dataset.parquet"
                ds = HFDataset.from_dict({"input_ids": chunks})
                ds.to_parquet(str(parquet_out))
                print(f"[DataBuilder] Parquet Dataset saved to: {parquet_out}")
                return parquet_out

            if save_as_hf and HF_AVAILABLE:
                hf_out = out.parent / "packed_hf_dataset"
                ds = HFDataset.from_dict({"input_ids": chunks})
                ds.save_to_disk(str(hf_out))
                print(f"[DataBuilder] HF Dataset saved to: {hf_out}")
                return hf_out

            # save as jsonl
            with open(out, "w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(json.dumps({"input_ids": chunk}, ensure_ascii=False) + "\n")
            print(f"[DataBuilder] jsonl saved to: {out}")

        else:
            # fallback: save raw text lines without tokenization
            txt_out = out.with_suffix(".txt")
            with open(txt_out, "w", encoding="utf-8") as fh:
                for line in mixed:
                    fh.write(line + "\n")
            print(f"[DataBuilder] raw text (no tokenization) saved to: {txt_out}")
            out = txt_out

        return out


# ─────────────────────────────────────────────────────────────────────────────
# PackedDataset: load the packed corpus for activation collection / training
# ─────────────────────────────────────────────────────────────────────────────

class PackedDataset:
    """Lightweight dataset that streams ``input_ids`` lists from packed_corpus.jsonl.

    Implements ``__len__`` and ``__getitem__`` for the PyTorch DataLoader interface.
    """

    def __init__(self, jsonl_path: Path):
        self.path = Path(jsonl_path)
        self._data: List[List[int]] = []
        self._load()

    def _load(self):
        print(f"[PackedDataset] loading: {self.path}")
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._data.append(obj["input_ids"])
        print(f"[PackedDataset] {len(self._data)} sequences")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> List[int]:
        return self._data[idx]

    def iter_batches(self, batch_size: int = 32) -> Iterator[List[List[int]]]:
        """Simple (non-shuffled) batch iterator for activation collection."""
        for start in range(0, len(self._data), batch_size):
            yield self._data[start : start + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the SAE training corpus")
    parser.add_argument("--geo_ratio",    type=float, default=0.2,          help="Fraction of geographic text (default 0.2)")
    parser.add_argument("--max_length",   type=int,   default=256,          help="Sequence block length (default 256)")
    parser.add_argument("--wiki_max_lines", type=int, default=0,            help="Max wiki lines to read (0 = all)")
    parser.add_argument("--no_tokenizer", action="store_true",              help="Disable the tokenizer (for debugging)")
    parser.add_argument("--chinese",      action="store_true",              help="Also add Chinese templates")
    parser.add_argument("--save_hf",      action="store_true",              help="Save as an HF Dataset")
    parser.add_argument("--save_parquet", action="store_true",              help="Save as Parquet (for sae_lens)")
    parser.add_argument("--output",       type=str,   default=str(_DEFAULT_OUT), help="Output path")
    args = parser.parse_args()

    cfg = CorpusConfig(
        geo_ratio           = args.geo_ratio,
        max_length          = args.max_length,
        wiki_max_lines      = args.wiki_max_lines,
        use_tokenizer       = not args.no_tokenizer,
        use_chinese_templates = args.chinese,
        output_path         = Path(args.output),
    )
    builder = GeoCorpusBuilder(cfg)
    saved_path = builder.build_and_save(save_as_hf=args.save_hf, save_as_parquet=args.save_parquet)
    print(f"[DataBuilder] done, output path: {saved_path}")
