"""Data processor for downloading and preparing text simplification datasets"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from datasets import load_dataset

from src.utils import ensure_dir, save_json, load_json, setup_logging


logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SimplificationDataset(Dataset):
    """
    PyTorch Dataset for (complex → simple) pairs.

    Each item is already tokenized; __getitem__ returns tensors ready for T5.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: T5Tokenizer,
        max_source_len: int = 128,
        max_target_len: int = 64,
        source_prefix: str = "simplify: ",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        source_text = self.source_prefix + item["source"]
        target_text = item["target"]

        # Tokenize source
        source_enc = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target; T5 uses -100 to ignore padding in loss
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data Processor
# ---------------------------------------------------------------------------


class DataProcessor:
    """
    Download, align, augment, split, tokenize, and serve text-simplification data.

    Supported corpora
    -----------------
    * wiki_auto   – Wikipedia → Simple-English-Wikipedia alignments (HuggingFace)
    * sample      – Built-in synthetic pairs (no download needed)

    Newsela requires a signed data-sharing agreement from newsela.com/research.
    Once you have the TSV files, call `load_newsela(path)` to ingest them.
    """

    WIKI_AUTO_HF = "wiki_auto"  # HuggingFace dataset id
    T5_MODEL = "t5-small"  # tokenizer to use

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "data/cache",
    ):
        self.data_dir = ensure_dir(data_dir)
        self.cache_dir = ensure_dir(cache_dir)
        self._tokenizer: Optional[T5Tokenizer] = None

    # ------------------------------------------------------------------
    # Tokenizer (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> T5Tokenizer:
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.T5_MODEL}")
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.T5_MODEL,
                cache_dir=str(self.cache_dir),
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # 1. Download / load corpora
    # ------------------------------------------------------------------

    def download_wiki_auto(self) -> List[Dict]:
        """
        Download the wiki_auto dataset from HuggingFace Hub and convert it
        to the internal {id, source, target, level} format.

        wiki_auto contains ~670 k automatically-aligned sentence pairs from
        English Wikipedia → Simple English Wikipedia.

        Install requirement:  pip install datasets
        """
        logger.info("Downloading wiki_auto from HuggingFace Hub …")

        # wiki_auto has two configs; 'part_1' + 'part_2' together give full data
        pairs: List[Dict] = []
        for config in ("part_1", "part_2"):
            ds = load_dataset(
                self.WIKI_AUTO_HF,
                config,
                cache_dir=str(self.cache_dir),
            )
            split = ds["full"] if "full" in ds else ds[list(ds.keys())[0]]

            for row in split:
                # Schema: {'normal_sentence': str, 'simple_sentence': str, ...}
                src = (row.get("normal_sentence") or "").strip()
                tgt = (row.get("simple_sentence") or "").strip()
                if src and tgt and src != tgt:
                    pairs.append(
                        {
                            "id": len(pairs),
                            "source": src,
                            "target": tgt,
                            "level": "wiki_auto",
                        }
                    )

        out = self.data_dir / "wiki_auto_raw.json"
        save_json(pairs, str(out))
        logger.info(f"wiki_auto: {len(pairs):,} pairs saved to {out}")
        return pairs

    def download_simple_wiki(self) -> List[Dict]:
        """
        Download the plain Simple-English-Wikipedia dump via HuggingFace
        (`wikipedia` dataset, language='simple').

        NOTE: This gives *articles*, not pre-aligned pairs. Use it to build
        a vocabulary / language-model baseline, or align manually against
        the standard English Wikipedia.

        Install requirement:  pip install datasets
        """
        logger.info("Downloading Simple-English Wikipedia …")
        ds = load_dataset(
            "wikipedia",
            "20220301.simple",
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
        )
        articles = [
            {"id": i, "title": row["title"], "text": row["text"]}
            for i, row in enumerate(ds["train"])
        ]
        out = self.data_dir / "simple_wiki_articles.json"
        save_json(articles, str(out))
        logger.info(f"Simple Wiki: {len(articles):,} articles saved to {out}")
        return articles

    def download_kaggle_simple_wiki(self) -> List[Dict]:
        """
        Download Simple English Wikipedia dataset from Kaggle using kagglehub.

        Dataset: "plain-text-wikipedia-simpleenglish" by ffatty

        Returns aligned (normal → simple) sentence pairs in internal format.
        """
        try:
            import kagglehub
        except ImportError:
            logger.warning("kagglehub not installed. Installing now...")
            import subprocess

            subprocess.check_call(["pip", "install", "-q", "kagglehub"])
            import kagglehub

        logger.info("Downloading Simple English Wikipedia from Kaggle...")
        path = kagglehub.dataset_download("ffatty/plain-text-wikipedia-simpleenglish")
        logger.info(f"Kaggle dataset path: {path}")

        # Look for the data files in the downloaded path
        data_path = Path(path)

        # Try to find the data file - check common patterns
        possible_files = (
            list(data_path.glob("**/*.txt"))
            + list(data_path.glob("**/*.csv"))
            + list(data_path.glob("**/*.json"))
        )

        pairs: List[Dict] = []

        for file_path in possible_files:
            logger.info(f"Processing file: {file_path}")
            try:
                if file_path.suffix == ".txt":
                    # Try to parse as aligned text file
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if "\t" in line:
                                parts = line.split("\t")
                                if len(parts) >= 2:
                                    src, tgt = parts[0].strip(), parts[1].strip()
                                    if src and tgt and src != tgt:
                                        pairs.append(
                                            {
                                                "id": len(pairs),
                                                "source": src,
                                                "target": tgt,
                                                "level": "kaggle_simple_wiki",
                                            }
                                        )
                elif file_path.suffix == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                src = (
                                    item.get("source")
                                    or item.get("original")
                                    or item.get("normal")
                                    or item.get("complex", "")
                                )
                                tgt = (
                                    item.get("target")
                                    or item.get("simplified")
                                    or item.get("simple", "")
                                )
                                if src and tgt and src != tgt:
                                    pairs.append(
                                        {
                                            "id": len(pairs),
                                            "source": src,
                                            "target": tgt,
                                            "level": "kaggle_simple_wiki",
                                        }
                                    )
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

        # If no pairs found, create sample data as fallback
        if not pairs:
            logger.warning("No pairs found in Kaggle dataset. Using sample data.")
            return self.create_sample_data(5000)

        out = self.data_dir / "kaggle_simple_wiki_raw.json"
        save_json(pairs, str(out))
        logger.info(f"Kaggle Simple Wiki: {len(pairs):,} pairs saved to {out}")
        return pairs

    def load_newsela(self, tsv_path: str) -> List[Dict]:
        """
        Ingest Newsela parallel data from the TSV file you receive after
        signing the data-sharing agreement at newsela.com/research.

        Expected TSV columns (no header):
            normal_sentence <TAB> simple_sentence [<TAB> grade_level]

        Returns pairs in the internal format.
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(
                f"{tsv_path} not found.\n"
                "Request the Newsela corpus at https://newsela.com/data/"
            )

        pairs: List[Dict] = []
        with tsv_path.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                src, tgt = parts[0].strip(), parts[1].strip()
                level = parts[2].strip() if len(parts) > 2 else "newsela"
                if src and tgt:
                    pairs.append(
                        {
                            "id": len(pairs),
                            "source": src,
                            "target": tgt,
                            "level": level,
                        }
                    )

        out = self.data_dir / "newsela_raw.json"
        save_json(pairs, str(out))
        logger.info(f"Newsela: {len(pairs):,} pairs saved to {out}")
        return pairs

    # ------------------------------------------------------------------
    # 2. Synthetic / fallback data (original method, kept intact)
    # ------------------------------------------------------------------

    def create_sample_data(self, num_samples: int = 1000) -> List[Dict]:
        """Create built-in synthetic parallel data (no download required)."""
        sample_pairs = [
            {
                "source": "The implementation utilizes sophisticated algorithms.",
                "target": "The program uses advanced methods.",
            },
            {
                "source": "The phenomenon demonstrates significant complexity.",
                "target": "The event shows much complexity.",
            },
            {
                "source": "His demeanor appeared particularly taciturn.",
                "target": "He seemed very quiet.",
            },
            {
                "source": "The methodology employed was comprehensive.",
                "target": "The method used was complete.",
            },
            {
                "source": "Substantial evidence supports the hypothesis.",
                "target": "Much evidence supports the idea.",
            },
            {
                "source": "The mechanism functions optimally.",
                "target": "The system works well.",
            },
            {
                "source": "The document provides comprehensive documentation.",
                "target": "The paper gives complete information.",
            },
            {
                "source": "Significant advancements have been achieved.",
                "target": "Big improvements have been made.",
            },
            {
                "source": "The analysis requires meticulous attention.",
                "target": "The study needs careful focus.",
            },
            {
                "source": "The subsequent evaluation revealed anomalies.",
                "target": "The later review showed problems.",
            },
        ]
        variations = [
            lambda s: s,
            lambda s: s.lower(),
            lambda s: "In summary, " + s.lower(),
            lambda s: "Essentially, " + s.lower(),
        ]
        data = []
        for i in range(num_samples):
            pair = sample_pairs[i % len(sample_pairs)].copy()
            v = variations[i % len(variations)]
            data.append(
                {
                    "id": i,
                    "source": v(pair["source"]),
                    "target": v(pair["target"]),
                    "level": "intermediate",
                }
            )
        out = self.data_dir / "train_data.json"
        save_json(data, str(out))
        logger.info(f"Created {len(data)} sample examples at {out}")
        return data

    # ------------------------------------------------------------------
    # 3. Augmentation
    # ------------------------------------------------------------------

    def augment_data(
        self,
        data: List[Dict],
        augmentation_factor: int = 3,
    ) -> List[Dict]:
        """Augment training data with lightweight surface-form strategies."""
        strategies = [
            lambda s: s,
            lambda s: s.lower(),
            lambda s: s.replace(".", " .").replace(",", " , "),
            lambda s: "In summary, " + s.lower(),
            lambda s: "Essentially, " + s.lower(),
        ]
        augmented: List[Dict] = []
        for item in data:
            for strategy in strategies[:augmentation_factor]:
                augmented.append(
                    {
                        "id": len(augmented),
                        "source": strategy(item["source"]),
                        "target": strategy(item["target"]),
                        "level": item.get("level", "intermediate"),
                    }
                )
        logger.info(f"Augmented {len(data)} → {len(augmented)} examples")
        return augmented

    # ------------------------------------------------------------------
    # 4. Train / val split
    # ------------------------------------------------------------------

    def split_data(
        self,
        data: List[Dict],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        random.seed(seed)
        shuffled = data.copy()
        random.shuffle(shuffled)
        idx = int(len(shuffled) * train_ratio)
        train_data, val_data = shuffled[:idx], shuffled[idx:]
        logger.info(f"Split → {len(train_data)} train / {len(val_data)} val")
        return train_data, val_data

    def save_splits(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        train_file: str = "train.json",
        val_file: str = "val.json",
    ) -> None:
        save_json(train_data, str(self.data_dir / train_file))
        save_json(val_data, str(self.data_dir / val_file))
        logger.info(f"Splits saved to {self.data_dir}")

    def load_splits(
        self,
        train_file: str = "train.json",
        val_file: str = "val.json",
    ) -> Tuple[List[Dict], List[Dict]]:
        return (
            load_json(str(self.data_dir / train_file)),
            load_json(str(self.data_dir / val_file)),
        )

    # ------------------------------------------------------------------
    # 5. DataLoaders (tokenized, batched, ready for training)
    # ------------------------------------------------------------------

    def get_dataloaders(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        batch_size: int = 16,
        max_source_len: int = 128,
        max_target_len: int = 64,
        num_workers: int = 2,
        source_prefix: str = "simplify: ",
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Tokenize both splits and return (train_loader, val_loader).

        Parameters
        ----------
        train_data / val_data : output of split_data()
        batch_size            : samples per gradient step
        max_source_len        : T5 input token budget
        max_target_len        : T5 output token budget
        num_workers           : parallel DataLoader workers
        source_prefix         : task prompt prepended to every source sentence

        Returns
        -------
        (train_loader, val_loader) – ready to pass directly into your training loop
        """
        common = dict(
            tokenizer=self.tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            source_prefix=source_prefix,
        )

        train_ds = SimplificationDataset(train_data, **common)
        val_ds = SimplificationDataset(val_data, **common)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            f"DataLoaders ready — "
            f"train: {len(train_loader)} batches, "
            f"val: {len(val_loader)} batches "
            f"(batch_size={batch_size})"
        )
        return train_loader, val_loader
