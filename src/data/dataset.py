"""Dataset classes for text simplification"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import DataConfig


class TextSimplificationDataset(Dataset):
    """Dataset for text simplification task"""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_target_length: int = 64,
        task_prefix: str = "simplify: ",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        source_text = self.task_prefix + item["source"]
        target_text = item["target"]

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class InferenceDataset(Dataset):
    """Dataset for inference (no labels)"""

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        task_prefix: str = "simplify: ",
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.task_prefix = task_prefix

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.task_prefix + self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


@dataclass
class DataCollator:
    """Data collator for batching"""

    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        if "input_ids" in features[0]:
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])

        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.stack(
                [f["attention_mask"] for f in features]
            )

        if "labels" in features[0]:
            labels = torch.stack([f["labels"] for f in features])
            batch["labels"] = labels

        return batch
