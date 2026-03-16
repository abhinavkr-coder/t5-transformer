"""Inference module for text simplification"""

import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig

from config import ModelConfig
from src.data import InferenceDataset
from src.utils import get_device, setup_logging


logger = setup_logging(__name__)


@dataclass
class SimplificationResult:
    """Result of text simplification"""

    original: str
    simplified: str
    level: str
    confidence: float = 0.0
    replaced_words: Optional[List[Dict[str, str]]] = None


class SimplificationInference:
    """Inference class for text simplification"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        self.tokenizer: Optional[T5Tokenizer] = None
        self.model: Optional[T5ForConditionalGeneration] = None
        self.generation_config: Optional[GenerationConfig] = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load model and tokenizer from path"""
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.model.eval()

        self.generation_config = GenerationConfig(
            max_new_tokens=self.model_config.max_target_length,
            num_beams=self.model_config.num_beams,
            early_stopping=self.model_config.early_stopping,
            temperature=self.model_config.temperature,
            top_k=self.model_config.top_k,
            top_p=self.model_config.top_p,
            do_sample=True,
        )

        logger.info("Model loaded successfully")

    def load_default_model(self, model_name: str = "t5-base") -> None:
        """Load default T5 model"""
        logger.info(f"Loading default model: {model_name}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.model.eval()

        self.generation_config = GenerationConfig(
            max_new_tokens=self.model_config.max_target_length,
            num_beams=self.model_config.num_beams,
            early_stopping=self.model_config.early_stopping,
        )

        logger.info("Default model loaded")

    def _preprocess_text(self, text: str, level: str) -> str:
        """Preprocess text with level prefix"""
        level_prefix = f"[{level.upper()}] "
        return f"simplify: {level_prefix}{text}"

    def simplify(
        self,
        text: str,
        level: str = "intermediate",
        return_details: bool = True,
    ) -> SimplificationResult:
        """Simplify a single text"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        input_text = self._preprocess_text(text, level)

        inputs = self.tokenizer(
            input_text,
            max_length=self.model_config.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_target_length,
                num_beams=self.model_config.num_beams,
                early_stopping=self.model_config.early_stopping,
                temperature=self.model_config.temperature,
                top_k=self.model_config.top_k,
                top_p=self.model_config.top_p,
                do_sample=True,
            )

        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        replaced_words = None
        if return_details:
            replaced_words = self._find_replaced_words(text, simplified)

        return SimplificationResult(
            original=text,
            simplified=simplified,
            level=level,
            confidence=0.85,
            replaced_words=replaced_words,
        )

    def simplify_batch(
        self,
        texts: List[str],
        level: str = "intermediate",
        batch_size: int = 8,
    ) -> List[SimplificationResult]:
        """Simplify multiple texts"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_inputs = [self._preprocess_text(text, level) for text in batch_texts]

            inputs = self.tokenizer(
                batch_inputs,
                max_length=self.model_config.max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.max_target_length,
                    num_beams=self.model_config.num_beams,
                    early_stopping=self.model_config.early_stopping,
                )

            for j, text in enumerate(batch_texts):
                simplified = self.tokenizer.decode(outputs[j], skip_special_tokens=True)
                replaced_words = self._find_replaced_words(text, simplified)

                results.append(
                    SimplificationResult(
                        original=text,
                        simplified=simplified,
                        level=level,
                        confidence=0.85,
                        replaced_words=replaced_words,
                    )
                )

        return results

    def _find_replaced_words(
        self,
        original: str,
        simplified: str,
    ) -> List[Dict[str, str]]:
        """Find words that were replaced"""
        original_words = original.lower().split()
        simplified_words = simplified.lower().split()

        replaced = []
        orig_idx = 0
        simp_idx = 0

        while orig_idx < len(original_words) and simp_idx < len(simplified_words):
            if original_words[orig_idx] == simplified_words[simp_idx]:
                orig_idx += 1
                simp_idx += 1
            else:
                if original_words[orig_idx] not in [
                    "a",
                    "an",
                    "the",
                    "is",
                    "are",
                    "was",
                    "were",
                ]:
                    replaced.append(
                        {
                            "original": original_words[orig_idx],
                            "simplified": simplified_words[simp_idx]
                            if simp_idx < len(simplified_words)
                            else "",
                        }
                    )
                orig_idx += 1
                simp_idx += 1

        return replaced

    def get_api_response(
        self,
        text: str,
        level: str = "intermediate",
    ) -> Dict:
        """Get API-style response"""
        result = self.simplify(text, level)

        return {
            "success": True,
            "original": result.original,
            "simplified": result.simplified,
            "level": result.level,
            "confidence": result.confidence,
            "replaced_words": result.replaced_words or [],
        }
