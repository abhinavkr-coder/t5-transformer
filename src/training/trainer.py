"""Training module for T5 text simplification model"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    AutoTokenizer
)

from config import TrainingConfig, ModelConfig
from src.data import TextSimplificationDataset, DataProcessor
from src.utils import ensure_dir, setup_logging, count_parameters, get_device


logger = setup_logging(__name__)


class SimplificationTrainer:
    """Trainer class for text simplification model"""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        self.tokenizer: Optional[T5Tokenizer] = None
        self.model: Optional[T5ForConditionalGeneration] = None
        self.trainer: Optional[Seq2SeqTrainer] = None

        ensure_dir(self.training_config.output_dir)
        ensure_dir(self.training_config.logging_dir)

    def load_tokenizer(self) -> T5Tokenizer:
        """Load tokenizer"""
        logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_config.model_name)
        return self.tokenizer

    def load_model(self) -> T5ForConditionalGeneration:
        """Load model"""
        logger.info(f"Loading model: {self.model_config.model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_config.model_name
        )
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        logger.info(
            f"Model loaded with {count_parameters(self.model):,} trainable parameters"
        )
        return self.model

    def prepare_dataset(
        self,
        train_data: list,
        val_data: list,
    ) -> tuple:
        """Prepare datasets"""
        if not self.tokenizer:
            self.load_tokenizer()

        train_dataset = TextSimplificationDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_input_length=self.model_config.max_input_length,
            max_target_length=self.model_config.max_target_length,
            task_prefix="simplify: ",
        )

        val_dataset = TextSimplificationDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_input_length=self.model_config.max_input_length,
            max_target_length=self.model_config.max_target_length,
            task_prefix="simplify: ",
        )

        logger.info(
            f"Prepared {len(train_dataset)} train, {len(val_dataset)} val examples"
        )
        return train_dataset, val_dataset

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = predictions[0]

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels = [
            [l if l != -100 else self.tokenizer.pad_token_id for l in label]
            for label in labels
        ]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = {}
        for pred, label in zip(decoded_preds, decoded_labels):
            if pred.strip() and label.strip():
                from difflib import SequenceMatcher

                similarity = SequenceMatcher(None, pred, label).ratio()
                result.setdefault("exact_match", 0)
                result["exact_match"] += int(pred.strip() == label.strip())
                result.setdefault("similarity", 0)
                result["similarity"] += similarity

        for key in result:
            result[key] /= len(decoded_preds) if decoded_preds else 1

        return result

    def train(
        self,
        train_data: list,
        val_data: list,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the model"""
        logger.info("Starting training...")

        self.load_tokenizer()
        self.load_model()

        train_dataset, val_dataset = self.prepare_dataset(train_data, val_data)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.model_config.max_input_length,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            learning_rate=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            weight_decay=self.training_config.weight_decay,
            logging_dir=self.training_config.logging_dir,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            predict_with_generate=True,
            eval_strategy="steps",
            save_strategy="steps",
            logging_first_step=True,
            report_to="none",
        )

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3, early_stopping_threshold=0.01
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )
        self.trainer.tokenizer = self.tokenizer
        
        logger.info("Training started")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        best_model_path = Path(self.training_config.output_dir) / "best_model"
        self.model.save_pretrained(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

        return metrics

    def evaluate(self, test_data: list) -> Dict[str, Any]:
        """Evaluate the model"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")

        if not self.tokenizer:
            self.load_tokenizer()

        test_dataset = TextSimplificationDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            max_input_length=self.model_config.max_input_length,
            max_target_length=self.model_config.max_target_length,
            task_prefix="simplify: ",
        )

        metrics = self.trainer.evaluate(test_dataset)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def save_model(self, path: str) -> None:
        """Save model and tokenizer"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")

        save_path = Path(path)
        ensure_dir(str(save_path))

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint"""
        logger.info(f"Loading checkpoint from {path}")

        self.load_tokenizer()
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        logger.info(f"Loaded model with {count_parameters(self.model):,} parameters")
