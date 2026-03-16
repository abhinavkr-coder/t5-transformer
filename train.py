#!/usr/bin/env python
"""Main training script for text simplification model"""

import argparse
import logging
import sys

from config import ModelConfig, TrainingConfig, DataConfig, CONFIG
from src.data import DataProcessor
from src.training import SimplificationTrainer
from src.utils import setup_logging, Timer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train T5 text simplification model")

    parser.add_argument(
        "--model_name", type=str, default="t5-base", help="Model name or path"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use (for debugging)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--use_kaggle",
        action="store_true",
        help="Download and use Kaggle Simple Wikipedia dataset",
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    logger = setup_logging("train", level=getattr(logging, args.log_level))

    logger.info("=" * 60)
    logger.info("T5 Text Simplification Model Training")
    logger.info("=" * 60)

    with Timer("Data preparation", logger):
        processor = DataProcessor(
            data_dir=args.data_dir, cache_dir=f"{args.data_dir}/cache"
        )

        if args.use_kaggle:
            logger.info("Downloading Kaggle Simple English Wikipedia dataset...")
            raw_data = processor.download_kaggle_simple_wiki()
        else:
            logger.info("Creating sample training data...")
            raw_data = processor.create_sample_data(
                num_samples=args.max_samples or 1000
            )

        # Limit samples if specified
        if args.max_samples and args.max_samples < len(raw_data):
            raw_data = raw_data[: args.max_samples]
            logger.info(f"Limited to {args.max_samples} samples")

        logger.info("Augmenting data...")
        augmented_data = processor.augment_data(raw_data, augmentation_factor=3)

        logger.info("Splitting data...")
        train_data, val_data = processor.split_data(augmented_data, train_ratio=0.9)

        processor.save_splits(train_data, val_data)

    with Timer("Model training", logger):
        model_config = ModelConfig(
            model_name=args.model_name,
            max_input_length=256,
            max_target_length=64,
            num_beams=4,
            early_stopping=True,
        )

        training_config = TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            fp16=True,
        )

        trainer = SimplificationTrainer(
            model_config=model_config,
            training_config=training_config,
        )

        metrics = trainer.train(
            train_data=train_data,
            val_data=val_data,
            resume_from_checkpoint=args.resume,
        )

        logger.info("Training completed!")
        logger.info(f"Final metrics: {metrics}")

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
