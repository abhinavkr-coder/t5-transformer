#!/usr/bin/env python
"""Inference script for text simplification"""

import argparse
import logging
import sys
import json

from config import ModelConfig
from src.inference import SimplificationInference
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run text simplification inference")

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (optional, loads t5-base if not provided)",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to simplify")
    parser.add_argument(
        "--level",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
        help="Simplification level",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to JSON file with texts to simplify",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results"
    )

    return parser.parse_args()


def run_interactive(inference: SimplificationInference):
    """Run interactive mode"""
    print("=" * 60)
    print("Text Simplification Interactive Mode")
    print("Enter text to simplify (or 'quit' to exit)")
    print("=" * 60)

    while True:
        print("\n> ", end="")
        text = input().strip()

        if text.lower() in ["quit", "exit", "q"]:
            break

        if not text:
            continue

        print("\nLevel: ", end="")
        level = input().strip() or "intermediate"

        result = inference.simplify(text, level)

        print("\n" + "-" * 40)
        print(f"ORIGINAL:   {result.original}")
        print(f"SIMPLIFIED: {result.simplified}")

        if result.replaced_words:
            print("\nReplaced words:")
            for word in result.replaced_words:
                print(f"  {word['original']} -> {word['simplified']}")

        print("-" * 40)


def run_batch(
    inference: SimplificationInference, batch_file: str, output_file: str = None
):
    """Run batch mode"""
    with open(batch_file, "r") as f:
        data = json.load(f)

    texts = data.get("texts", [])
    level = data.get("level", "intermediate")

    results = []
    for text in texts:
        result = inference.simplify(text, level)
        results.append(
            {
                "original": result.original,
                "simplified": result.simplified,
                "level": result.level,
                "replaced_words": result.replaced_words or [],
            }
        )

    output = {"results": results}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print(json.dumps(output, indent=2))


def main():
    """Main inference function"""
    args = parse_args()

    logger = setup_logging("inference", level=logging.INFO)

    logger.info("Loading model...")

    model_config = ModelConfig()
    inference = SimplificationInference(model_config=model_config)

    if args.model_path:
        inference.load_model(args.model_path)
    else:
        logger.info("No model path provided, using default t5-base")
        inference.load_default_model("t5-base")

    if args.interactive:
        run_interactive(inference)
    elif args.batch:
        run_batch(inference, args.batch, args.output)
    elif args.text:
        result = inference.simplify(args.text, args.level)

        print("\n" + "=" * 60)
        print(f"ORIGINAL:   {result.original}")
        print(f"SIMPLIFIED: {result.simplified}")

        if result.replaced_words:
            print("\nReplaced words:")
            for word in result.replaced_words:
                print(f"  {word['original']} -> {word['simplified']}")

        print("=" * 60)
    else:
        print("Please provide --text, --batch, or --interactive")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
