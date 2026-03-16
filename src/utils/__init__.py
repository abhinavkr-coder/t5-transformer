import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def setup_logging(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Setup logging with console and optional file handler"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_cache_path(cache_dir: str, text: str, suffix: str = "json") -> Path:
    """Generate cache file path based on text hash"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return Path(cache_dir) / f"{text_hash}.{suffix}"


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_lines(path: str) -> List[str]:
    """Load text file line by line"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_lines(lines: List[str], path: str) -> None:
    """Save lines to text file"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """Get available device (cuda/cpu)"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Timer:
    """Context manager for timing code blocks"""

    def __init__(
        self, name: str = "Operation", logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def __enter__(self):
        self.start_time = datetime.now()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if self.logger:
                self.logger.info(f"Completed {self.name} in {format_time(duration)}")
            else:
                print(f"{self.name} completed in {format_time(duration)}")
