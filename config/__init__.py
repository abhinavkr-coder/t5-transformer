"""Configuration for text simplification model"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class ModelConfig:
    """Configuration for T5 model"""

    model_name: str = "t5-base"
    max_input_length: int = 256
    max_target_length: int = 64
    num_beams: int = 4
    early_stopping: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for training"""

    output_dir: str = "models/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "models/logs"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    dataloader_num_workers: int = 0
    gradient_accumulation_steps: int = 4


@dataclass
class DataConfig:
    """Configuration for data processing"""

    data_dir: str = "data"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    cache_dir: str = "data/cache"
    max_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    train_split: float = 0.9
    seed: int = 42


@dataclass
class SimplificationConfig:
    """Configuration for text simplification"""

    vocabulary_levels: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "beginner": ["A1", "A2"],
            "intermediate": ["A2", "B1"],
            "advanced": ["B1", "B2", "C1"],
        }
    )

    simplification_modes: Tuple[str, ...] = ("replace", "explain", "keep")
    default_mode: str = "replace"
    min_word_frequency: int = 1000
    complexity_threshold: float = 0.6


class Config:
    """Main configuration class"""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    simplification: SimplificationConfig = field(default_factory=SimplificationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            simplification=SimplificationConfig(**config_dict.get("simplification", {})),
        )


CONFIG = Config()
