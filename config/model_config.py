"""
Model configuration for Financial Sentiment Analysis.

This module provides dataclass-based configuration for transformer models,
including predefined configurations for FinBERT and RoBERTa.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """
    Configuration for transformer model training.

    Attributes:
        model_name: Human-readable model name
        model_checkpoint: HuggingFace model checkpoint path
        num_labels: Number of classification labels
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for scheduler
        weight_decay: Weight decay for regularization
        max_seq_length: Maximum sequence length for tokenization
        optimizer: Optimizer type (adamw, adam, sgd)
        scheduler: Learning rate scheduler type
        patience: Early stopping patience
        min_delta: Minimum improvement for early stopping
        dropout: Dropout rate
        gradient_clip_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
        device: Device to use (cuda/cpu), auto-detected if None
    """

    # Model selection
    model_name: str
    model_checkpoint: str
    num_labels: int = 3

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_seq_length: int = 128

    # Optimizer and scheduler
    optimizer: str = "adamw"
    scheduler: str = "linear"

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Regularization
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0

    # Reproducibility
    seed: int = 42

    # Hardware
    device: Optional[str] = None

    def __post_init__(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined model configurations
FINBERT_CONFIG = ModelConfig(
    model_name="finbert",
    model_checkpoint="ProsusAI/finbert",
    learning_rate=1e-5,  # Lower for domain-specific model
    batch_size=16,
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    max_seq_length=128,
    patience=3
)

ROBERTA_CONFIG = ModelConfig(
    model_name="roberta-base",
    model_checkpoint="roberta-base",
    learning_rate=2e-5,  # Higher for general model
    batch_size=16,
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    max_seq_length=128,
    patience=3
)


def get_config(model_name: str) -> ModelConfig:
    """
    Get predefined model configuration by name.

    Args:
        model_name: Model name ('finbert' or 'roberta-base')

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If model name is not recognized
    """
    configs = {
        "finbert": FINBERT_CONFIG,
        "roberta-base": ROBERTA_CONFIG,
    }

    if model_name.lower() not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return configs[model_name.lower()]


def print_config(config: ModelConfig) -> None:
    """
    Print configuration in readable format.

    Args:
        config: ModelConfig instance to print
    """
    print("=" * 60)
    print(f"Model Configuration: {config.model_name}")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"{key:.<30} {value}")
    print("=" * 60)
