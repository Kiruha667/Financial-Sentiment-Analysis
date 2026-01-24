"""
Helper utilities for Financial Sentiment Analysis.

This module provides utility functions for logging, timing,
random seed management, and other common operations.
"""

import logging
import random
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import LOGS_DIR, MAIN_LOG_FILE
from config.params import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL, RANDOM_SEED


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = LOG_LEVEL,
    format_string: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file. If None, uses default.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Log message format.
        date_format: Date format for log messages.

    Returns:
        Configured root logger.
    """
    if log_file is None:
        log_file = MAIN_LOG_FILE

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger()
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed}")


@contextmanager
def timer(description: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager for timing operations.

    Args:
        description: Description of the operation being timed.

    Yields:
        None

    Example:
        >>> with timer("Data loading"):
        ...     df = load_data()
        Data loading completed in 2.34 seconds
    """
    logger = logging.getLogger(__name__)
    start_time = time.perf_counter()

    logger.info(f"{description} started...")

    try:
        yield
    finally:
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"{description} completed in {elapsed_time:.2f} seconds")


def timed(func: Callable) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        func: Function to time.

    Returns:
        Wrapped function.

    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


def format_number(num: float, precision: int = 2) -> str:
    """
    Format number with thousands separator.

    Args:
        num: Number to format.
        precision: Decimal precision.

    Returns:
        Formatted string.

    Examples:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(1234567)
        '1,234,567.00'
    """
    return f"{num:,.{precision}f}"


def ensure_list(value: Any) -> list:
    """
    Ensure value is a list.

    Args:
        value: Value to convert to list if needed.

    Returns:
        List containing value(s).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value if division by zero.

    Returns:
        Result of division or default.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def create_directory_structure(base_path: Path) -> None:
    """
    Create the full project directory structure.

    Args:
        base_path: Base project path.
    """
    directories = [
        "config",
        "data/raw",
        "data/processed",
        "data/splits",
        "notebooks",
        "src/data",
        "src/models",
        "src/visualization",
        "src/utils",
        "outputs/figures",
        "outputs/reports",
        "outputs/logs",
        "outputs/checkpoints",
    ]

    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Directory structure created at {base_path}")


# ==============================================================================
# Part 2: Training Utilities
# ==============================================================================

def get_device() -> str:
    """
    Auto-detect and return available device (cuda/cpu).

    Returns:
        Device string ('cuda' or 'cpu')

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger = logging.getLogger(__name__)
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            logger = logging.getLogger(__name__)
            logger.warning("CUDA not available, using CPU")
    except ImportError:
        device = 'cpu'
        logger = logging.getLogger(__name__)
        logger.warning("PyTorch not installed, defaulting to CPU")

    return device


def count_parameters(model) -> int:
    """
    Count total trainable parameters in model.

    Args:
        model: PyTorch model (nn.Module)

    Returns:
        Number of trainable parameters

    Example:
        >>> params = count_parameters(model)
        >>> print(f"Trainable parameters: {params:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")

    Examples:
        >>> format_time(3665)
        '1h 1m 5s'
        >>> format_time(125)
        '2m 5s'
        >>> format_time(45)
        '45s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_gpu_memory_usage() -> Optional[dict]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory stats or None if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
            }
    except ImportError:
        pass
    return None


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger = logging.getLogger(__name__)
            logger.info("GPU memory cache cleared")
    except ImportError:
        pass
