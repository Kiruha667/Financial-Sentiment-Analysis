"""Data processing modules."""

from src.data.loader import load_financial_phrasebank, DataLoader
from src.data.preprocessor import FinancialTextPreprocessor
from src.data.analyzer import DatasetAnalyzer
from src.data.dataset import (
    FinancialSentimentDataset,
    create_data_splits,
    create_dataloaders,
    display_batch_example,
    save_splits,
    load_splits,
    load_balanced_splits,
    get_class_weights,
)
from src.data.augmentor import FinancialDataAugmentor

__all__ = [
    # Part 1: Data loading and analysis
    "load_financial_phrasebank",
    "DataLoader",
    "FinancialTextPreprocessor",
    "DatasetAnalyzer",
    # Part 2: PyTorch dataset and utilities
    "FinancialSentimentDataset",
    "create_data_splits",
    "create_dataloaders",
    "display_batch_example",
    "save_splits",
    "load_splits",
    "load_balanced_splits",
    "get_class_weights",
    # Part 2a: Data augmentation
    "FinancialDataAugmentor",
]
