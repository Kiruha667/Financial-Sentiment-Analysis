"""
PyTorch Dataset for Financial Sentiment Analysis.

This module provides:
- FinancialSentimentDataset: PyTorch Dataset for tokenized financial texts
- create_data_splits: Stratified train/val/test splitting
- create_dataloaders: DataLoader creation utility
- save_splits/load_splits: Persistence utilities
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class FinancialSentimentDataset(Dataset):
    """
    PyTorch Dataset for financial sentiment classification.

    Tokenizes text on-the-fly and returns tensors suitable for transformer models.

    Args:
        texts: List of text strings
        labels: List of integer labels (0=negative, 1=neutral, 2=positive)
        tokenizer_name: HuggingFace tokenizer name/path
        max_length: Maximum sequence length for tokenization

    Example:
        >>> dataset = FinancialSentimentDataset(
        ...     texts=["Stock prices rose sharply"],
        ...     labels=[2],
        ...     tokenizer_name="ProsusAI/finbert",
        ...     max_length=128
        ... )
        >>> sample = dataset[0]
        >>> sample['input_ids'].shape
        torch.Size([128])
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        logger.info(f"Created dataset with {len(self.texts)} samples")
        logger.info(f"Tokenizer: {tokenizer_name}")
        logger.info(f"Max length: {max_length}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample from dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - input_ids: Token IDs tensor [max_length]
                - attention_mask: Attention mask tensor [max_length]
                - labels: Label tensor (scalar)
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_splits(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits from dataframe.

    Maintains class distribution across all splits using stratified sampling.

    Args:
        df: Input dataframe with text and labels
        text_column: Name of text column
        label_column: Name of label column
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        AssertionError: If split ratios don't sum to 1.0

    Example:
        >>> train, val, test = create_data_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        >>> len(train) + len(val) + len(test) == len(df)
        True
    """
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {ratio_sum}"

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df[label_column]
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=train_val[label_column]
    )

    # Reset indices
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Log split information
    logger.info("Created data splits:")
    logger.info(f"  Train: {len(train)} samples ({train_ratio:.1%})")
    logger.info(f"  Val:   {len(val)} samples ({val_ratio:.1%})")
    logger.info(f"  Test:  {len(test)} samples ({test_ratio:.1%})")

    # Log label distribution for each split
    for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        dist = split_df[label_column].value_counts(normalize=True).sort_index()
        logger.info(f"  {split_name} label distribution: {dist.to_dict()}")

    return train, val, test


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer_name: str,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 2,
    text_column: str = 'text',
    label_column: str = 'label'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_df, val_df, test_df: DataFrames with text and labels
        tokenizer_name: HuggingFace tokenizer name/checkpoint
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Note:
        - Train loader shuffles data; val/test loaders do not
        - pin_memory=True for faster GPU transfer
        - On Windows, num_workers=0 may be needed to avoid multiprocessing issues
    """
    # Create datasets
    train_dataset = FinancialSentimentDataset(
        texts=train_df[text_column].tolist(),
        labels=train_df[label_column].tolist(),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    val_dataset = FinancialSentimentDataset(
        texts=val_df[text_column].tolist(),
        labels=val_df[label_column].tolist(),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    test_dataset = FinancialSentimentDataset(
        texts=test_df[text_column].tolist(),
        labels=test_df[label_column].tolist(),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    # Check if CUDA is available for pin_memory
    pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created DataLoaders with batch_size={batch_size}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")
    logger.info(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def display_batch_example(batch: Dict[str, torch.Tensor], tokenizer_name: str) -> None:
    """
    Display example from a batch for debugging.

    Args:
        batch: Dictionary with input_ids, attention_mask, labels
        tokenizer_name: HuggingFace tokenizer name for decoding
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Get first example from batch
    input_ids = batch['input_ids'][0]
    attention_mask = batch['attention_mask'][0]
    label = batch['labels'][0]

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Count actual tokens (non-padding)
    actual_tokens = attention_mask.sum().item()

    print("=" * 80)
    print("BATCH EXAMPLE")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Label: {label.item()}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Actual tokens (non-padding): {actual_tokens}")
    print(f"\nFirst 10 tokens: {tokens[:10]}")
    print(f"Input IDs (first 10): {input_ids[:10].tolist()}")
    print(f"Attention mask (first 10): {attention_mask[:10].tolist()}")
    print("=" * 80)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'data/splits'
) -> None:
    """
    Save train/val/test splits to disk as CSV files.

    Args:
        train_df, val_df, test_df: DataFrames to save
        output_dir: Directory to save splits
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

    logger.info(f"Saved splits to {output_dir}/")
    logger.info(f"  train.csv: {len(train_df)} samples")
    logger.info(f"  val.csv: {len(val_df)} samples")
    logger.info(f"  test.csv: {len(test_df)} samples")


def load_splits(splits_dir: str = 'data/splits') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits from disk.

    Args:
        splits_dir: Directory containing saved splits

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If split files don't exist
    """
    splits_path = Path(splits_dir)

    train_df = pd.read_csv(splits_path / 'train.csv')
    val_df = pd.read_csv(splits_path / 'val.csv')
    test_df = pd.read_csv(splits_path / 'test.csv')

    logger.info(f"Loaded splits from {splits_dir}/")
    logger.info(f"  train.csv: {len(train_df)} samples")
    logger.info(f"  val.csv: {len(val_df)} samples")
    logger.info(f"  test.csv: {len(test_df)} samples")

    return train_df, val_df, test_df


def get_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.

    Uses inverse frequency weighting to give more weight to minority classes.

    Args:
        labels: List of integer labels
        num_classes: Number of classes

    Returns:
        Tensor of class weights [num_classes]

    Example:
        >>> weights = get_class_weights([0, 1, 1, 1, 2], num_classes=3)
        >>> weights  # Higher weight for class 0 and 2 (minority)
    """
    import numpy as np

    label_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    # Inverse frequency weighting
    weights = total / (num_classes * label_counts)

    # Normalize so that weights sum to num_classes
    weights = weights / weights.sum() * num_classes

    logger.info(f"Class weights: {dict(enumerate(weights))}")

    return torch.tensor(weights, dtype=torch.float32)
