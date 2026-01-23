"""
Dataset loading utilities for Financial Sentiment Analysis.

This module provides functions to load the Financial PhraseBank dataset
from HuggingFace or local files.
"""

import logging
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from datasets import load_dataset, DatasetDict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.params import (
    DATASET_SOURCE,
    DATASET_CONFIGS,
    DEFAULT_DATASET_CONFIG,
    LABEL_NAMES,
    DATA_VERSION,
)

# Configure logging
logger = logging.getLogger(__name__)


AgreementLevel = Literal[
    "sentences_50agree",
    "sentences_66agree",
    "sentences_75agree",
    "sentences_allagree",
]


class DataLoader:
    """
    Data loader for Financial PhraseBank dataset.

    Handles loading from HuggingFace Hub or local files with proper
    error handling and logging.

    Attributes:
        agreement_level: The annotation agreement level to use.
        data_version: Version string for data tracking.
    """

    def __init__(
        self,
        agreement_level: AgreementLevel = DEFAULT_DATASET_CONFIG,
        data_version: str = DATA_VERSION,
    ) -> None:
        """
        Initialize the DataLoader.

        Args:
            agreement_level: Agreement level for annotations.
                Options: sentences_50agree, sentences_66agree,
                sentences_75agree, sentences_allagree.
            data_version: Version string for tracking data changes.
        """
        if agreement_level not in DATASET_CONFIGS:
            raise ValueError(
                f"Invalid agreement level: {agreement_level}. "
                f"Must be one of {DATASET_CONFIGS}"
            )

        self.agreement_level = agreement_level
        self.data_version = data_version
        self._dataset: Optional[pd.DataFrame] = None

        logger.info(
            f"DataLoader initialized with agreement_level={agreement_level}, "
            f"version={data_version}"
        )

    def load_from_huggingface(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load dataset from HuggingFace Hub.

        Args:
            force_reload: If True, reload even if already cached.

        Returns:
            DataFrame with columns ['sentence', 'label', 'label_name'].

        Raises:
            Exception: If loading fails.
        """
        if self._dataset is not None and not force_reload:
            logger.info("Returning cached dataset")
            return self._dataset

        logger.info(f"Loading dataset from HuggingFace: {DATASET_SOURCE}")
        logger.info(f"Configuration: {self.agreement_level}")

        try:
            dataset = load_dataset(
                DATASET_SOURCE,
                self.agreement_level,
                trust_remote_code=True,
            )

            # Convert to DataFrame
            if isinstance(dataset, DatasetDict):
                # Combine all splits if multiple exist
                dfs = []
                for split_name, split_data in dataset.items():
                    df = pd.DataFrame(split_data)
                    df['split'] = split_name
                    dfs.append(df)
                    logger.info(f"Loaded split '{split_name}': {len(df)} samples")
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.DataFrame(dataset)

            # Add label names
            df['label_name'] = df['label'].map(LABEL_NAMES)

            # Add metadata
            df['data_version'] = self.data_version
            df['agreement_level'] = self.agreement_level

            self._dataset = df

            logger.info(f"Dataset loaded successfully: {len(df)} samples")
            logger.info(f"Label distribution:\n{df['label_name'].value_counts()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def load_from_local(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load dataset from local CSV file.

        Args:
            file_path: Path to CSV file. If None, uses default raw data path.

        Returns:
            DataFrame with dataset.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if file_path is None:
            file_path = RAW_DATA_DIR / f"financial_phrasebank_{self.agreement_level}.csv"

        logger.info(f"Loading dataset from local file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Ensure label_name column exists
        if 'label_name' not in df.columns and 'label' in df.columns:
            df['label_name'] = df['label'].map(LABEL_NAMES)

        self._dataset = df
        logger.info(f"Loaded {len(df)} samples from {file_path}")

        return df

    def save_to_local(
        self,
        df: Optional[pd.DataFrame] = None,
        file_path: Optional[Path] = None,
        processed: bool = False,
    ) -> Path:
        """
        Save dataset to local CSV file.

        Args:
            df: DataFrame to save. If None, uses cached dataset.
            file_path: Output path. If None, uses default path.
            processed: If True, saves to processed directory.

        Returns:
            Path where file was saved.
        """
        if df is None:
            df = self._dataset

        if df is None:
            raise ValueError("No dataset to save. Load data first.")

        if file_path is None:
            base_dir = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR
            file_path = base_dir / f"financial_phrasebank_{self.agreement_level}.csv"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

        logger.info(f"Dataset saved to {file_path}")

        return file_path

    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        """Return cached dataset."""
        return self._dataset


def load_financial_phrasebank(
    agreement_level: AgreementLevel = DEFAULT_DATASET_CONFIG,
    source: Literal["huggingface", "local"] = "huggingface",
    local_path: Optional[Path] = None,
    save_local: bool = True,
) -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset.

    Convenience function that wraps DataLoader functionality.

    Args:
        agreement_level: Annotation agreement level.
            - sentences_50agree: At least 50% annotator agreement
            - sentences_66agree: At least 66% annotator agreement
            - sentences_75agree: At least 75% annotator agreement
            - sentences_allagree: 100% annotator agreement
        source: Where to load data from ('huggingface' or 'local').
        local_path: Path for local loading/saving.
        save_local: Whether to save a local copy after loading from HuggingFace.

    Returns:
        DataFrame with columns:
            - sentence: The financial text
            - label: Numeric label (0=negative, 1=neutral, 2=positive)
            - label_name: String label name

    Examples:
        >>> df = load_financial_phrasebank()
        >>> print(df.head())

        >>> df = load_financial_phrasebank(agreement_level="sentences_allagree")
        >>> print(f"Samples with 100% agreement: {len(df)}")
    """
    loader = DataLoader(agreement_level=agreement_level)

    if source == "huggingface":
        df = loader.load_from_huggingface()
        if save_local:
            loader.save_to_local()
    else:
        df = loader.load_from_local(local_path)

    return df
