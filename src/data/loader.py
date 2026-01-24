"""
Dataset loading utilities for Financial Sentiment Analysis.

This module provides functions to load the Financial PhraseBank dataset
from HuggingFace or local files.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from huggingface_hub import hf_hub_download

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

        # Map agreement level to filename
        file_mapping = {
            "sentences_50agree": "Sentences_50Agree.txt",
            "sentences_66agree": "Sentences_66Agree.txt",
            "sentences_75agree": "Sentences_75Agree.txt",
            "sentences_allagree": "Sentences_AllAgree.txt",
        }

        # Label mapping from string to int
        label_to_int = {"negative": 0, "neutral": 1, "positive": 2}

        try:
            # Download the zip file from HuggingFace
            zip_path = hf_hub_download(
                repo_id=DATASET_SOURCE,
                filename="data/FinancialPhraseBank-v1.0.zip",
                repo_type="dataset",
            )

            # Extract and find the correct file
            filename = file_mapping[self.agreement_level]

            sentences = []
            labels = []

            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find the file in the archive
                for name in zf.namelist():
                    if name.endswith(filename):
                        with zf.open(name) as f:
                            content = f.read().decode("utf-8", errors="replace")
                            for line in content.split("\n"):
                                line = line.strip()
                                if not line:
                                    continue
                                # Split from the right to handle @ in sentences
                                parts = line.rsplit("@", 1)
                                if len(parts) == 2:
                                    sentence, label = parts
                                    sentence = sentence.strip()
                                    label = label.strip().lower()
                                    if label in label_to_int:
                                        sentences.append(sentence)
                                        labels.append(label_to_int[label])
                        break

            df = pd.DataFrame({"sentence": sentences, "label": labels})
            df["split"] = "train"

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
