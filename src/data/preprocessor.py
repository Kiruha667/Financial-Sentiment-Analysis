"""
Text preprocessing utilities for Financial Sentiment Analysis.

This module provides the FinancialTextPreprocessor class for cleaning
and preparing financial text data for analysis and modeling.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import PROCESSED_DATA_DIR
from config.params import (
    STOPWORDS_LANGUAGE,
    ADDITIONAL_STOPWORDS,
    MIN_WORD_LENGTH,
    DATA_VERSION,
)

# Configure logging
logger = logging.getLogger(__name__)


class FinancialTextPreprocessor:
    """
    Preprocessor for financial text data.

    Handles text cleaning, tokenization, and data quality operations
    specific to financial sentiment analysis.

    Attributes:
        stopwords: Set of stopwords to remove.
        min_word_length: Minimum word length to keep.
    """

    def __init__(
        self,
        custom_stopwords: Optional[List[str]] = None,
        min_word_length: int = MIN_WORD_LENGTH,
        use_default_stopwords: bool = True,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            custom_stopwords: Additional stopwords to use.
            min_word_length: Minimum word length to keep in tokenization.
            use_default_stopwords: Whether to use NLTK's default stopwords.
        """
        self.min_word_length = min_word_length
        self.stopwords: Set[str] = set()

        if use_default_stopwords:
            try:
                import nltk
                try:
                    from nltk.corpus import stopwords
                    self.stopwords = set(stopwords.words(STOPWORDS_LANGUAGE))
                except LookupError:
                    logger.info("Downloading NLTK stopwords...")
                    nltk.download('stopwords', quiet=True)
                    from nltk.corpus import stopwords
                    self.stopwords = set(stopwords.words(STOPWORDS_LANGUAGE))
            except ImportError:
                logger.warning("NLTK not installed. Using basic stopwords.")
                self.stopwords = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                    'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
                    'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                    'does', 'did', 'will', 'would', 'could', 'should', 'may',
                    'might', 'must', 'shall', 'can', 'this', 'that', 'these',
                    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                }

        # Add additional stopwords
        self.stopwords.update(ADDITIONAL_STOPWORDS)

        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        logger.info(f"Preprocessor initialized with {len(self.stopwords)} stopwords")

    def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_numbers: bool = False,
        remove_special_chars: bool = True,
        remove_extra_whitespace: bool = True,
    ) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text to clean.
            lowercase: Whether to convert to lowercase.
            remove_numbers: Whether to remove numeric characters.
            remove_special_chars: Whether to remove special characters.
            remove_extra_whitespace: Whether to normalize whitespace.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        if lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers (optional, often important in financial text)
        if remove_numbers:
            text = re.sub(r'\d+\.?\d*', '', text)

        # Remove special characters but keep apostrophes and hyphens
        if remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s'-]", '', text)

        # Normalize whitespace
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(
        self,
        text: str,
        remove_stopwords: bool = True,
        apply_min_length: bool = True,
    ) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text to tokenize.
            remove_stopwords: Whether to remove stopwords.
            apply_min_length: Whether to filter by minimum word length.

        Returns:
            List of tokens.
        """
        # Simple whitespace tokenization
        tokens = text.split()

        if remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]

        if apply_min_length:
            tokens = [t for t in tokens if len(t) >= self.min_word_length]

        return tokens

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first',
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.

        Args:
            df: Input DataFrame.
            subset: Columns to consider for duplicates. If None, uses 'sentence'.
            keep: Which duplicates to keep ('first', 'last', False).

        Returns:
            DataFrame with duplicates removed.
        """
        if subset is None:
            subset = ['sentence']

        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_clean)

        logger.info(f"Removed {removed_count} duplicate rows ({removed_count/initial_count*100:.2f}%)")

        return df_clean

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check and report missing values in DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing value statistics.
        """
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percent': (df.isnull().sum() / len(df) * 100).values,
            'dtype': df.dtypes.values,
        })

        logger.info(f"Missing values check:\n{missing_stats.to_string()}")

        return missing_stats

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'sentence',
        clean_text_params: Optional[dict] = None,
        remove_duplicates: bool = True,
        add_text_features: bool = True,
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to a DataFrame.

        Args:
            df: Input DataFrame.
            text_column: Name of the text column.
            clean_text_params: Parameters for clean_text method.
            remove_duplicates: Whether to remove duplicate texts.
            add_text_features: Whether to add text length/word count features.

        Returns:
            Preprocessed DataFrame.
        """
        logger.info(f"Starting preprocessing pipeline on {len(df)} rows")
        df = df.copy()

        # Check missing values
        missing_stats = self.check_missing_values(df)
        missing_text = df[text_column].isnull().sum()

        if missing_text > 0:
            logger.warning(f"Found {missing_text} missing values in '{text_column}'. Removing.")
            df = df.dropna(subset=[text_column])

        # Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df, subset=[text_column])

        # Clean text
        clean_params = clean_text_params or {}
        logger.info("Cleaning text...")

        df['sentence_clean'] = df[text_column].apply(
            lambda x: self.clean_text(x, **clean_params)
        )

        # Add text features
        if add_text_features:
            logger.info("Adding text features...")
            df['char_count'] = df[text_column].str.len()
            df['word_count'] = df[text_column].str.split().str.len()
            df['char_count_clean'] = df['sentence_clean'].str.len()
            df['word_count_clean'] = df['sentence_clean'].str.split().str.len()

        logger.info(f"Preprocessing complete. Final dataset: {len(df)} rows")

        return df

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str = "financial_phrasebank_processed.csv",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save processed DataFrame to file.

        Args:
            df: DataFrame to save.
            filename: Output filename.
            output_dir: Output directory. If None, uses default processed dir.

        Returns:
            Path to saved file.
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Add metadata
        df_save = df.copy()
        df_save['preprocessing_version'] = DATA_VERSION

        df_save.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        return output_path

    def get_word_frequencies(
        self,
        df: pd.DataFrame,
        text_column: str = 'sentence_clean',
        top_n: int = 20,
        by_label: bool = False,
        label_column: str = 'label_name',
    ) -> pd.DataFrame:
        """
        Calculate word frequencies.

        Args:
            df: Input DataFrame.
            text_column: Column containing text.
            top_n: Number of top words to return.
            by_label: Whether to calculate frequencies per label.
            label_column: Column containing labels.

        Returns:
            DataFrame with word frequencies.
        """
        from collections import Counter

        if by_label and label_column in df.columns:
            results = []
            for label in df[label_column].unique():
                label_df = df[df[label_column] == label]
                all_words = []
                for text in label_df[text_column].dropna():
                    tokens = self.tokenize(str(text))
                    all_words.extend(tokens)

                word_counts = Counter(all_words).most_common(top_n)
                for word, count in word_counts:
                    results.append({
                        'label': label,
                        'word': word,
                        'count': count,
                    })

            return pd.DataFrame(results)
        else:
            all_words = []
            for text in df[text_column].dropna():
                tokens = self.tokenize(str(text))
                all_words.extend(tokens)

            word_counts = Counter(all_words).most_common(top_n)
            return pd.DataFrame(word_counts, columns=['word', 'count'])
