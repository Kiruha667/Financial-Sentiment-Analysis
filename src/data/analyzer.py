"""
Dataset analysis utilities for Financial Sentiment Analysis.

This module provides the DatasetAnalyzer class for computing
statistics and quality metrics on the dataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import REPORTS_DIR, DATASET_STATS_CSV
from config.params import LABEL_NAMES

# Configure logging
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Analyzer for dataset statistics and quality metrics.

    Provides methods for computing various statistics about the
    Financial PhraseBank dataset.

    Attributes:
        df: The DataFrame being analyzed.
        text_column: Name of the text column.
        label_column: Name of the label column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = 'sentence',
        label_column: str = 'label_name',
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            df: DataFrame to analyze.
            text_column: Name of the column containing text.
            label_column: Name of the column containing labels.
        """
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column

        # Ensure text features exist
        if 'char_count' not in self.df.columns:
            self.df['char_count'] = self.df[text_column].str.len()
        if 'word_count' not in self.df.columns:
            self.df['word_count'] = self.df[text_column].str.split().str.len()

        logger.info(f"DatasetAnalyzer initialized with {len(df)} samples")

    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic dataset statistics.

        Returns:
            Dictionary containing:
                - total_samples: Total number of samples
                - num_classes: Number of unique classes
                - class_names: List of class names
                - samples_per_class: Dict of samples per class
                - class_balance: Dict of class proportions
        """
        total_samples = len(self.df)
        class_counts = self.df[self.label_column].value_counts()
        class_proportions = self.df[self.label_column].value_counts(normalize=True)

        stats = {
            'total_samples': total_samples,
            'num_classes': self.df[self.label_column].nunique(),
            'class_names': list(class_counts.index),
            'samples_per_class': class_counts.to_dict(),
            'class_balance': class_proportions.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
        }

        logger.info(f"Basic stats: {total_samples} samples, {stats['num_classes']} classes")

        return stats

    def get_text_stats(self) -> Dict[str, Any]:
        """
        Get text-related statistics.

        Returns:
            Dictionary containing text length statistics.
        """
        char_stats = self.df['char_count'].describe()
        word_stats = self.df['word_count'].describe()

        stats = {
            'character_length': {
                'min': int(char_stats['min']),
                'max': int(char_stats['max']),
                'mean': float(char_stats['mean']),
                'median': float(self.df['char_count'].median()),
                'std': float(char_stats['std']),
            },
            'word_count': {
                'min': int(word_stats['min']),
                'max': int(word_stats['max']),
                'mean': float(word_stats['mean']),
                'median': float(self.df['word_count'].median()),
                'std': float(word_stats['std']),
            },
            'text_stats_by_class': self._get_text_stats_by_class(),
        }

        logger.info(
            f"Text stats: avg {stats['word_count']['mean']:.1f} words, "
            f"avg {stats['character_length']['mean']:.1f} chars"
        )

        return stats

    def _get_text_stats_by_class(self) -> Dict[str, Dict[str, float]]:
        """Get text statistics grouped by class."""
        stats_by_class = {}

        for label in self.df[self.label_column].unique():
            label_df = self.df[self.df[self.label_column] == label]
            stats_by_class[label] = {
                'count': len(label_df),
                'avg_word_count': float(label_df['word_count'].mean()),
                'avg_char_count': float(label_df['char_count'].mean()),
                'std_word_count': float(label_df['word_count'].std()),
            }

        return stats_by_class

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Check data quality issues.

        Returns:
            Dictionary containing data quality metrics.
        """
        # Missing values
        missing_values = self.df.isnull().sum().to_dict()
        total_missing = sum(missing_values.values())

        # Duplicates
        duplicate_texts = self.df[self.text_column].duplicated().sum()
        duplicate_rows = self.df.duplicated().sum()

        # Empty or very short texts
        empty_texts = (self.df[self.text_column].str.len() == 0).sum()
        short_texts = (self.df['word_count'] < 3).sum()

        # Very long texts (potential outliers)
        q99_length = self.df['word_count'].quantile(0.99)
        long_texts = (self.df['word_count'] > q99_length).sum()

        quality_report = {
            'missing_values': missing_values,
            'total_missing': total_missing,
            'duplicate_texts': int(duplicate_texts),
            'duplicate_rows': int(duplicate_rows),
            'empty_texts': int(empty_texts),
            'short_texts_under_3_words': int(short_texts),
            'long_texts_above_99th_percentile': int(long_texts),
            'data_quality_score': self._calculate_quality_score(
                total_missing, duplicate_texts, empty_texts, len(self.df)
            ),
        }

        logger.info(
            f"Data quality check: {duplicate_texts} duplicates, "
            f"{total_missing} missing values"
        )

        return quality_report

    def _calculate_quality_score(
        self,
        missing: int,
        duplicates: int,
        empty: int,
        total: int,
    ) -> float:
        """Calculate a simple data quality score (0-100)."""
        if total == 0:
            return 0.0

        issues = missing + duplicates + empty
        score = max(0, 100 * (1 - issues / total))

        return round(score, 2)

    def get_label_distribution(self) -> pd.DataFrame:
        """
        Get detailed label distribution.

        Returns:
            DataFrame with label distribution statistics.
        """
        distribution = self.df[self.label_column].value_counts()
        percentages = self.df[self.label_column].value_counts(normalize=True) * 100

        result = pd.DataFrame({
            'label': distribution.index,
            'count': distribution.values,
            'percentage': percentages.values.round(2),
        })

        # Add cumulative percentage
        result['cumulative_percentage'] = result['percentage'].cumsum().round(2)

        return result

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame combining all statistics.

        Returns:
            DataFrame with comprehensive dataset statistics.
        """
        basic_stats = self.get_basic_stats()
        text_stats = self.get_text_stats()
        quality_stats = self.check_data_quality()

        summary_data = []

        # Basic statistics
        summary_data.append({'category': 'Basic', 'metric': 'Total Samples', 'value': basic_stats['total_samples']})
        summary_data.append({'category': 'Basic', 'metric': 'Number of Classes', 'value': basic_stats['num_classes']})
        summary_data.append({'category': 'Basic', 'metric': 'Imbalance Ratio', 'value': f"{basic_stats['imbalance_ratio']:.2f}"})

        # Per-class counts
        for class_name, count in basic_stats['samples_per_class'].items():
            summary_data.append({
                'category': 'Class Distribution',
                'metric': f'{class_name} Count',
                'value': count,
            })

        # Text statistics
        summary_data.append({'category': 'Text', 'metric': 'Avg Word Count', 'value': f"{text_stats['word_count']['mean']:.1f}"})
        summary_data.append({'category': 'Text', 'metric': 'Min Word Count', 'value': text_stats['word_count']['min']})
        summary_data.append({'category': 'Text', 'metric': 'Max Word Count', 'value': text_stats['word_count']['max']})
        summary_data.append({'category': 'Text', 'metric': 'Avg Char Length', 'value': f"{text_stats['character_length']['mean']:.1f}"})

        # Quality metrics
        summary_data.append({'category': 'Quality', 'metric': 'Duplicate Texts', 'value': quality_stats['duplicate_texts']})
        summary_data.append({'category': 'Quality', 'metric': 'Missing Values', 'value': quality_stats['total_missing']})
        summary_data.append({'category': 'Quality', 'metric': 'Quality Score', 'value': f"{quality_stats['data_quality_score']:.1f}%"})

        return pd.DataFrame(summary_data)

    def save_stats_report(
        self,
        output_path: Optional[Path] = None,
        format: str = 'csv',
    ) -> Path:
        """
        Save statistics report to file.

        Args:
            output_path: Output file path. If None, uses default.
            format: Output format ('csv' or 'markdown').

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = DATASET_STATS_CSV

        summary_df = self.get_summary_dataframe()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            summary_df.to_csv(output_path, index=False)
        elif format == 'markdown':
            output_path = output_path.with_suffix('.md')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Dataset Statistics Report\n\n")
                f.write(summary_df.to_markdown(index=False))

        logger.info(f"Statistics report saved to {output_path}")

        return output_path

    def get_all_statistics(self) -> Dict[str, Any]:
        """
        Get all statistics in a single dictionary.

        Returns:
            Dictionary containing all computed statistics.
        """
        return {
            'basic_stats': self.get_basic_stats(),
            'text_stats': self.get_text_stats(),
            'data_quality': self.check_data_quality(),
            'label_distribution': self.get_label_distribution().to_dict('records'),
        }


def export_to_report(
    analyzer: DatasetAnalyzer,
    output_path: Optional[Path] = None,
    include_quality: bool = True,
) -> Path:
    """
    Export analysis results to a markdown report.

    Args:
        analyzer: DatasetAnalyzer instance with computed statistics.
        output_path: Output file path.
        include_quality: Whether to include data quality section.

    Returns:
        Path to generated report.
    """
    from datetime import datetime
    from config.params import REPORT_AUTHOR, REPORT_DATE_FORMAT

    if output_path is None:
        output_path = REPORTS_DIR / "analysis_report.md"

    stats = analyzer.get_all_statistics()

    report_lines = [
        "# Financial Sentiment Analysis - Dataset Report",
        "",
        f"**Generated:** {datetime.now().strftime(REPORT_DATE_FORMAT)}",
        f"**Author:** {REPORT_AUTHOR}",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"- **Total Samples:** {stats['basic_stats']['total_samples']:,}",
        f"- **Number of Classes:** {stats['basic_stats']['num_classes']}",
        f"- **Class Names:** {', '.join(stats['basic_stats']['class_names'])}",
        "",
        "### Class Distribution",
        "",
        "| Label | Count | Percentage |",
        "|-------|-------|------------|",
    ]

    for item in stats['label_distribution']:
        report_lines.append(
            f"| {item['label']} | {item['count']:,} | {item['percentage']:.1f}% |"
        )

    report_lines.extend([
        "",
        f"**Imbalance Ratio:** {stats['basic_stats']['imbalance_ratio']:.2f}",
        "",
        "## 2. Text Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg Word Count | {stats['text_stats']['word_count']['mean']:.1f} |",
        f"| Min Word Count | {stats['text_stats']['word_count']['min']} |",
        f"| Max Word Count | {stats['text_stats']['word_count']['max']} |",
        f"| Std Word Count | {stats['text_stats']['word_count']['std']:.1f} |",
        f"| Avg Char Length | {stats['text_stats']['character_length']['mean']:.1f} |",
        "",
    ])

    if include_quality:
        quality = stats['data_quality']
        report_lines.extend([
            "## 3. Data Quality",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Duplicate Texts | {quality['duplicate_texts']} |",
            f"| Missing Values | {quality['total_missing']} |",
            f"| Empty Texts | {quality['empty_texts']} |",
            f"| Short Texts (<3 words) | {quality['short_texts_under_3_words']} |",
            f"| Quality Score | {quality['data_quality_score']:.1f}% |",
            "",
        ])

    report_lines.extend([
        "---",
        "",
        "*Report generated automatically by Financial Sentiment Analysis pipeline.*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report exported to {output_path}")

    return output_path
