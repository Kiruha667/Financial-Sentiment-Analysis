"""
Path configurations for Financial Sentiment Analysis project.

All paths are relative to project root for portability.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Source directory
SRC_DIR = PROJECT_ROOT / "src"

# Dataset paths
FINANCIAL_PHRASEBANK_RAW = RAW_DATA_DIR / "financial_phrasebank.csv"
FINANCIAL_PHRASEBANK_PROCESSED = PROCESSED_DATA_DIR / "financial_phrasebank_processed.csv"

# Report paths
DATASET_STATS_CSV = REPORTS_DIR / "dataset_stats.csv"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.md"
ANALYSIS_REPORT = REPORTS_DIR / "analysis_report.md"

# Log paths
MAIN_LOG_FILE = LOGS_DIR / "main.log"
DATA_PROCESSING_LOG = LOGS_DIR / "data_processing.log"

# Figure paths
LABEL_DISTRIBUTION_FIG = FIGURES_DIR / "label_distribution.png"
TEXT_LENGTH_DISTRIBUTION_FIG = FIGURES_DIR / "text_length_distribution.png"
WORD_FREQUENCY_FIG = FIGURES_DIR / "word_frequency.png"
WORDCLOUD_POSITIVE_FIG = FIGURES_DIR / "wordcloud_positive.png"
WORDCLOUD_NEGATIVE_FIG = FIGURES_DIR / "wordcloud_negative.png"
WORDCLOUD_NEUTRAL_FIG = FIGURES_DIR / "wordcloud_neutral.png"
SENTIMENT_SCATTER_FIG = FIGURES_DIR / "sentiment_scatter.png"


def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
