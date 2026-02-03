"""
Hyperparameters and constants for Financial Sentiment Analysis project.
"""

from typing import Dict, List, Tuple

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Dataset parameters
DATASET_NAME: str = "financial_phrasebank"
DATASET_SOURCE: str = "takala/financial_phrasebank"  # HuggingFace dataset name
DATASET_CONFIGS: List[str] = [
    "sentences_50agree",
    "sentences_66agree",
    "sentences_75agree",
    "sentences_allagree",
]
DEFAULT_DATASET_CONFIG: str = "sentences_75agree"

# Label mapping
LABEL_NAMES: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
LABEL_TO_ID: Dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}

# Analysis parameters
TOP_N_WORDS: int = 20
MIN_WORD_LENGTH: int = 3  # Minimum word length for frequency analysis

# Visualization parameters
FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"

# Figure sizes (width, height) in inches
FIGSIZE_SMALL: Tuple[int, int] = (8, 6)
FIGSIZE_MEDIUM: Tuple[int, int] = (10, 8)
FIGSIZE_LARGE: Tuple[int, int] = (12, 10)
FIGSIZE_WIDE: Tuple[int, int] = (14, 6)

# Color palettes
SENTIMENT_COLORS: Dict[str, str] = {
    "positive": "#2ecc71",  # Green
    "neutral": "#3498db",   # Blue
    "negative": "#e74c3c",  # Red
}

# Color palette for plots (seaborn compatible)
PLOT_PALETTE: List[str] = ["#e74c3c", "#3498db", "#2ecc71"]  # neg, neu, pos

# Text preprocessing parameters
STOPWORDS_LANGUAGE: str = "english"
ADDITIONAL_STOPWORDS: List[str] = [
    "said", "would", "could", "also", "one", "two",
    "company", "companies", "year", "years", "million",
    "billion", "quarter", "percent", "eur", "usd",
]

# WordCloud parameters
WORDCLOUD_MAX_WORDS: int = 100
WORDCLOUD_WIDTH: int = 800
WORDCLOUD_HEIGHT: int = 400
WORDCLOUD_BACKGROUND: str = "white"

# Logging configuration
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"

# Data versioning
DATA_VERSION: str = "1.0.0"

# Report generation
REPORT_AUTHOR: str = "Financial Sentiment Analysis Project"
REPORT_DATE_FORMAT: str = "%Y-%m-%d"

# ==============================================================================
# Part 2: Training Configuration
# ==============================================================================

# Data split ratios
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Text processing for models
MAX_SEQ_LENGTH: int = 128

# DataLoader configuration
BATCH_SIZE: int = 16
NUM_WORKERS: int = 2  # Worker processes for data loading

# Label mapping (list format for HuggingFace compatibility)
LABEL_LIST: List[str] = ['negative', 'neutral', 'positive']
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ==============================================================================
# Part 2a: Data Augmentation Configuration
# ==============================================================================

# Augmentation techniques to apply
AUGMENTATION_TECHNIQUES: List[str] = [
    "synonym",
    "insert",
    "delete",
    "back_translate",
]

# Back-translation model pairs (English -> X -> English)
BACK_TRANSLATION_SRC_MODEL: str = "Helsinki-NLP/opus-mt-en-de"
BACK_TRANSLATION_TGT_MODEL: str = "Helsinki-NLP/opus-mt-de-en"

# Maximum fraction of deficit to fill with template-generated sentences
TEMPLATE_MAX_RATIO: float = 0.15

# nlpaug parameters
AUGMENT_SYNONYM_TOP_K: int = 5
AUGMENT_INSERT_TOP_K: int = 5
AUGMENT_DELETE_MIN_TOKENS: int = 4

# External dataset for additional samples
TWITTER_FINANCIAL_DATASET: str = "zeroshot/twitter-financial-news-sentiment"
