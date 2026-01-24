# Financial Sentiment Analysis

NLP project for sentiment analysis of financial texts using the Financial PhraseBank dataset.

## Project Overview

This project implements a complete pipeline for financial sentiment analysis, from data exploration and preprocessing to model training and evaluation.

**Task:** Classify financial news sentences into three sentiment classes:
- **Positive** - Bullish/optimistic sentiment
- **Neutral** - Factual/objective statements
- **Negative** - Bearish/pessimistic sentiment

## Dataset

**Financial PhraseBank** (Malo et al., 2014)

- 4,846 sentences from financial news (50% agreement level)
- Annotated by 5-8 annotators with varying agreement levels
- Three sentiment classes: positive, neutral, negative
- Four agreement configurations: 50%, 66%, 75%, 100%
- **Used in this project:** 75% agreement (3,453 samples)

Source: [HuggingFace Datasets](https://huggingface.co/datasets/takala/financial_phrasebank)

## Project Structure

```
project/
├── config/
│   ├── __init__.py
│   ├── paths.py          # Path configurations
│   └── params.py         # Hyperparameters and constants
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Preprocessed data
│   └── README.md         # Dataset documentation
├── notebooks/
│   └── 01_data_analysis.ipynb  # EDA and analysis (Part 1)
├── src/
│   ├── data/
│   │   ├── loader.py     # Dataset loading utilities
│   │   ├── preprocessor.py  # Text preprocessing
│   │   └── analyzer.py   # Statistical analysis
│   ├── visualization/
│   │   └── plots.py      # Visualization functions
│   └── utils/
│       └── helpers.py    # Utility functions
├── outputs/
│   ├── figures/          # Generated plots
│   ├── reports/          # Analysis reports
│   └── logs/             # Application logs
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-sentiment-analysis
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Part 1: Data Analysis

1. Open the analysis notebook:
```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

2. Or run analysis programmatically:
```python
from src.data import load_financial_phrasebank, FinancialTextPreprocessor, DatasetAnalyzer
from src.visualization import plot_label_distribution, generate_all_wordclouds

# Load dataset
df = load_financial_phrasebank(agreement_level="sentences_75agree")

# Preprocess
preprocessor = FinancialTextPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df)

# Analyze
analyzer = DatasetAnalyzer(df_processed)
stats = analyzer.get_all_statistics()

# Generate plots
plot_label_distribution(df_processed, save=True)
generate_all_wordclouds(df_processed, save=True)
```

### Part 2: Model Training (Coming Soon)

```python
# Placeholder for Part 2
from src.models import SentimentClassifier

model = SentimentClassifier(model_name="roberta-base")
model.train(train_df)
predictions = model.predict(test_df)
```

## Key Features

- **Modular Architecture**: Reusable components for data loading, preprocessing, and analysis
- **Comprehensive EDA**: Statistical analysis, visualizations, and data quality checks
- **Publication-Ready Plots**: High-resolution figures with proper formatting
- **Logging**: Detailed logging for debugging and monitoring
- **Type Hints**: Full type annotations for better code quality

## Related Work

1. **FinBERT** (Araci, 2019) - Domain-specific BERT for financial sentiment
2. **Good Debt or Bad Debt** (Malo et al., 2014) - Original Financial PhraseBank paper
3. **Financial Sentiment Analysis** (Theil et al., 2018) - Common mistakes and best practices

## Results (Part 1)

### Dataset Statistics (75% Agreement Level)

| Metric | Value |
|--------|-------|
| Total Samples | 3,453 |
| Positive | 887 (25.7%) |
| Neutral | 2,146 (62.2%) |
| Negative | 420 (12.2%) |
| Imbalance Ratio | 5.11 |

### Text Characteristics

| Metric | Value |
|--------|-------|
| Avg. Word Count | 22.8 |
| Avg. Char Count | 124.9 |
| Min Words | 2 |
| Max Words | 81 |

### Data Quality

| Metric | Value |
|--------|-------|
| Missing Values | 0 |
| Duplicates | 5 (0.14%) |
| Quality Score | 99.9% |

### Agreement Level Comparison

| Agreement | Samples | Positive | Neutral | Negative |
|-----------|---------|----------|---------|----------|
| 50% | 4,846 | 1,363 | 2,879 | 604 |
| 66% | 4,217 | 1,168 | 2,535 | 514 |
| 75% | 3,453 | 887 | 2,146 | 420 |
| 100% | 2,264 | 570 | 1,391 | 303 |

## License

This project is for educational purposes.

## Acknowledgments

- Malo et al. for the Financial PhraseBank dataset
- HuggingFace for dataset hosting
- Course instructors and TAs
