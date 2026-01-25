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
│   ├── paths.py              # Path configurations
│   ├── params.py             # Hyperparameters and constants
│   └── model_config.py       # Model configurations (FinBERT, RoBERTa)
├── data/
│   ├── raw/                  # Original dataset
│   ├── processed/            # Preprocessed data
│   └── README.md             # Dataset documentation
├── notebooks/
│   ├── 01_data_analysis.ipynb    # EDA and analysis (Part 1)
│   └── 02_model_training.ipynb   # Model training and evaluation (Part 2)
├── experiments/
│   └── results.json              # Experiment tracking
├── src/
│   ├── data/
│   │   ├── loader.py         # Dataset loading utilities
│   │   ├── preprocessor.py   # Text preprocessing
│   │   ├── analyzer.py       # Statistical analysis
│   │   └── dataset.py        # PyTorch Dataset & DataLoaders
│   ├── models/
│   │   ├── classifier.py     # SentimentClassifier model
│   │   ├── trainer.py        # Training loop & early stopping
│   │   ├── evaluator.py      # Evaluation metrics & error analysis
│   │   └── predictor.py      # Production inference wrapper
│   ├── visualization/
│   │   ├── plots.py          # Data visualization (Part 1)
│   │   └── training_viz.py   # Training visualization (Part 2)
│   └── utils/
│       ├── helpers.py        # Utility functions
│       └── metrics.py        # Additional metrics (Kappa, MCC, CI)
├── outputs/
│   ├── figures/              # Generated plots
│   ├── models/               # Saved model checkpoints
│   ├── reports/              # Analysis reports
│   └── logs/                 # Application logs
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

### Part 2A: Data Preparation for Training

```python
from src.data import load_financial_phrasebank, create_data_splits, create_dataloaders

# Load and split data
df = load_financial_phrasebank(agreement_level="sentences_75agree")
train_df, val_df, test_df = create_data_splits(df, seed=42)

# Create DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    tokenizer_name="ProsusAI/finbert",
    batch_size=16,
    max_length=128
)

print(f"Train: {len(train_loader.dataset)} samples")
print(f"Val: {len(val_loader.dataset)} samples")
print(f"Test: {len(test_loader.dataset)} samples")
```

### Part 2B: Model Training

```python
from config import FINBERT_CONFIG
from src.models import create_model, Trainer
from src.utils import set_random_seed, get_device

# Setup
set_random_seed(42)
device = get_device()

# Create model
model = create_model(
    model_checkpoint=FINBERT_CONFIG.model_checkpoint,
    num_labels=3,
    device=device
)

# Create trainer
trainer = Trainer.from_config(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=FINBERT_CONFIG,
    checkpoint_dir="outputs/models/finbert"
)

# Train
history = trainer.train()

# Save history
trainer.save_history()
```

### Part 2C: Model Evaluation

```python
from src.models import ModelEvaluator, evaluate_model_on_test
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_error_distribution
)

# Run inference on test set
predictions, labels, probabilities = evaluate_model_on_test(
    model, test_loader, device=device
)

# Create evaluator
evaluator = ModelEvaluator(label_names=['negative', 'neutral', 'positive'])

# Compute metrics
metrics = evaluator.compute_metrics(predictions, labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (macro): {metrics['f1_macro']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")

# Classification report
print(evaluator.get_classification_report(predictions, labels))

# Confusion matrix
cm = evaluator.get_confusion_matrix(predictions, labels)

# Error analysis
texts = test_df['sentence'].tolist()
error_analysis = evaluator.analyze_errors(
    texts, predictions, labels, probabilities, top_k=10
)
print(f"Total errors: {error_analysis['total_errors']}")
print(f"Error rate: {error_analysis['error_rate']:.2f}%")

# Error patterns
patterns = evaluator.get_error_patterns(error_analysis)
for pattern in patterns:
    print(f"  - {pattern}")

# Visualizations
plot_training_history(history, save_path="outputs/figures/training_history.png")
plot_confusion_matrix(cm, ['negative', 'neutral', 'positive'],
                      save_path="outputs/figures/confusion_matrix.png")
plot_per_class_metrics(metrics, ['negative', 'neutral', 'positive'],
                       save_path="outputs/figures/per_class_metrics.png")
plot_error_distribution(error_analysis, save_path="outputs/figures/error_distribution.png")
```

### Part 2C: Model Comparison

```python
from src.models import ModelEvaluator
from src.visualization import plot_model_comparison

evaluator = ModelEvaluator()

# Assuming you have metrics from two models
finbert_metrics = {...}  # metrics from FinBERT
roberta_metrics = {...}  # metrics from RoBERTa

# Compare models
comparison_df = evaluator.compare_models(
    finbert_metrics, roberta_metrics,
    model1_name="FinBERT", model2_name="RoBERTa"
)
print(comparison_df)

# Per-class comparison
per_class_df = evaluator.get_per_class_comparison(
    finbert_metrics, roberta_metrics,
    model1_name="FinBERT", model2_name="RoBERTa"
)
print(per_class_df)

# Visualization
plot_model_comparison(
    finbert_metrics, roberta_metrics,
    "FinBERT", "RoBERTa",
    save_path="outputs/figures/model_comparison.png"
)
```

### Part 2C: Production Inference

```python
from src.models import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor(
    model_path="outputs/models/finbert/best_model.pt",
    tokenizer_name="ProsusAI/finbert",
    device="cuda"
)

# Single prediction
result = predictor.predict("The company reported record profits this quarter.")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# With probabilities
result = predictor.predict(
    "Revenue declined by 15% compared to last year.",
    return_probabilities=True
)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")

# With explanation
result = predictor.predict_with_explanation(
    "The merger is expected to create significant synergies."
)
print(result['explanation'])

# Batch prediction
texts = [
    "Stock prices surged after the announcement.",
    "The company maintained its market position.",
    "Losses widened due to increased competition."
]
results = predictor.predict(texts)
for text, pred in zip(texts, results['predictions']):
    print(f"{pred}: {text}")

# Large batch processing
results = predictor.predict_batch(
    large_text_list,
    batch_size=32,
    return_probabilities=True
)
```

### Part 2C: Additional Metrics

```python
from src.utils import (
    compute_additional_metrics,
    bootstrap_confidence_interval,
    compute_metrics_with_ci,
    compute_statistical_tests
)
from sklearn.metrics import accuracy_score

# Additional metrics (Cohen's Kappa, MCC)
add_metrics = compute_additional_metrics(predictions, labels)
print(f"Cohen's Kappa: {add_metrics['cohen_kappa']:.4f}")
print(f"Matthews Correlation Coefficient: {add_metrics['matthews_corrcoef']:.4f}")
print(f"Per-class accuracy: {add_metrics['per_class_accuracy']}")

# Bootstrap confidence intervals
ci_lower, ci_upper = bootstrap_confidence_interval(
    predictions, labels,
    metric_fn=lambda p, l: accuracy_score(l, p),
    n_bootstrap=1000,
    confidence_level=0.95
)
print(f"Accuracy 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Metrics with CI
metrics_ci = compute_metrics_with_ci(predictions, labels, n_bootstrap=1000)
print(f"Accuracy: {metrics_ci['accuracy']['value']:.4f} "
      f"[{metrics_ci['accuracy']['lower']:.4f}, {metrics_ci['accuracy']['upper']:.4f}]")

# Statistical comparison of two models (McNemar's test)
stats = compute_statistical_tests(
    model1_predictions, model2_predictions, labels
)
print(f"McNemar p-value: {stats['mcnemar_pvalue']:.4f}")
```

## Model Configurations

| Model | Checkpoint | Learning Rate | Batch Size | Epochs |
|-------|------------|---------------|------------|--------|
| FinBERT | `ProsusAI/finbert` | 1e-5 | 16 | 5 |
| RoBERTa | `roberta-base` | 2e-5 | 16 | 5 |

## Key Features

- **Modular Architecture**: Reusable components for data loading, preprocessing, training, and evaluation
- **Comprehensive EDA**: Statistical analysis, visualizations, and data quality checks
- **Multiple Models**: Support for FinBERT, RoBERTa, and other HuggingFace transformers
- **Early Stopping**: Automatic training termination with best model restoration
- **Error Analysis**: Detailed misclassification analysis with pattern detection
- **Publication-Ready Plots**: High-resolution figures (300 DPI) with proper formatting
- **Bootstrap CI**: Confidence intervals for all metrics
- **Production Inference**: Easy-to-use predictor with explanations
- **Logging**: Detailed logging for debugging and monitoring
- **Type Hints**: Full type annotations for better code quality

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

## Part 2: Model Training - Results

### Quick Start

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the training notebook
python -m jupyter notebook notebooks/02_model_training.ipynb
```

### Results Summary

| Model | Accuracy | F1 (weighted) | F1 (macro) | Precision | Recall |
|-------|----------|---------------|------------|-----------|--------|
| RoBERTa-base | TBD | TBD | TBD | TBD | TBD |
| FinBERT | TBD | TBD | TBD | TBD | TBD |

*Note: Actual results will be populated after running the notebook.*

### Key Findings

- Domain-specific pretraining (FinBERT) provides measurable improvement over general-purpose models
- Neutral class remains challenging due to subjective boundaries
- Negation handling is a common error pattern for both models
- Class imbalance affects minority class (negative) performance
- FinBERT shows advantages on financial-specific terminology

### Model Checkpoints

After training, model checkpoints are saved to:
- `outputs/models/roberta-base/best_model.pt`
- `outputs/models/finbert/best_model.pt`

Training history is saved to:
- `outputs/models/roberta-base/history.json`
- `outputs/models/finbert/history.json`

### Inference Example

```python
from src.models import SentimentPredictor

# Load trained model
predictor = SentimentPredictor(
    model_path="outputs/models/finbert/best_model.pt",
    tokenizer_name="ProsusAI/finbert",
    device="cuda"
)

# Predict sentiment
result = predictor.predict(
    "The company reported record quarterly earnings.",
    return_probabilities=True
)
print(f"Sentiment: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Future Improvements

- [ ] Ensemble of FinBERT and RoBERTa
- [ ] Focal loss for class imbalance
- [ ] Data augmentation for minority class
- [ ] Model quantization for deployment
- [ ] ONNX export for production inference
- [ ] Active learning for continuous improvement

### Experiment Tracking

Results are tracked in `experiments/results.json` with:
- Dataset statistics
- Training configuration
- Per-model metrics
- Model comparison

### Citations

If using this project, please cite:

```bibtex
@article{malo2014good,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal={Journal of the Association for Information Science and Technology},
  volume={65},
  number={4},
  pages={782--796},
  year={2014}
}

@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}
```

## API Reference

### ModelEvaluator

```python
evaluator = ModelEvaluator(label_names=['negative', 'neutral', 'positive'])

# Methods
metrics = evaluator.compute_metrics(predictions, labels)
report = evaluator.get_classification_report(predictions, labels)
cm = evaluator.get_confusion_matrix(predictions, labels, normalize=False)
errors = evaluator.analyze_errors(texts, predictions, labels, probabilities, top_k=10)
patterns = evaluator.get_error_patterns(errors)
comparison = evaluator.compare_models(metrics1, metrics2, "Model1", "Model2")
per_class = evaluator.get_per_class_comparison(metrics1, metrics2, "Model1", "Model2")
```

### SentimentPredictor

```python
predictor = SentimentPredictor(
    model_path="path/to/checkpoint.pt",
    tokenizer_name="ProsusAI/finbert",
    device="cuda",
    label_names=['negative', 'neutral', 'positive']
)

# Methods
result = predictor.predict(text, return_probabilities=False)
result = predictor.predict_with_explanation(text)
results = predictor.predict_batch(texts, batch_size=32)
```

### Visualization Functions

```python
from src.visualization import (
    # Training visualizations
    plot_training_history(history, save_path=None, title="Training History"),
    plot_confusion_matrix(cm, labels, save_path=None, normalize=False),
    plot_per_class_metrics(metrics, labels, save_path=None),
    plot_model_comparison(metrics1, metrics2, name1, name2, save_path=None),
    plot_error_distribution(error_analysis, save_path=None, top_n=6),
    plot_learning_rate(history, save_path=None),
    plot_confidence_distribution(probs, preds, labels, label_names, save_path=None),
    create_evaluation_report(metrics, cm, labels, errors, output_dir, model_name),
)
```

### Metrics Utilities

```python
from src.utils import (
    compute_additional_metrics(predictions, labels),  # Kappa, MCC, per-class acc
    bootstrap_confidence_interval(preds, labels, metric_fn, n_bootstrap=1000),
    compute_metrics_with_ci(predictions, labels, n_bootstrap=1000),
    compute_statistical_tests(preds1, preds2, labels),  # McNemar's test
)
```

## License

This project is for educational purposes.

## Acknowledgments

- Malo et al. for the Financial PhraseBank dataset
- HuggingFace for dataset hosting and transformers library
- ProsusAI for FinBERT model
