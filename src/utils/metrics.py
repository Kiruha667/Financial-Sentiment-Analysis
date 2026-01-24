"""
Additional Metrics Utilities for Financial Sentiment Analysis.

This module provides:
- compute_additional_metrics: Cohen's Kappa, MCC, per-class accuracy
- bootstrap_confidence_interval: Bootstrap confidence intervals for metrics
"""

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

logger = logging.getLogger(__name__)


def compute_additional_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute additional classification metrics beyond standard ones.

    Args:
        predictions: Predicted labels [n_samples]
        labels: True labels [n_samples]

    Returns:
        Dictionary with:
            - cohen_kappa: Cohen's Kappa score (agreement accounting for chance)
            - matthews_corrcoef: Matthews Correlation Coefficient
            - per_class_accuracy: List of accuracy per class
    """
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, predictions)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, predictions)

    # Per-class accuracy
    unique_labels = np.unique(labels)
    per_class_accuracy = []

    for label in sorted(unique_labels):
        mask = labels == label
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == labels[mask])
        else:
            class_acc = 0.0
        per_class_accuracy.append(float(class_acc))

    return {
        'cohen_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'per_class_accuracy': per_class_accuracy,
    }


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        predictions: Predicted labels [n_samples]
        labels: True labels [n_samples]
        metric_fn: Function that takes (predictions, labels) and returns float
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval
    """
    np.random.seed(42)
    n_samples = len(labels)
    scores = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, size=n_samples)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]

        # Compute metric
        try:
            score = metric_fn(boot_preds, boot_labels)
            scores.append(score)
        except Exception:
            # Skip if metric computation fails (e.g., single class in sample)
            continue

    if len(scores) == 0:
        logger.warning("Bootstrap failed to produce any valid samples")
        return (0.0, 0.0)

    scores = np.array(scores)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(scores, alpha / 2 * 100)
    upper = np.percentile(scores, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def accuracy_score_fn(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy score function for bootstrap."""
    return float(np.mean(predictions == labels))


def f1_macro_score_fn(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Macro F1 score function for bootstrap."""
    from sklearn.metrics import f1_score
    return float(f1_score(labels, predictions, average='macro', zero_division=0))


def compute_metrics_with_ci(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Compute key metrics with bootstrap confidence intervals.

    Args:
        predictions: Predicted labels [n_samples]
        labels: True labels [n_samples]
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary mapping metric names to {value, lower, upper}
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Compute point estimates
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)

    # Compute confidence intervals
    acc_ci = bootstrap_confidence_interval(
        predictions, labels, accuracy_score_fn,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    f1_ci = bootstrap_confidence_interval(
        predictions, labels, f1_macro_score_fn,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level
    )

    return {
        'accuracy': {
            'value': float(accuracy),
            'lower': acc_ci[0],
            'upper': acc_ci[1],
        },
        'f1_macro': {
            'value': float(f1_macro),
            'lower': f1_ci[0],
            'upper': f1_ci[1],
        },
        'f1_weighted': {
            'value': float(f1_weighted),
            'lower': None,  # Not computed
            'upper': None,
        },
    }


def format_metric_with_ci(
    value: float,
    lower: float,
    upper: float,
    decimals: int = 4
) -> str:
    """
    Format a metric with confidence interval.

    Args:
        value: Point estimate
        lower: Lower bound of CI
        upper: Upper bound of CI
        decimals: Number of decimal places

    Returns:
        Formatted string like "0.8500 [0.8200, 0.8800]"
    """
    fmt = f".{decimals}f"
    return f"{value:{fmt}} [{lower:{fmt}}, {upper:{fmt}}]"


def compute_statistical_tests(
    model1_predictions: np.ndarray,
    model2_predictions: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute statistical tests comparing two models.

    Uses McNemar's test to determine if the difference between models
    is statistically significant.

    Args:
        model1_predictions: Predictions from model 1
        model2_predictions: Predictions from model 2
        labels: True labels

    Returns:
        Dictionary with:
            - mcnemar_statistic: Chi-squared statistic
            - mcnemar_pvalue: p-value for McNemar's test
            - model1_better_count: Cases where only model 1 is correct
            - model2_better_count: Cases where only model 2 is correct
    """
    # Find correct predictions for each model
    model1_correct = model1_predictions == labels
    model2_correct = model2_predictions == labels

    # Build contingency table
    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    b = np.sum(model1_correct & ~model2_correct)
    c = np.sum(~model1_correct & model2_correct)

    # McNemar's test (with continuity correction)
    if b + c == 0:
        statistic = 0.0
        pvalue = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy import stats
        pvalue = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'mcnemar_statistic': float(statistic),
        'mcnemar_pvalue': float(pvalue),
        'model1_better_count': int(b),
        'model2_better_count': int(c),
    }
