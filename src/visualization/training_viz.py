"""
Training Visualization Functions for Financial Sentiment Analysis.

This module provides publication-quality visualizations for:
- Training history (loss and metrics curves)
- Confusion matrices
- Per-class performance metrics
- Model comparison charts
- Error distribution analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

logger = logging.getLogger(__name__)


def plot_training_history(
    history: Union[Dict[str, List[float]], object],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot training and validation loss/metrics over epochs.

    Args:
        history: Dictionary or TrainingHistory object with keys:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - val_acc or val_accuracy: List of validation accuracies
            - val_f1 (optional): List of validation F1 scores
        save_path: Path to save figure (optional)
        title: Overall figure title

    Returns:
        Tuple of (Figure, list of Axes)
    """
    # Convert to dict if needed
    if hasattr(history, 'to_dict'):
        history = history.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Get epochs (1-indexed)
    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Subplot 1: Loss curves
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'o-',
                 label='Training Loss', linewidth=2, markersize=6)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'o-',
                 label='Validation Loss', linewidth=2, markersize=6)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Validation metrics
    ax2 = axes[1]

    # Try different key names for accuracy
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    if val_acc_key in history:
        ax2.plot(epochs, history[val_acc_key], 'o-',
                 label='Validation Accuracy', linewidth=2, markersize=6)

    if 'val_f1' in history:
        ax2.plot(epochs, history['val_f1'], 'o-',
                 label='Validation F1', linewidth=2, markersize=6)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")

    plt.show()

    return fig, list(axes)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix [n_classes, n_classes]
        labels: List of class labels
        save_path: Path to save figure (optional)
        title: Figure title
        normalize: If True, display as percentages

    Returns:
        Tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize if requested
    if normalize:
        cm_display = cm.astype(float)
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_display = cm_display / row_sums
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")

    plt.show()

    return fig, ax


def plot_per_class_metrics(
    metrics: Dict,
    labels: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Class Performance"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot per-class precision, recall, and F1 scores as grouped bar chart.

    Args:
        metrics: Dictionary with keys:
            - precision_per_class: List of precision values
            - recall_per_class: List of recall values
            - f1_per_class: List of F1 values
        labels: List of class labels
        save_path: Path to save figure (optional)
        title: Figure title

    Returns:
        Tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.25

    precision = metrics.get('precision_per_class', [0] * len(labels))
    recall = metrics.get('recall_per_class', [0] * len(labels))
    f1 = metrics.get('f1_per_class', [0] * len(labels))

    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision',
                   alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall',
                   alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score',
                   alpha=0.8, color='#e74c3c')

    # Add value labels on bars
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    add_bar_labels(bars1)
    add_bar_labels(bars2)
    add_bar_labels(bars3)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.capitalize() for label in labels])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class metrics plot to {save_path}")

    plt.show()

    return fig, ax


def plot_model_comparison(
    model1_metrics: Dict[str, float],
    model2_metrics: Dict[str, float],
    model1_name: str,
    model2_name: str,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot side-by-side comparison of two models.

    Args:
        model1_metrics: Metrics dictionary for model 1
        model2_metrics: Metrics dictionary for model 2
        model1_name: Display name for model 1
        model2_name: Display name for model 2
        save_path: Path to save figure (optional)

    Returns:
        Tuple of (Figure, Axes)
    """
    metrics_keys = [
        'accuracy',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
        'f1_macro'
    ]
    metric_labels = [
        'Accuracy',
        'Precision',
        'Recall',
        'F1 (weighted)',
        'F1 (macro)'
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metric_labels))
    width = 0.35

    values1 = [model1_metrics.get(k, 0.0) for k in metrics_keys]
    values2 = [model2_metrics.get(k, 0.0) for k in metrics_keys]

    # Create bars
    bars1 = ax.bar(x - width/2, values1, width, label=model1_name,
                   alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, values2, width, label=model2_name,
                   alpha=0.8, color='#e74c3c')

    # Add value labels
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    add_bar_labels(bars1)
    add_bar_labels(bars2)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")

    plt.show()

    return fig, ax


def plot_error_distribution(
    error_analysis: Dict,
    save_path: Optional[str] = None,
    top_n: int = 6
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of error types as horizontal bar chart.

    Args:
        error_analysis: Output from ModelEvaluator.analyze_errors()
        save_path: Path to save figure (optional)
        top_n: Number of top error types to show

    Returns:
        Tuple of (Figure, Axes)
    """
    error_type_counts = error_analysis.get('error_type_counts', {})

    if not error_type_counts:
        logger.warning("No errors to plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No errors found',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig, ax

    # Get top N error types
    sorted_errors = sorted(
        error_type_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    error_types = [item[0] for item in sorted_errors]
    counts = [item[1] for item in sorted_errors]

    # Reverse for horizontal bar chart (top error at top)
    error_types = error_types[::-1]
    counts = counts[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color gradient
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(error_types)))

    # Create horizontal bars
    bars = ax.barh(error_types, counts, color=colors[::-1], edgecolor='black',
                   linewidth=0.5)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.annotate(f'{count}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Errors', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {min(top_n, len(error_type_counts))} Error Types',
                 fontsize=14, fontweight='bold')
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")

    plt.show()

    return fig, ax


def plot_learning_rate(
    history: Union[Dict, object],
    save_path: Optional[str] = None,
    title: str = "Learning Rate Schedule"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot learning rate over training steps.

    Args:
        history: Dictionary or TrainingHistory with 'learning_rates' key
        save_path: Path to save figure (optional)
        title: Figure title

    Returns:
        Tuple of (Figure, Axes)
    """
    if hasattr(history, 'to_dict'):
        history = history.to_dict()

    learning_rates = history.get('learning_rates', [])

    if not learning_rates:
        logger.warning("No learning rate data to plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No learning rate data',
                ha='center', va='center', fontsize=14)
        return fig, ax

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, 'b-', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning rate plot to {save_path}")

    plt.show()

    return fig, ax


def plot_confidence_distribution(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot confidence distribution for correct vs incorrect predictions.

    Args:
        probabilities: Probability distributions [n_samples, n_classes]
        predictions: Predicted labels [n_samples]
        labels: True labels [n_samples]
        label_names: List of class labels
        save_path: Path to save figure (optional)

    Returns:
        Tuple of (Figure, list of Axes)
    """
    # Get max confidence for each prediction
    confidences = np.max(probabilities, axis=1)

    # Separate correct and incorrect
    correct_mask = predictions == labels
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Histogram comparison
    ax1 = axes[0]
    ax1.hist(correct_conf, bins=30, alpha=0.7, label='Correct',
             color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax1.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect',
             color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Box plot by class
    ax2 = axes[1]
    data = []
    positions = []
    colors = []

    for i, label in enumerate(label_names):
        mask = labels == i
        if np.any(mask):
            data.append(confidences[mask])
            positions.append(i)
            colors.append(['#3498db', '#2ecc71', '#e74c3c'][i % 3])

    if data:
        bp = ax2.boxplot(data, positions=positions, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_xticks(range(len(label_names)))
    ax2.set_xticklabels([l.capitalize() for l in label_names])
    ax2.set_xlabel('True Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence by True Class', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confidence distribution plot to {save_path}")

    plt.show()

    return fig, list(axes)


def create_evaluation_report(
    metrics: Dict,
    confusion_mat: np.ndarray,
    label_names: List[str],
    error_analysis: Dict,
    output_dir: str,
    model_name: str = "Model"
) -> Dict[str, str]:
    """
    Generate all evaluation plots and save to directory.

    Args:
        metrics: Metrics dictionary from ModelEvaluator.compute_metrics()
        confusion_mat: Confusion matrix
        label_names: List of class labels
        error_analysis: Output from ModelEvaluator.analyze_errors()
        output_dir: Directory to save plots
        model_name: Name of the model for titles

    Returns:
        Dictionary mapping plot names to saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Confusion matrix
    path = output_dir / f"{model_name.lower()}_confusion_matrix.png"
    plot_confusion_matrix(
        confusion_mat, label_names,
        save_path=str(path),
        title=f"{model_name} Confusion Matrix"
    )
    saved_files['confusion_matrix'] = str(path)
    plt.close()

    # Normalized confusion matrix
    path = output_dir / f"{model_name.lower()}_confusion_matrix_normalized.png"
    plot_confusion_matrix(
        confusion_mat, label_names,
        save_path=str(path),
        title=f"{model_name} Confusion Matrix (Normalized)",
        normalize=True
    )
    saved_files['confusion_matrix_normalized'] = str(path)
    plt.close()

    # Per-class metrics
    path = output_dir / f"{model_name.lower()}_per_class_metrics.png"
    plot_per_class_metrics(
        metrics, label_names,
        save_path=str(path),
        title=f"{model_name} Per-Class Performance"
    )
    saved_files['per_class_metrics'] = str(path)
    plt.close()

    # Error distribution
    path = output_dir / f"{model_name.lower()}_error_distribution.png"
    plot_error_distribution(
        error_analysis,
        save_path=str(path)
    )
    saved_files['error_distribution'] = str(path)
    plt.close()

    logger.info(f"Generated {len(saved_files)} evaluation plots in {output_dir}")

    return saved_files
