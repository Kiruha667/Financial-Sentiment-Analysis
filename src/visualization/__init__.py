"""Visualization modules."""

from src.visualization.plots import (
    plot_label_distribution,
    plot_text_length_distribution,
    plot_word_frequency,
    generate_wordcloud,
    plot_sentiment_scatter,
    set_plot_style,
)
from src.visualization.training_viz import (
    plot_training_history,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_model_comparison,
    plot_error_distribution,
    plot_learning_rate,
    plot_confidence_distribution,
    create_evaluation_report,
)

__all__ = [
    # Data visualization (Part 1)
    "plot_label_distribution",
    "plot_text_length_distribution",
    "plot_word_frequency",
    "generate_wordcloud",
    "plot_sentiment_scatter",
    "set_plot_style",
    # Training visualization (Part 2C)
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_model_comparison",
    "plot_error_distribution",
    "plot_learning_rate",
    "plot_confidence_distribution",
    "create_evaluation_report",
]
