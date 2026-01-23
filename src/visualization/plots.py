"""
Visualization functions for Financial Sentiment Analysis.

This module provides functions for creating publication-ready
visualizations of dataset statistics and analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import FIGURES_DIR
from config.params import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGSIZE_SMALL,
    FIGSIZE_MEDIUM,
    FIGSIZE_WIDE,
    SENTIMENT_COLORS,
    PLOT_PALETTE,
    WORDCLOUD_MAX_WORDS,
    WORDCLOUD_WIDTH,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_BACKGROUND,
)

# Configure logging
logger = logging.getLogger(__name__)


def set_plot_style() -> None:
    """Set consistent style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def _save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure to save.
        filename: Output filename (without extension).
        output_dir: Output directory. If None, uses default figures dir.

    Returns:
        Path to saved figure.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.{FIGURE_FORMAT}"

    fig.savefig(output_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
    logger.info(f"Figure saved to {output_path}")

    return output_path


def plot_label_distribution(
    df: pd.DataFrame,
    label_column: str = 'label_name',
    title: str = 'Sentiment Label Distribution',
    save: bool = True,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = FIGSIZE_SMALL,
    show_percentages: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a bar plot showing label distribution.

    Args:
        df: DataFrame with labels.
        label_column: Column name containing labels.
        title: Plot title.
        save: Whether to save the figure.
        output_dir: Output directory for saving.
        figsize: Figure size (width, height).
        show_percentages: Whether to show percentage labels on bars.

    Returns:
        Tuple of (Figure, Axes).
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Get counts and order
    order = ['negative', 'neutral', 'positive']
    counts = df[label_column].value_counts()

    # Create bar plot
    bars = ax.bar(
        order,
        [counts.get(label, 0) for label in order],
        color=[SENTIMENT_COLORS[label] for label in order],
        edgecolor='black',
        linewidth=1.2,
    )

    # Add value labels on bars
    total = len(df)
    for bar, label in zip(bars, order):
        height = bar.get_height()
        percentage = height / total * 100
        if show_percentages:
            label_text = f'{int(height):,}\n({percentage:.1f}%)'
        else:
            label_text = f'{int(height):,}'

        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax.set_xlabel('Sentiment Label', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust y-axis to make room for labels
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()

    if save:
        _save_figure(fig, 'label_distribution', output_dir)

    return fig, ax


def plot_text_length_distribution(
    df: pd.DataFrame,
    length_column: str = 'word_count',
    label_column: str = 'label_name',
    title: str = 'Text Length Distribution',
    save: bool = True,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create histogram and boxplot of text lengths.

    Args:
        df: DataFrame with text length data.
        length_column: Column name containing lengths.
        label_column: Column name containing labels.
        title: Plot title.
        save: Whether to save the figure.
        output_dir: Output directory for saving.
        figsize: Figure size (width, height).

    Returns:
        Tuple of (Figure, list of Axes).
    """
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1 = axes[0]
    for label in ['negative', 'neutral', 'positive']:
        if label in df[label_column].values:
            subset = df[df[label_column] == label][length_column]
            ax1.hist(
                subset,
                bins=30,
                alpha=0.6,
                label=label.capitalize(),
                color=SENTIMENT_COLORS[label],
                edgecolor='black',
                linewidth=0.5,
            )

    ax1.set_xlabel('Word Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Word Count Distribution by Sentiment', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Boxplot
    ax2 = axes[1]
    order = ['negative', 'neutral', 'positive']
    palette = {label: SENTIMENT_COLORS[label] for label in order}

    sns.boxplot(
        data=df,
        x=label_column,
        y=length_column,
        order=order,
        palette=palette,
        ax=ax2,
    )

    ax2.set_xlabel('Sentiment Label', fontsize=12)
    ax2.set_ylabel('Word Count', fontsize=12)
    ax2.set_title('Word Count by Sentiment Class', fontsize=12, fontweight='bold')

    # Add mean markers
    means = df.groupby(label_column)[length_column].mean()
    for i, label in enumerate(order):
        if label in means.index:
            ax2.scatter(i, means[label], color='red', s=100, zorder=5, marker='D')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        _save_figure(fig, 'text_length_distribution', output_dir)

    return fig, list(axes)


def plot_word_frequency(
    word_freq_df: pd.DataFrame,
    title: str = 'Top Words by Sentiment',
    save: bool = True,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = FIGSIZE_WIDE,
    top_n: int = 15,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create bar plots of word frequencies by sentiment.

    Args:
        word_freq_df: DataFrame with columns ['label', 'word', 'count'].
        title: Plot title.
        save: Whether to save the figure.
        output_dir: Output directory for saving.
        figsize: Figure size (width, height).
        top_n: Number of top words to show per class.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    set_plot_style()

    labels = ['negative', 'neutral', 'positive']
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, label in zip(axes, labels):
        label_data = word_freq_df[word_freq_df['label'] == label].head(top_n)

        if len(label_data) > 0:
            bars = ax.barh(
                label_data['word'],
                label_data['count'],
                color=SENTIMENT_COLORS[label],
                edgecolor='black',
                linewidth=0.5,
            )

            ax.set_xlabel('Frequency', fontsize=10)
            ax.set_title(f'{label.capitalize()}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f'{int(width)}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left',
                    va='center',
                    fontsize=8,
                )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label.capitalize()}', fontsize=12, fontweight='bold')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        _save_figure(fig, 'word_frequency', output_dir)

    return fig, list(axes)


def generate_wordcloud(
    texts: Union[List[str], pd.Series],
    label: str,
    title: Optional[str] = None,
    save: bool = True,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = FIGSIZE_MEDIUM,
    stopwords: Optional[set] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate word cloud from texts.

    Args:
        texts: List or Series of texts.
        label: Sentiment label for coloring and naming.
        title: Plot title. If None, auto-generated.
        save: Whether to save the figure.
        output_dir: Output directory for saving.
        figsize: Figure size (width, height).
        stopwords: Additional stopwords to exclude.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.error("wordcloud package not installed. Install with: pip install wordcloud")
        raise

    set_plot_style()

    # Combine all texts
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    combined_text = ' '.join(str(t) for t in texts if pd.notna(t))

    # Get color based on sentiment
    color = SENTIMENT_COLORS.get(label.lower(), '#333333')

    # Create colormap from sentiment color
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return color

    # Generate word cloud
    wc = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        max_words=WORDCLOUD_MAX_WORDS,
        background_color=WORDCLOUD_BACKGROUND,
        stopwords=stopwords,
        color_func=color_func,
        random_state=42,
    )

    wc.generate(combined_text)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    if title is None:
        title = f'Word Cloud - {label.capitalize()} Sentiment'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()

    if save:
        filename = f'wordcloud_{label.lower()}'
        _save_figure(fig, filename, output_dir)

    return fig, ax


def generate_all_wordclouds(
    df: pd.DataFrame,
    text_column: str = 'sentence',
    label_column: str = 'label_name',
    save: bool = True,
    output_dir: Optional[Path] = None,
    stopwords: Optional[set] = None,
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """
    Generate word clouds for all sentiment classes.

    Args:
        df: DataFrame with texts and labels.
        text_column: Column containing texts.
        label_column: Column containing labels.
        save: Whether to save figures.
        output_dir: Output directory for saving.
        stopwords: Additional stopwords to exclude.

    Returns:
        Dictionary mapping label to (Figure, Axes) tuple.
    """
    results = {}

    for label in ['negative', 'neutral', 'positive']:
        if label in df[label_column].values:
            texts = df[df[label_column] == label][text_column]
            fig, ax = generate_wordcloud(
                texts,
                label=label,
                save=save,
                output_dir=output_dir,
                stopwords=stopwords,
            )
            results[label] = (fig, ax)

    return results


def plot_sentiment_scatter(
    df: pd.DataFrame,
    x_column: str = 'word_count',
    label_column: str = 'label_name',
    title: str = 'Text Length vs Sentiment',
    save: bool = True,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = FIGSIZE_MEDIUM,
    sample_size: Optional[int] = 1000,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create scatter plot of text length vs sentiment.

    Args:
        df: DataFrame with data.
        x_column: Column for x-axis (typically word count).
        label_column: Column containing sentiment labels.
        title: Plot title.
        save: Whether to save the figure.
        output_dir: Output directory for saving.
        figsize: Figure size (width, height).
        sample_size: Number of points to sample for plotting (None for all).

    Returns:
        Tuple of (Figure, Axes).
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Sample data if needed
    plot_df = df.copy()
    if sample_size and len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    # Create numeric y-values with jitter for visualization
    label_to_y = {'negative': 0, 'neutral': 1, 'positive': 2}
    plot_df['y_value'] = plot_df[label_column].map(label_to_y)
    plot_df['y_jittered'] = plot_df['y_value'] + np.random.uniform(-0.2, 0.2, len(plot_df))

    # Plot each class
    for label in ['negative', 'neutral', 'positive']:
        subset = plot_df[plot_df[label_column] == label]
        ax.scatter(
            subset[x_column],
            subset['y_jittered'],
            c=SENTIMENT_COLORS[label],
            label=label.capitalize(),
            alpha=0.5,
            s=30,
            edgecolors='none',
        )

    ax.set_xlabel('Word Count', fontsize=12)
    ax.set_ylabel('Sentiment', fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add mean lines
    for label, y_pos in label_to_y.items():
        mean_x = df[df[label_column] == label][x_column].mean()
        ax.axvline(x=mean_x, color=SENTIMENT_COLORS[label], linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save:
        _save_figure(fig, 'sentiment_scatter', output_dir)

    return fig, ax


def create_all_plots(
    df: pd.DataFrame,
    preprocessor=None,
    text_column: str = 'sentence',
    label_column: str = 'label_name',
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generate all standard plots and save them.

    Args:
        df: DataFrame with data.
        preprocessor: FinancialTextPreprocessor instance for word frequencies.
        text_column: Column containing texts.
        label_column: Column containing labels.
        output_dir: Output directory for saving.

    Returns:
        Dictionary mapping plot names to saved file paths.
    """
    saved_files = {}

    logger.info("Generating all plots...")

    # 1. Label distribution
    fig, _ = plot_label_distribution(df, label_column, save=True, output_dir=output_dir)
    saved_files['label_distribution'] = FIGURES_DIR / 'label_distribution.png'
    plt.close(fig)

    # 2. Text length distribution
    if 'word_count' not in df.columns:
        df['word_count'] = df[text_column].str.split().str.len()

    fig, _ = plot_text_length_distribution(df, 'word_count', label_column, save=True, output_dir=output_dir)
    saved_files['text_length_distribution'] = FIGURES_DIR / 'text_length_distribution.png'
    plt.close(fig)

    # 3. Word frequency (if preprocessor provided)
    if preprocessor is not None:
        clean_col = 'sentence_clean' if 'sentence_clean' in df.columns else text_column
        word_freq_df = preprocessor.get_word_frequencies(df, clean_col, by_label=True)
        fig, _ = plot_word_frequency(word_freq_df, save=True, output_dir=output_dir)
        saved_files['word_frequency'] = FIGURES_DIR / 'word_frequency.png'
        plt.close(fig)

    # 4. Word clouds
    wordcloud_figs = generate_all_wordclouds(df, text_column, label_column, save=True, output_dir=output_dir)
    for label, (fig, _) in wordcloud_figs.items():
        saved_files[f'wordcloud_{label}'] = FIGURES_DIR / f'wordcloud_{label}.png'
        plt.close(fig)

    # 5. Sentiment scatter
    fig, _ = plot_sentiment_scatter(df, 'word_count', label_column, save=True, output_dir=output_dir)
    saved_files['sentiment_scatter'] = FIGURES_DIR / 'sentiment_scatter.png'
    plt.close(fig)

    logger.info(f"Generated {len(saved_files)} plots")

    return saved_files
