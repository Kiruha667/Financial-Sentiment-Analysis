# Dataset Documentation

## Financial PhraseBank

### Overview

The Financial PhraseBank dataset consists of sentences from English financial news categorized by sentiment.

### Source

- **Paper**: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). "Good debt or bad debt: Detecting semantic orientations in economic texts"
- **HuggingFace**: https://huggingface.co/datasets/takala/financial_phrasebank
- **Original**: https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt

### Dataset Statistics

| Configuration | Samples | Description |
|--------------|---------|-------------|
| sentences_50agree | ~4,846 | At least 50% annotator agreement |
| sentences_66agree | ~4,217 | At least 66% annotator agreement |
| sentences_75agree | ~3,453 | At least 75% annotator agreement |
| sentences_allagree | ~2,264 | 100% annotator agreement |

### Label Distribution (sentences_75agree)

| Label | Count | Percentage |
|-------|-------|------------|
| Neutral | ~1,826 | ~52.9% |
| Positive | ~1,110 | ~32.1% |
| Negative | ~517 | ~15.0% |

### Data Format

**Columns:**
- `sentence`: The financial text (string)
- `label`: Numeric label (0=negative, 1=neutral, 2=positive)

### Data Files

```
data/
├── raw/
│   └── financial_phrasebank_sentences_75agree.csv
└── processed/
    └── financial_phrasebank_processed.csv
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-XX-XX | Initial dataset download |

### Usage Notes

1. The dataset is imbalanced with neutral being the majority class
2. Higher agreement levels have fewer samples but more reliable labels
3. Sentences are from financial news, primarily about company performance
4. Some sentences may contain company names, numbers, and financial terminology

### Citation

```bibtex
@article{malo2014good,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={Malo, Pekka and Sinha, Ankur and Korhonen, Pekka and Wallenius, Jyrki and Takala, Pyry},
  journal={Journal of the Association for Information Science and Technology},
  volume={65},
  number={4},
  pages={782--796},
  year={2014},
  publisher={Wiley}
}
```

### License

The dataset is available for research purposes. Please cite the original paper when using this dataset.
