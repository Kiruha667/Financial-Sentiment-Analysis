# Financial Sentiment Analysis

Проект анализа тональности финансовых текстов с использованием трансформеров (FinBERT, RoBERTa).

## Быстрый старт

### 1. Установка

```bash
# Клонировать репозиторий
git clone <repository-url>
cd financial-sentiment-analysis

# Создать виртуальное окружение
python -m venv .venv

# Активировать (Windows)
.venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### 2. Запуск ноутбуков

```bash
# Активировать окружение
.venv\Scripts\activate

# Часть 1: Анализ данных
jupyter notebook notebooks/01_data_analysis.ipynb

# Часть 2: Обучение моделей
jupyter notebook notebooks/02_model_training.ipynb
```

### 3. Использование обученной модели

```python
from src.models import SentimentPredictor

# Загрузить модель (после обучения)
predictor = SentimentPredictor(
    model_path="outputs/models/finbert/best_model.pt",
    tokenizer_name="ProsusAI/finbert"
)

# Предсказать тональность
result = predictor.predict("The company reported record profits.")
print(f"Тональность: {result['prediction']}")  # positive/neutral/negative
print(f"Уверенность: {result['confidence']:.1%}")
```

---

## Описание проекта

**Задача:** Классификация финансовых новостей на 3 класса:
- **Positive** — позитивная тональность (рост, прибыль, успех)
- **Neutral** — нейтральная (факты без оценки)
- **Negative** — негативная (убытки, падение, проблемы)

**Датасет:** Financial PhraseBank (3,453 предложения, 75% agreement)

**Модели:**
| Модель | Описание | Ожидаемый F1 |
|--------|----------|--------------|
| FinBERT | Специализирована на финансовых текстах | ~0.85-0.90 |
| RoBERTa | Общего назначения (baseline) | ~0.78-0.82 |

---

## Структура проекта

```
project/
├── notebooks/
│   ├── 01_data_analysis.ipynb    # Часть 1: EDA и визуализация
│   └── 02_model_training.ipynb   # Часть 2: Обучение и оценка
│
├── src/                          # Исходный код
│   ├── data/                     # Загрузка и обработка данных
│   ├── models/                   # Модели и обучение
│   ├── visualization/            # Визуализация
│   └── utils/                    # Утилиты
│
├── config/                       # Конфигурация
│   ├── paths.py                  # Пути к файлам
│   ├── params.py                 # Гиперпараметры
│   └── model_config.py           # Настройки моделей
│
├── outputs/                      # Результаты
│   ├── figures/                  # Графики (PNG)
│   ├── models/                   # Чекпоинты моделей
│   ├── reports/                  # Отчеты
│   └── logs/                     # Логи
│
├── experiments/
│   └── results.json              # Результаты экспериментов
│
└── data/
    ├── raw/                      # Исходные данные
    ├── processed/                # Обработанные данные
    └── splits/                   # Train/Val/Test splits
```

---

## Пошаговое руководство

### Часть 1: Анализ данных

Ноутбук `01_data_analysis.ipynb` выполняет:
- Загрузку датасета Financial PhraseBank
- Статистический анализ (распределение классов, длины текстов)
- Визуализацию (гистограммы, wordcloud)
- Сохранение обработанных данных

**Результаты сохраняются в:**
- `outputs/figures/` — графики
- `outputs/reports/` — статистика
- `data/processed/` — обработанные данные

### Часть 2: Обучение моделей

Ноутбук `02_model_training.ipynb` выполняет:
1. Создание train/val/test splits (70/15/15)
2. Обучение RoBERTa-base (baseline)
3. Обучение FinBERT (domain-specific)
4. Оценка на тестовом наборе
5. Анализ ошибок
6. Демо инференса

**Результаты сохраняются в:**
- `outputs/models/roberta-base/best_model.pt`
- `outputs/models/finbert/best_model.pt`
- `outputs/figures/` — графики обучения
- `experiments/results.json` — метрики

---

## Примеры использования

### Загрузка данных

```python
from src.data import load_financial_phrasebank

# Загрузить датасет
df = load_financial_phrasebank(agreement_level="sentences_75agree")
print(f"Всего: {len(df)} примеров")
```

### Обучение модели

```python
from config import FINBERT_CONFIG
from src.data import load_financial_phrasebank, create_data_splits, create_dataloaders
from src.models import create_model, Trainer
from src.utils import get_device

# Данные
df = load_financial_phrasebank()
train_df, val_df, test_df = create_data_splits(df)
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df,
    tokenizer_name="ProsusAI/finbert",
    batch_size=16
)

# Модель
device = get_device()
model = create_model("ProsusAI/finbert", num_labels=3, device=device)

# Обучение
trainer = Trainer.from_config(model, train_loader, val_loader, config=FINBERT_CONFIG)
history = trainer.train()
```

### Оценка модели

```python
from src.models import ModelEvaluator, evaluate_model_on_test

# Инференс на тесте
predictions, labels, probabilities = evaluate_model_on_test(model, test_loader, device)

# Метрики
evaluator = ModelEvaluator()
metrics = evaluator.compute_metrics(predictions, labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")

# Отчет
print(evaluator.get_classification_report(predictions, labels))
```

### Инференс на новых данных

```python
from src.models import SentimentPredictor

predictor = SentimentPredictor(
    model_path="outputs/models/finbert/best_model.pt",
    tokenizer_name="ProsusAI/finbert"
)

# Одиночное предсказание
result = predictor.predict("Revenue increased by 25% year-over-year.")
print(result)
# {'prediction': 'positive', 'confidence': 0.94}

# С вероятностями
result = predictor.predict("The merger is pending approval.", return_probabilities=True)
print(result['probabilities'])
# {'negative': 0.05, 'neutral': 0.85, 'positive': 0.10}

# Батч предсказаний
texts = [
    "Profits soared to record highs.",
    "The company maintained stable operations.",
    "Losses widened amid market turmoil."
]
results = predictor.predict(texts)
for text, pred in zip(texts, results['predictions']):
    print(f"{pred}: {text}")
```

---

## Конфигурация моделей

Настройки в `config/model_config.py`:

| Параметр | FinBERT | RoBERTa |
|----------|---------|---------|
| Learning Rate | 1e-5 | 2e-5 |
| Batch Size | 16 | 16 |
| Epochs | 5 | 5 |
| Max Length | 128 | 128 |
| Early Stopping | 3 epochs | 3 epochs |

Изменить конфигурацию:
```python
from config import FINBERT_CONFIG

# Посмотреть настройки
print(FINBERT_CONFIG.learning_rate)  # 1e-5
print(FINBERT_CONFIG.batch_size)     # 16

# Или создать свою
from config import ModelConfig
my_config = ModelConfig(
    model_name="my-model",
    model_checkpoint="ProsusAI/finbert",
    learning_rate=5e-6,
    batch_size=8,
    num_epochs=10
)
```

---

## Результаты

### Датасет (75% agreement)

| Класс | Количество | Доля |
|-------|------------|------|
| Positive | 887 | 25.7% |
| Neutral | 2,146 | 62.2% |
| Negative | 420 | 12.2% |

### Метрики моделей

*Заполняются после запуска ноутбука обучения*

| Модель | Accuracy | F1 (weighted) | F1 (macro) |
|--------|----------|---------------|------------|
| RoBERTa | — | — | — |
| FinBERT | — | — | — |

---

## Файлы после обучения

```
outputs/
├── figures/
│   ├── data_splits_distribution.png
│   ├── roberta_training_history.png
│   ├── roberta_confusion_matrix.png
│   ├── finbert_training_history.png
│   ├── finbert_confusion_matrix.png
│   ├── model_comparison.png
│   ├── error_distributions.png
│   └── inference_demo.png
│
├── models/
│   ├── roberta-base/
│   │   ├── best_model.pt
│   │   └── history.json
│   └── finbert/
│       ├── best_model.pt
│       └── history.json
│
└── logs/
    └── training.log
```

---

## Требования

- Python 3.10+
- PyTorch 2.0+
- CUDA (рекомендуется для GPU)
- 6+ GB GPU VRAM (для batch_size=16)

Основные библиотеки:
- `transformers` — модели HuggingFace
- `datasets` — загрузка датасетов
- `scikit-learn` — метрики
- `matplotlib`, `seaborn` — визуализация
- `tqdm` — прогресс-бары

---

## Troubleshooting

**CUDA out of memory:**
```python
# Уменьшить batch_size
train_loader, val_loader, test_loader = create_dataloaders(
    ..., batch_size=8  # вместо 16
)
```

**Import errors в ноутбуке:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
```

**Windows DataLoader workers:**
```python
# Использовать num_workers=0
train_loader, val_loader, test_loader = create_dataloaders(
    ..., num_workers=0
)
```

---

## Лицензия

Проект создан в образовательных целях.

## Благодарности

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) — Malo et al.
- [FinBERT](https://huggingface.co/ProsusAI/finbert) — ProsusAI
- [HuggingFace Transformers](https://huggingface.co/transformers)
