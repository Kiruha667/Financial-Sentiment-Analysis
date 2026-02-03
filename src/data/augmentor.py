"""
Data augmentation for Financial Sentiment Analysis.

Balances the training set using three sources (priority order):
1. External HuggingFace datasets (real human-written text)
2. nlpaug augmentation (synonym replacement, insert, delete, back-translation)
3. Template-generated sentences (capped at TEMPLATE_MAX_RATIO of deficit)

Only the training set is augmented — val/test remain clean.
"""

import logging
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.params import (
    AUGMENTATION_TECHNIQUES,
    AUGMENT_DELETE_MIN_TOKENS,
    AUGMENT_INSERT_TOP_K,
    AUGMENT_SYNONYM_TOP_K,
    BACK_TRANSLATION_SRC_MODEL,
    BACK_TRANSLATION_TGT_MODEL,
    LABEL_NAMES,
    RANDOM_SEED,
    TEMPLATE_MAX_RATIO,
    TWITTER_FINANCIAL_DATASET,
)
from config.paths import AUGMENTED_TRAIN_CSV, AUGMENTED_DATA_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template data for generation
# ---------------------------------------------------------------------------

_COMPANIES = [
    "Nokia", "Ericsson", "Fortum", "Stora Enso", "UPM-Kymmene",
    "Outokumpu", "Wärtsilä", "Metso", "Kone", "Neste",
    "Nordea", "Sampo", "Elisa", "Telia", "Huhtamäki",
    "Cargotec", "Valmet", "Orion", "Kemira", "Ahlstrom",
    "Finnair", "Tieto", "Sanoma", "Stockmann", "Fiskars",
    "Raisio", "Atria", "HKScan", "Olvi", "Rapala",
]

_METRICS = [
    "net sales", "revenue", "operating profit", "net income",
    "EBITDA", "turnover", "gross margin", "operating income",
    "earnings per share", "cash flow from operations",
]

_POSITIVE_TEMPLATES = [
    "{company} reported a {pct}% increase in {metric} for the period.",
    "{metric} of {company} rose to EUR {amount} million, up from EUR {amount_prev} million.",
    "{company}'s {metric} improved significantly, reaching EUR {amount} million.",
    "The Board of Directors of {company} proposes a dividend of EUR {div} per share.",
    "{company} expects {metric} to grow in the coming quarter.",
    "{company} won a new contract worth EUR {amount} million.",
    "Strong demand boosted {company}'s {metric} by {pct}% year-on-year.",
    "{company} announced plans to expand operations in new markets.",
    "Analysts raised their price target for {company} shares following strong results.",
    "{company} achieved record-high {metric} in the reporting period.",
    "The order intake of {company} increased by {pct}% compared to the previous year.",
    "{company} reported better-than-expected results for the fiscal year.",
    "Profit before taxes of {company} rose {pct}% to EUR {amount} million.",
    "{company}'s market share grew to {pct}% in the segment.",
    "{company} signed a strategic partnership to accelerate growth.",
    "Customer satisfaction scores at {company} reached an all-time high.",
    "{company}'s cost reduction program delivered savings of EUR {amount} million.",
    "Sales in {company}'s core business area grew by {pct}%.",
    "{company} completed the acquisition, strengthening its market position.",
    "{company} raised its full-year guidance for {metric}.",
    "The volume of orders received by {company} surged {pct}% during the quarter.",
    "{company}'s share price gained {pct}% after the positive earnings report.",
    "{company} increased its workforce by {pct_small}% to meet growing demand.",
    "Demand for {company}'s products remained strong across all regions.",
    "{company} inaugurated a new production facility in Europe.",
    "Operating cash flow at {company} improved to EUR {amount} million.",
    "{company} successfully launched a new product line in the market.",
    "{company} reported an improvement of {pct}% in {metric}.",
    "Recurring revenue at {company} increased {pct}% year-over-year.",
    "{company}'s management expressed confidence in continued growth.",
    "{company} delivered solid performance with {metric} up {pct}%.",
    "{company} received a major order valued at EUR {amount} million.",
    "The CEO of {company} highlighted the strong growth trajectory.",
    "{company}'s exports grew {pct}% during the first half of the year.",
    "{company} declared a special dividend following exceptional results.",
    "Organic growth at {company} was {pct}% for the reporting period.",
    "{company} outperformed market expectations with robust {metric}.",
    "{company} expanded its customer base by {pct}% year-on-year.",
    "{company}'s profitability improved due to higher volumes and lower costs.",
    "Net profit at {company} rose {pct}% to EUR {amount} million.",
    "{company}'s backlog of orders reached EUR {amount} million, a new high.",
    "Analysts upgraded {company}'s stock rating following the quarterly report.",
    "{company} reduced its net debt by EUR {amount} million during the period.",
    "{company}'s return on equity improved to {pct}% in the fiscal year.",
    "The strong order book gives {company} good visibility for the next quarters.",
    "{company} achieved positive free cash flow of EUR {amount} million.",
    "{company}'s earnings exceeded consensus estimates by {pct}%.",
    "Production efficiency at {company} improved by {pct}%.",
    "{company} reported continued momentum in all business segments.",
    "{company}'s annual report showed growth across every key metric.",
]

_NEGATIVE_TEMPLATES = [
    "{company} reported a {pct}% decline in {metric} for the period.",
    "{metric} of {company} fell to EUR {amount} million, down from EUR {amount_prev} million.",
    "{company}'s {metric} decreased significantly to EUR {amount} million.",
    "{company} announced a restructuring plan involving {headcount} job cuts.",
    "{company} expects {metric} to decline in the coming quarter.",
    "Weak demand reduced {company}'s {metric} by {pct}% year-on-year.",
    "{company} issued a profit warning due to deteriorating market conditions.",
    "Analysts downgraded {company} shares following disappointing results.",
    "{company} recorded a net loss of EUR {amount} million for the period.",
    "The operating loss of {company} widened to EUR {amount} million.",
    "Order intake at {company} dropped {pct}% compared to the previous year.",
    "{company}'s market share shrank to {pct_small}% in the segment.",
    "{company} faces increasing cost pressures in its key markets.",
    "Profit before taxes of {company} fell {pct}% to EUR {amount} million.",
    "{company} cut its full-year guidance for {metric}.",
    "{company}'s share price declined {pct}% after the earnings report.",
    "Rising raw material costs weighed on {company}'s profitability.",
    "{company} postponed planned investments due to uncertain demand.",
    "{company} reported weaker-than-expected results for the quarter.",
    "Cash flow from operations at {company} decreased to EUR {amount} million.",
    "Sales in {company}'s main segment contracted by {pct}%.",
    "{company} closed a production facility due to low utilisation.",
    "{company} reported an impairment charge of EUR {amount} million.",
    "Net sales of {company} decreased {pct}% to EUR {amount} million.",
    "{company}'s debt-to-equity ratio deteriorated during the reporting period.",
    "{company} experienced supply chain disruptions that hurt production.",
    "Customer churn at {company} increased by {pct}% year-over-year.",
    "The Board of {company} decided to suspend dividend payments.",
    "{company}'s return on invested capital fell below the target level.",
    "Foreign exchange losses negatively impacted {company}'s bottom line.",
    "{company} wrote down assets worth EUR {amount} million in the quarter.",
    "{company} lost a major contract to a competitor in the segment.",
    "{company}'s revenue missed analyst estimates by {pct}%.",
    "Inventory levels at {company} rose significantly, signalling weak demand.",
    "Operating margins at {company} compressed by {pct} percentage points.",
    "{company} lowered its dividend proposal to EUR {div} per share.",
    "{company}'s earnings per share fell {pct}% compared to the prior year.",
    "Weak export demand contributed to {company}'s lower-than-expected {metric}.",
    "{company} announced the closure of operations in one of its markets.",
    "{company} reported negative free cash flow of EUR {amount} million.",
    "{company}'s management acknowledged the challenging market environment.",
    "{company} incurred restructuring costs of EUR {amount} million.",
    "The competitive landscape worsened for {company} during the quarter.",
    "{company}'s stock was removed from a major index.",
    "Order cancellations at {company} increased during the period.",
    "{company} delayed the launch of a key product due to technical issues.",
    "Labor disputes at {company} disrupted production for several weeks.",
    "{company} faced regulatory penalties totaling EUR {amount} million.",
    "{company}'s credit rating was downgraded by the rating agency.",
    "Recurring revenue at {company} contracted {pct}% year-over-year.",
]

_NEUTRAL_TEMPLATES = [
    "{company} will publish its annual report on {date}.",
    "{company} appointed {name} as the new Chief Financial Officer.",
    "The Annual General Meeting of {company} will be held on {date}.",
    "{company} transferred its listing to the NASDAQ OMX Helsinki exchange.",
    "{company} and {company2} signed a cooperation agreement.",
    "No dividend was proposed by the Board of {company} for the fiscal year.",
    "{company} announced changes to its organizational structure.",
    "The number of {company} shares traded on the exchange totalled {amount} million.",
    "{company}'s Annual General Meeting approved all proposals of the Board.",
    "{company} reclassified its business segments effective from next quarter.",
    "{company} is headquartered in Helsinki, Finland.",
    "The report of {company} has been prepared in accordance with IFRS standards.",
    "{company}'s Board of Directors comprises {headcount} members.",
    "{company} operates in {count} countries worldwide.",
    "{company} confirmed the previously announced schedule for the rights issue.",
    "Trading in {company} shares was temporarily halted pending an announcement.",
    "{company} updated its corporate governance policies.",
    "{company} held a Capital Markets Day event for investors and analysts.",
    "The transaction between {company} and {company2} is subject to regulatory approval.",
    "{company} will release its interim report on {date}.",
    "{company} participated in an industry conference in Stockholm.",
    "{company}'s shares are listed on the Helsinki Stock Exchange.",
    "{company} completed the previously announced share buyback program.",
    "Shares of {company} were traded at EUR {price} at market close.",
    "{company} published its sustainability report for the fiscal year.",
    "{company}'s Nomination Board proposed the re-election of current directors.",
    "{company} entered into a lease agreement for new office premises.",
    "The extraordinary general meeting of {company} will take place on {date}.",
    "{company} issued EUR {amount} million in senior unsecured notes.",
    "{company} adopted a new share-based incentive plan for management.",
    "{company}'s financial statements are audited by KPMG.",
    "A total of {amount} million {company} shares changed hands during the week.",
    "{company} restated its segment reporting following organizational changes.",
    "{company} maintained its credit rating at the current level.",
    "{company} released details of its remuneration policy for executives.",
    "The merger of {company} and {company2} was registered with the Trade Register.",
    "{company} made no changes to its full-year outlook.",
    "{company}'s CEO will present at an investor event on {date}.",
    "Ownership in {company} by foreign investors stood at {pct}% at year-end.",
    "{company} reported its results in line with previous guidance.",
    "{company} announced the composition of its new executive team.",
    "{company} renewed its revolving credit facility for EUR {amount} million.",
    "The conversion period for {company}'s convertible bonds begins on {date}.",
    "{company} relocated its regional office to a new location.",
    "{company} reaffirmed its long-term strategic priorities.",
    "{company} disclosed related-party transactions in its annual report.",
    "Flagging notification: the holding in {company} crossed the {pct}% threshold.",
    "Changes in {company}'s management were announced effective immediately.",
    "{company} will host a webcast to discuss its quarterly results.",
    "{company} filed its annual accounts with the Finnish Financial Supervisory Authority.",
]

_FIRST_NAMES = [
    "Matti", "Pekka", "Juha", "Mikko", "Antti", "Timo", "Jari",
    "Eero", "Sari", "Anne", "Maria", "Kirsi", "Leena",
]

_LAST_NAMES = [
    "Virtanen", "Korhonen", "Mäkinen", "Nieminen", "Hämäläinen",
    "Laine", "Heikkinen", "Koskinen", "Järvinen", "Lehtonen",
]


class FinancialDataAugmentor:
    """
    Balances training data using external datasets, nlpaug, and templates.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        logger.info(f"FinancialDataAugmentor initialized (seed={seed})")

    # ------------------------------------------------------------------
    # 1. External data: additional PhraseBank sentences
    # ------------------------------------------------------------------

    def fetch_additional_phrasebank(
        self, existing_sentences: Set[str]
    ) -> pd.DataFrame:
        """
        Load sentences_50agree and return rows NOT in *existing_sentences*.

        Returns DataFrame with columns [sentence, label, label_name, source].
        """
        logger.info("Fetching additional PhraseBank sentences (50agree)...")

        label_to_int = {"negative": 0, "neutral": 1, "positive": 2}

        zip_path = hf_hub_download(
            repo_id="takala/financial_phrasebank",
            filename="data/FinancialPhraseBank-v1.0.zip",
            repo_type="dataset",
        )

        sentences, labels = [], []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("Sentences_50Agree.txt"):
                    with zf.open(name) as f:
                        content = f.read().decode("utf-8", errors="replace")
                        for line in content.split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.rsplit("@", 1)
                            if len(parts) == 2:
                                sent, lbl = parts[0].strip(), parts[1].strip().lower()
                                if lbl in label_to_int and sent not in existing_sentences:
                                    sentences.append(sent)
                                    labels.append(label_to_int[lbl])
                    break

        df = pd.DataFrame({"sentence": sentences, "label": labels})
        df["label_name"] = df["label"].map(LABEL_NAMES)
        df["source"] = "phrasebank_50agree"

        logger.info(
            f"Found {len(df)} new sentences from 50agree "
            f"(neg={sum(df.label==0)}, neu={sum(df.label==1)}, pos={sum(df.label==2)})"
        )
        return df

    # ------------------------------------------------------------------
    # 2. External data: Twitter Financial News Sentiment
    # ------------------------------------------------------------------

    def fetch_twitter_financial(self) -> pd.DataFrame:
        """
        Load twitter-financial-news-sentiment and map to 3-class labels.

        Mapping: Bearish -> 0, Bullish -> 2. Neutral rows are skipped
        because the neutral class is already the majority.

        Returns DataFrame with columns [sentence, label, label_name, source].
        """
        logger.info("Fetching Twitter Financial News Sentiment dataset...")
        try:
            from datasets import load_dataset

            ds = load_dataset(TWITTER_FINANCIAL_DATASET, "sentiment", split="train")
        except Exception as e:
            logger.warning(f"Could not load Twitter dataset: {e}")
            return pd.DataFrame(columns=["sentence", "label", "label_name", "source"])

        rows = []
        for row in ds:
            text = row.get("text", "")
            lbl = row.get("label", -1)
            if lbl == 0:  # Bearish -> negative
                rows.append({"sentence": text, "label": 0})
            elif lbl == 2:  # Bullish -> positive
                rows.append({"sentence": text, "label": 2})
            # Skip neutral (label==1) — already majority class

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return pd.DataFrame(columns=["sentence", "label", "label_name", "source"])

        df["label_name"] = df["label"].map(LABEL_NAMES)
        df["source"] = "twitter_financial"

        logger.info(
            f"Fetched {len(df)} Twitter samples "
            f"(neg={sum(df.label==0)}, pos={sum(df.label==2)})"
        )
        return df

    # ------------------------------------------------------------------
    # 3. nlpaug-based augmentation
    # ------------------------------------------------------------------

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        n_per_technique: Dict[int, int],
        techniques: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply nlpaug techniques to generate new samples.

        Args:
            df: Source dataframe with columns [sentence, label].
            n_per_technique: {label: n_samples_per_technique}.
            techniques: List of technique names to apply. Defaults to
                AUGMENTATION_TECHNIQUES from config.

        Returns:
            DataFrame of augmented rows [sentence, label, label_name, source].
        """
        if techniques is None:
            techniques = list(AUGMENTATION_TECHNIQUES)

        augmenters = self._build_augmenters(techniques)
        all_rows: List[dict] = []

        for label_id, n_needed in n_per_technique.items():
            if n_needed <= 0:
                continue

            pool = df[df["label"] == label_id]["sentence"].tolist()
            if not pool:
                continue

            for tech_name, aug in augmenters.items():
                sampled = self.rng.choices(pool, k=n_needed)
                for text in sampled:
                    try:
                        augmented = aug.augment(text)
                        if isinstance(augmented, list):
                            augmented = augmented[0]
                        all_rows.append(
                            {
                                "sentence": augmented,
                                "label": label_id,
                                "label_name": LABEL_NAMES[label_id],
                                "source": f"augmented_{tech_name}",
                            }
                        )
                    except Exception:
                        continue

        result = pd.DataFrame(all_rows)
        logger.info(f"nlpaug generated {len(result)} samples")
        return result

    def _build_augmenters(self, techniques: List[str]) -> dict:
        """Instantiate nlpaug augmenter objects."""
        import nlpaug.augmenter.word as naw

        augmenters = {}

        if "synonym" in techniques:
            augmenters["synonym"] = naw.SynonymAug(
                aug_src="wordnet", aug_max=AUGMENT_SYNONYM_TOP_K
            )
        if "insert" in techniques:
            augmenters["insert"] = naw.SynonymAug(
                aug_src="wordnet",
                aug_max=AUGMENT_INSERT_TOP_K,
                aug_p=0.1,
            )
        if "delete" in techniques:
            augmenters["delete"] = naw.RandomWordAug(
                action="delete",
                aug_min=1,
                aug_max=2,
                aug_p=0.1,
            )
        if "back_translate" in techniques:
            try:
                augmenters["back_translate"] = naw.BackTranslationAug(
                    from_model_name=BACK_TRANSLATION_SRC_MODEL,
                    to_model_name=BACK_TRANSLATION_TGT_MODEL,
                    device="cpu",
                    max_length=256,
                )
            except Exception as e:
                logger.warning(f"Back-translation unavailable: {e}")

        logger.info(f"Built augmenters: {list(augmenters.keys())}")
        return augmenters

    # ------------------------------------------------------------------
    # 4. Template generation
    # ------------------------------------------------------------------

    def generate_from_templates(self, n_per_class: Dict[int, int]) -> pd.DataFrame:
        """
        Generate sentences from templates with randomized slot-filling.

        Args:
            n_per_class: {label: n_samples_to_generate}.

        Returns:
            DataFrame with columns [sentence, label, label_name, source].
        """
        templates_by_class = {
            0: _NEGATIVE_TEMPLATES,
            1: _NEUTRAL_TEMPLATES,
            2: _POSITIVE_TEMPLATES,
        }

        rows: List[dict] = []
        for label_id, n_needed in n_per_class.items():
            if n_needed <= 0:
                continue
            templates = templates_by_class[label_id]
            for _ in range(n_needed):
                tpl = self.rng.choice(templates)
                filled = self._fill_template(tpl)
                rows.append(
                    {
                        "sentence": filled,
                        "label": label_id,
                        "label_name": LABEL_NAMES[label_id],
                        "source": "template_generated",
                    }
                )

        result = pd.DataFrame(rows)
        logger.info(f"Template generation produced {len(result)} samples")
        return result

    def _fill_template(self, template: str) -> str:
        """Fill a template string with random values."""
        companies = self.rng.sample(_COMPANIES, min(2, len(_COMPANIES)))
        first = self.rng.choice(_FIRST_NAMES)
        last = self.rng.choice(_LAST_NAMES)

        amount = self.rng.randint(5, 500)
        amount_prev = max(1, amount + self.rng.randint(-100, 100))
        months = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
        day = self.rng.randint(1, 28)
        month = self.rng.choice(months)
        year = self.rng.choice([2023, 2024, 2025])

        replacements = {
            "{company}": companies[0],
            "{company2}": companies[1] if len(companies) > 1 else companies[0],
            "{metric}": self.rng.choice(_METRICS),
            "{pct}": str(self.rng.randint(2, 45)),
            "{pct_small}": str(self.rng.randint(1, 15)),
            "{amount}": str(amount),
            "{amount_prev}": str(amount_prev),
            "{div}": f"{self.rng.uniform(0.10, 3.50):.2f}",
            "{headcount}": str(self.rng.randint(3, 12)),
            "{count}": str(self.rng.randint(5, 60)),
            "{price}": f"{self.rng.uniform(2.0, 80.0):.2f}",
            "{name}": f"{first} {last}",
            "{date}": f"{day} {month} {year}",
        }

        result = template
        for key, val in replacements.items():
            result = result.replace(key, val)
        return result

    # ------------------------------------------------------------------
    # 5. Orchestrator
    # ------------------------------------------------------------------

    def balance_training_set(
        self,
        train_df: pd.DataFrame,
        target_per_class: Optional[int] = None,
        techniques: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Balance the training set to have *target_per_class* samples per class.

        Strategy (priority order):
        1. External PhraseBank (50agree minus 75agree overlap)
        2. Twitter Financial News Sentiment (neg & pos only)
        3. nlpaug augmentation
        4. Template generation (capped at TEMPLATE_MAX_RATIO of deficit)

        Args:
            train_df: Original training DataFrame (columns: sentence, label).
            target_per_class: Target count per class. Defaults to majority
                class count.
            techniques: nlpaug techniques. Defaults to config list.

        Returns:
            Balanced DataFrame with added ``source`` column.
        """
        # Ensure required columns
        if "sentence" not in train_df.columns:
            if "text" in train_df.columns:
                train_df = train_df.rename(columns={"text": "sentence"})
            else:
                raise ValueError("DataFrame must have 'sentence' or 'text' column")

        # Add source tag to originals
        original = train_df.copy()
        original["source"] = "original"
        if "label_name" not in original.columns:
            original["label_name"] = original["label"].map(LABEL_NAMES)

        counts = original["label"].value_counts().to_dict()
        if target_per_class is None:
            target_per_class = max(counts.values())

        deficit = {lbl: target_per_class - counts.get(lbl, 0) for lbl in [0, 1, 2]}
        logger.info(f"Target per class: {target_per_class}")
        logger.info(f"Initial deficit: {deficit}")

        parts = [original]
        existing_sentences = set(original["sentence"].tolist())

        # --- Source 1: Additional PhraseBank ---
        try:
            pb_extra = self.fetch_additional_phrasebank(existing_sentences)
            if len(pb_extra) > 0:
                selected = self._select_up_to(pb_extra, deficit)
                if len(selected) > 0:
                    parts.append(selected)
                    existing_sentences.update(selected["sentence"].tolist())
                    self._update_deficit(deficit, selected)
                    logger.info(f"After PhraseBank: deficit={deficit}")
        except Exception as e:
            logger.warning(f"PhraseBank fetch failed: {e}")

        # --- Source 2: Twitter Financial ---
        try:
            twitter = self.fetch_twitter_financial()
            if len(twitter) > 0:
                # Deduplicate against existing
                twitter = twitter[~twitter["sentence"].isin(existing_sentences)]
                selected = self._select_up_to(twitter, deficit)
                if len(selected) > 0:
                    parts.append(selected)
                    existing_sentences.update(selected["sentence"].tolist())
                    self._update_deficit(deficit, selected)
                    logger.info(f"After Twitter: deficit={deficit}")
        except Exception as e:
            logger.warning(f"Twitter fetch failed: {e}")

        # --- Source 3: nlpaug augmentation ---
        remaining = {k: v for k, v in deficit.items() if v > 0}
        if remaining:
            # Reserve template budget
            template_budget = {
                lbl: int(target_per_class * TEMPLATE_MAX_RATIO)
                for lbl in remaining
            }
            aug_budget = {
                lbl: max(0, remaining[lbl] - template_budget.get(lbl, 0))
                for lbl in remaining
            }

            if any(v > 0 for v in aug_budget.values()):
                n_techniques = len(techniques or AUGMENTATION_TECHNIQUES)
                n_per_technique = {
                    lbl: max(1, n // n_techniques) if n > 0 else 0
                    for lbl, n in aug_budget.items()
                }
                try:
                    aug_df = self.augment_dataframe(
                        original, n_per_technique, techniques
                    )
                    if len(aug_df) > 0:
                        selected = self._select_up_to(aug_df, deficit)
                        if len(selected) > 0:
                            parts.append(selected)
                            self._update_deficit(deficit, selected)
                            logger.info(f"After nlpaug: deficit={deficit}")
                except Exception as e:
                    logger.warning(f"nlpaug augmentation failed: {e}")

        # --- Source 4: Templates (fill remaining) ---
        remaining = {k: v for k, v in deficit.items() if v > 0}
        if remaining:
            template_df = self.generate_from_templates(remaining)
            if len(template_df) > 0:
                selected = self._select_up_to(template_df, deficit)
                if len(selected) > 0:
                    parts.append(selected)
                    self._update_deficit(deficit, selected)
                    logger.info(f"After templates: deficit={deficit}")

        balanced = pd.concat(parts, ignore_index=True)

        # Keep only the columns we need
        balanced = balanced[["sentence", "label", "label_name", "source"]]

        final_counts = balanced["label"].value_counts().sort_index()
        logger.info(f"Final distribution:\n{final_counts}")
        logger.info(f"Sources:\n{balanced['source'].value_counts()}")

        return balanced

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------

    def save_augmented_dataset(
        self,
        df: pd.DataFrame,
        path: Optional[Path] = None,
    ) -> Path:
        """Save the balanced dataset to CSV."""
        if path is None:
            path = AUGMENTED_TRAIN_CSV
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved augmented dataset ({len(df)} rows) to {path}")
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_up_to(df: pd.DataFrame, deficit: Dict[int, int]) -> pd.DataFrame:
        """Select up to *deficit[label]* rows per label from *df*."""
        parts = []
        for label_id, n_need in deficit.items():
            if n_need <= 0:
                continue
            subset = df[df["label"] == label_id]
            if len(subset) > n_need:
                subset = subset.sample(n=n_need, random_state=42)
            parts.append(subset)
        if not parts:
            return pd.DataFrame(columns=df.columns)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _update_deficit(deficit: Dict[int, int], selected: pd.DataFrame) -> None:
        """Subtract selected counts from deficit in-place."""
        added = selected["label"].value_counts().to_dict()
        for lbl, cnt in added.items():
            deficit[lbl] = max(0, deficit.get(lbl, 0) - cnt)
