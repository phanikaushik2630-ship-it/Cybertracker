"""
sms_classifier.py — ML-Based SMS Text Classifier
==================================================
Trains and runs a Multinomial Naive Bayes classifier on the
UCI/Kaggle SMS Spam Collection Dataset to classify message text
as spam (fraudulent) or ham (legitimate).

WHY NAIVE BAYES?
  - Extremely fast training and inference (< 1 second on CPU)
  - Works well on bag-of-words / TF-IDF text features
  - Probabilistic output gives us a spam *probability* (0.0–1.0)
    that can be scaled into a partial risk score
  - Widely used as a text classification baseline in academic work

DATASET FORMAT (SMS Spam Collection):
  CSV with two columns:
    - 'label': 'ham' or 'spam'
    - 'message': raw SMS text
  Available at: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

ALTERNATIVE: Logistic Regression (also supported here) tends to give
  slightly better precision on imbalanced data.
"""

import os
import pickle
import logging
from typing import Tuple, Optional

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default path where the trained model is saved / loaded from
DEFAULT_SMS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "sms_classifier.pkl"
)


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_sms_classifier(
    csv_path: str,
    label_col: str = "label",
    text_col: str = "message",
    model_type: str = "naive_bayes",
    save_path: str = DEFAULT_SMS_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a TF-IDF + Naive Bayes (or Logistic Regression) pipeline on
    the SMS Spam Collection dataset and save the model to disk.

    PIPELINE STRUCTURE:
        Raw SMS text
            ↓
        TF-IDF Vectorizer
          (converts text to weighted word-frequency matrix)
            ↓
        Multinomial Naive Bayes / Logistic Regression
          (predicts spam probability per message)

    Args:
        csv_path:     Path to the SMS dataset CSV file.
        label_col:    Column name containing 'ham'/'spam' labels.
        text_col:     Column name containing raw message text.
        model_type:   'naive_bayes' or 'logistic_regression'.
        save_path:    Where to save the trained .pkl model file.
        test_size:    Fraction of data held out for evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Trained scikit-learn Pipeline object.

    Example:
        >>> pipeline = train_sms_classifier("data/spam.csv")
        >>> print(pipeline.predict(["Congratulations! You won Rs 50000!"]))
        ['spam']
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for training. Run: pip install pandas")

    logger.info(f"Loading SMS dataset from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Handle datasets that include extra unnamed columns (common in this dataset)
    df = df[[label_col, text_col]].dropna()

    # Map labels to binary integers: spam → 1, ham → 0
    df["label_int"] = df[label_col].map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label_int"])  # Drop any rows with unexpected labels

    X = df[text_col].values
    y = df["label_int"].values.astype(int)

    logger.info(f"Dataset loaded: {len(X)} samples "
                f"({y.sum()} spam, {len(y) - y.sum()} ham)")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build the pipeline: TF-IDF → Classifier
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"  # Handles class imbalance (more ham than spam)
        )
        logger.info("Using: Logistic Regression classifier")
    else:
        # Multinomial Naive Bayes — default and recommended for text
        classifier = MultinomialNB(alpha=0.1)  # alpha = Laplace smoothing
        logger.info("Using: Multinomial Naive Bayes classifier")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),     # Use unigrams + bigrams for richer context
            max_features=10_000,    # Cap vocabulary size for lightweight inference
            sublinear_tf=True,      # Apply log normalization to term frequencies
            stop_words="english",   # Remove common English words (the, is, at...)
        )),
        ("classifier", classifier),
    ])

    # Train the pipeline
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Save the trained pipeline to disk for later inference
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to: {save_path}")

    return pipeline


# ---------------------------------------------------------------------------
# Inference Class
# ---------------------------------------------------------------------------

class SMSClassifier:
    """
    Wrapper around the trained SMS spam pipeline.

    Loads the model once and exposes a predict() method
    that returns a fraud probability (0.0 = safe, 1.0 = spam/fraud).
    """

    def __init__(self, model_path: str = DEFAULT_SMS_MODEL_PATH):
        """
        Load a previously trained pipeline from disk.

        Args:
            model_path: Path to the .pkl file saved by train_sms_classifier().

        Raises:
            FileNotFoundError: If no trained model exists at the given path.
                               Run train_sms_classifier() first.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained SMS model found at '{model_path}'.\n"
                f"Please run train.py first to train and save the model."
            )

        with open(model_path, "rb") as f:
            self._pipeline: Pipeline = pickle.load(f)

        logger.info(f"SMS classifier loaded from: {model_path}")

    def predict_proba(self, text: str) -> float:
        """
        Return the probability that a message is spam/fraudulent.

        Internally uses the pipeline's predict_proba() method which
        gives the softmax output of the final classifier layer.

        Args:
            text: Raw SMS message string.

        Returns:
            Float in [0.0, 1.0]. Higher = more likely spam.

        Example:
            >>> clf = SMSClassifier()
            >>> clf.predict_proba("Your account has been blocked. Click now!")
            0.97
        """
        # predict_proba returns [[P(ham), P(spam)]] — we want P(spam)
        proba = self._pipeline.predict_proba([text])[0]
        spam_probability = float(proba[1])  # index 1 = spam class
        return spam_probability

    def predict_label(self, text: str) -> str:
        """
        Predict class label for a message: 'spam' or 'ham'.

        Args:
            text: Raw SMS message string.

        Returns:
            String: 'spam' or 'ham'
        """
        prediction = self._pipeline.predict([text])[0]
        return "spam" if prediction == 1 else "ham"
