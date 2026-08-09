"""
url_classifier.py — ML-Based Malicious URL Classifier
=======================================================
Trains and runs a Logistic Regression classifier on extracted
URL features (from url_analyzer.py) to predict whether a URL
is malicious (phishing, malware, defacement) or benign.

WHY LOGISTIC REGRESSION FOR URLS?
  - Works well on the numerical feature vectors we engineered
  - Outputs calibrated probabilities (unlike SVM)
  - L2 regularization prevents overfitting on small datasets
  - Fast inference: suitable for real-time SMS scanning

DATASET FORMAT (Malicious URLs Dataset — Kaggle):
  CSV with at least two columns:
    - 'url':  the raw URL string
    - 'type': 'benign' | 'phishing' | 'malware' | 'defacement'
  Available at: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
  (Also works with: "Malicious and Benign Websites" dataset)
"""

import os
import pickle
import logging
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .url_analyzer import extract_url_features, URL_FEATURE_NAMES

logger = logging.getLogger(__name__)

DEFAULT_URL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "url_classifier.pkl"
)


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_url_classifier(
    csv_path: str,
    url_col: str = "url",
    label_col: str = "type",
    save_path: str = DEFAULT_URL_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a Logistic Regression classifier on URL features extracted
    from a labelled malicious URL dataset.

    PIPELINE STRUCTURE:
        Raw URL string
            ↓
        Feature Engineering (url_analyzer.extract_url_features)
          [17 numerical features per URL]
            ↓
        StandardScaler
          (normalises features to zero mean, unit variance)
            ↓
        Logistic Regression
          (predicts malicious probability)

    Args:
        csv_path:     Path to the Kaggle URL dataset CSV.
        url_col:      Column name containing raw URL strings.
        label_col:    Column name containing type labels.
        save_path:    Where to save the trained model .pkl file.
        test_size:    Fraction held out for evaluation.
        random_state: Seed for reproducibility.

    Returns:
        Trained scikit-learn Pipeline.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for training. Run: pip install pandas")

    logger.info(f"Loading URL dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=[url_col, label_col])

    # Binarise labels: 'benign' → 0, everything else → 1 (malicious)
    df["label_int"] = df[label_col].apply(lambda x: 0 if str(x).lower() == "benign" else 1)

    logger.info(f"Dataset loaded: {len(df)} URLs "
                f"({df['label_int'].sum()} malicious, "
                f"{len(df) - df['label_int'].sum()} benign)")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}")

    # Extract feature vectors for all URLs
    logger.info("Extracting URL features (this may take a moment)...")
    X = np.array([extract_url_features(url) for url in df[url_col].values])
    y = df["label_int"].values.astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build the pipeline: Scale → Classify
    pipeline = Pipeline([
        ("scaler", StandardScaler()),           # Normalise feature magnitudes
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,                              # Inverse regularisation strength
            class_weight="balanced",            # Handle imbalanced classes
            random_state=random_state,
            solver="lbfgs",
        )),
    ])

    logger.info("Training URL classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("\n" + classification_report(
        y_test, y_pred, target_names=["benign", "malicious"]
    ))

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"URL classifier saved to: {save_path}")

    return pipeline


# ---------------------------------------------------------------------------
# Inference Class
# ---------------------------------------------------------------------------

class URLClassifier:
    """
    Wrapper around the trained URL classification pipeline.

    Loads the model once and exposes predict_proba() for
    real-time URL risk scoring during inference.
    """

    def __init__(self, model_path: str = DEFAULT_URL_MODEL_PATH):
        """
        Load a previously trained URL classifier from disk.

        Args:
            model_path: Path to the .pkl file.

        Raises:
            FileNotFoundError: If model does not exist — run train.py first.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained URL model found at '{model_path}'.\n"
                f"Please run train.py first to train and save the model."
            )
        with open(model_path, "rb") as f:
            self._pipeline: Pipeline = pickle.load(f)
        logger.info(f"URL classifier loaded from: {model_path}")

    def predict_proba(self, url: str) -> float:
        """
        Return the probability that a URL is malicious.

        Args:
            url: A URL string (e.g. 'http://secure-sbi.xyz/login').

        Returns:
            Float in [0.0, 1.0]. Higher = more likely malicious.

        Example:
            >>> clf = URLClassifier()
            >>> clf.predict_proba("http://sbi.secure-login.xyz/verify")
            0.94
        """
        features = np.array([extract_url_features(url)])   # Shape: (1, 17)
        proba = self._pipeline.predict_proba(features)[0]
        malicious_probability = float(proba[1])  # index 1 = malicious class
        return malicious_probability
