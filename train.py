"""
train.py — Model Training Script
==================================
Run this script once after downloading the required datasets to train
and save both ML classifiers (SMS + URL) to the models/ directory.

REQUIRED DATASETS (download from Kaggle):
  1. SMS Spam Collection Dataset
     URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
     File: spam.csv
     Columns: v1 (label: ham/spam), v2 (message text)

  2. Malicious URLs Dataset
     URL: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
     File: malicious_phish.csv
     Columns: url, type (benign/phishing/malware/defacement)

HOW TO RUN:
  python train.py --sms_data data/spam.csv --url_data data/malicious_phish.csv

AFTER RUNNING:
  Models are saved to:
    models/sms_classifier.pkl
    models/url_classifier.pkl

  Then use FraudScorer(use_ml=True) in your application.
"""

import argparse
import logging
import os
import sys

# Set up logging so training progress is visible in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train CyberTrack fraud detection classifiers"
    )
    parser.add_argument(
        "--sms_data",
        type=str,
        default="data/spam.csv",
        help="Path to the SMS Spam Collection CSV (default: data/spam.csv)"
    )
    parser.add_argument(
        "--url_data",
        type=str,
        default="data/malicious_phish.csv",
        help="Path to the Malicious URLs CSV (default: data/malicious_phish.csv)"
    )
    parser.add_argument(
        "--sms_label_col",
        type=str,
        default="v1",       # Kaggle version uses v1/v2 column names
        help="Column name for SMS labels (default: v1)"
    )
    parser.add_argument(
        "--sms_text_col",
        type=str,
        default="v2",
        help="Column name for SMS text (default: v2)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="naive_bayes",
        choices=["naive_bayes", "logistic_regression"],
        help="SMS classifier type (default: naive_bayes)"
    )
    return parser.parse_args()


def train_sms(args: argparse.Namespace) -> bool:
    """Train and save the SMS classifier. Returns True on success."""
    from fraud_detector.sms_classifier import train_sms_classifier

    if not os.path.exists(args.sms_data):
        logger.error(
            f"SMS dataset not found: {args.sms_data}\n"
            f"Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset"
        )
        return False

    logger.info("━" * 50)
    logger.info("TRAINING SMS CLASSIFIER")
    logger.info("━" * 50)

    try:
        train_sms_classifier(
            csv_path=args.sms_data,
            label_col=args.sms_label_col,
            text_col=args.sms_text_col,
            model_type=args.model_type,
        )
        logger.info("✓ SMS classifier trained and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"SMS training failed: {e}")
        return False


def train_url(args: argparse.Namespace) -> bool:
    """Train and save the URL classifier. Returns True on success."""
    from fraud_detector.url_classifier import train_url_classifier

    if not os.path.exists(args.url_data):
        logger.error(
            f"URL dataset not found: {args.url_data}\n"
            f"Download from: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset"
        )
        return False

    logger.info("━" * 50)
    logger.info("TRAINING URL CLASSIFIER")
    logger.info("━" * 50)

    try:
        train_url_classifier(csv_path=args.url_data)
        logger.info("✓ URL classifier trained and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"URL training failed: {e}")
        return False


def main() -> None:
    args = parse_args()

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    sms_ok = train_sms(args)
    url_ok = train_url(args)

    logger.info("━" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("━" * 50)
    logger.info(f"  SMS Classifier : {'✓ Ready' if sms_ok else '✗ Failed / Skipped'}")
    logger.info(f"  URL Classifier : {'✓ Ready' if url_ok else '✗ Failed / Skipped'}")

    if sms_ok or url_ok:
        logger.info("\nYou can now run example_usage.py with use_ml=True.")
    else:
        logger.warning("\nNo models were trained. Check dataset paths and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
