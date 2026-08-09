"""
fraud_detector — CyberTrack Phishing & Fraud Detection Package
==============================================================
Entry point for the fraud detection pipeline.

Usage:
    from fraud_detector import FraudScorer
    scorer = FraudScorer()
    result = scorer.score("Congrats! You won Rs.50,000. Click http://prize-claim.xyz now!")
    print(result)
"""

from .scorer import FraudScorer

__all__ = ["FraudScorer"]
__version__ = "1.0.0"
