"""
scorer.py — Unified Fraud Risk Scorer (0–100)
==============================================
This is the central module that combines rule-based signals and ML
classifier probabilities into a single, interpretable risk score
on a 0–100 scale.

SCORING ARCHITECTURE:
┌─────────────────────┐
│   Input Message     │
│   (text + URL)      │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │ Rule Engine │  → Rule Score (0-100, based on penalty weights)
    └─────────────┘
           │
    ┌──────▼──────┐
    │SMS Classifier│ → ML SMS Score (0-100, based on spam probability)
    └─────────────┘  [only if ML models are loaded]
           │
    ┌──────▼──────┐
    │URL Classifier│ → ML URL Score (0-100, based on malicious URL prob)
    └─────────────┘  [only if URLs are detected AND ML model loaded]
           │
    ┌──────▼─────────────────────────────────────────────────┐
    │  Weighted Average → Final Risk Score (0–100)           │
    │  Rule (40%) + SMS ML (35%) + URL ML (25%)              │
    └────────────────────────────────────────────────────────┘

SCORE INTERPRETATION:
  0–25   → Low Risk (Safe)
  26–50  → Moderate Risk (Suspicious)
  51–75  → High Risk (Likely Fraud)
  76–100 → Critical Risk (Phishing / Scam)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .rule_engine import run_rule_engine, RuleMatch
from .url_analyzer import extract_url_features

logger = logging.getLogger(__name__)

# Maximum raw rule penalty used to normalise rule score to 0-100.
# Calibrated so that a message hitting ALL rules would approach 100.
MAX_RULE_PENALTY = 150

# Weighted contribution of each scoring component to the final score
WEIGHT_RULES = 0.40      # 40% — rule engine (always available, no model needed)
WEIGHT_SMS_ML = 0.35     # 35% — SMS classifier (requires trained model)
WEIGHT_URL_ML = 0.25     # 25% — URL classifier (requires trained model + URL in message)

# Regex to detect URLs in the input message
URL_DETECTION_REGEX = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how the final risk score was computed."""
    rule_score: float = 0.0         # Normalised rule engine score (0-100)
    sms_ml_score: float = 0.0       # SMS classifier contribution (0-100)
    url_ml_score: float = 0.0       # URL classifier contribution (0-100)
    final_score: float = 0.0        # Weighted combined score (0-100)
    risk_label: str = "Unknown"     # Human-readable risk level
    triggered_rules: List[str] = field(default_factory=list)
    rule_matches: List[RuleMatch] = field(default_factory=list)
    detected_urls: List[str] = field(default_factory=list)
    ml_available: bool = False      # Whether ML models were used
    warnings: List[str] = field(default_factory=list)


def get_risk_label(score: float) -> str:
    """
    Convert a numeric score (0-100) to a human-readable risk label.

    Args:
        score: Final combined risk score.

    Returns:
        One of: 'Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk'.
    """
    if score <= 25:
        return "Low Risk [SAFE]"
    elif score <= 50:
        return "Moderate Risk [WARNING]"
    elif score <= 75:
        return "High Risk [DANGER]"
    else:
        return "Critical Risk [PHISHING]"


# ---------------------------------------------------------------------------
# Main Scorer Class
# ---------------------------------------------------------------------------

class FraudScorer:
    """
    Primary public interface for the CyberTrack fraud detection system.

    Combines rule-based and ML-based scoring into a unified 0-100 risk score.

    Usage:
        # Rule-only mode (no dataset or training needed):
        scorer = FraudScorer()
        result = scorer.score("Claim your prize now! http://win-rs.xyz")

        # Full ML mode (after running train.py):
        scorer = FraudScorer(use_ml=True)
        result = scorer.score("Claim your prize now! http://win-rs.xyz")

        print(result.final_score)      # e.g. 84.7
        print(result.risk_label)       # "Critical Risk 🚨"
        print(result.triggered_rules)  # ["PRIZE_LOTTERY_BAIT", "SUSPICIOUS_TLD", ...]
    """

    def __init__(
        self,
        use_ml: bool = True,
        sms_model_path: Optional[str] = None,
        url_model_path: Optional[str] = None,
    ):
        """
        Initialise the fraud scorer, optionally loading ML models.

        Args:
            use_ml:          If True, attempt to load both ML classifiers.
                             If models are not found, falls back to rules-only.
            sms_model_path:  Custom path to the SMS classifier .pkl file.
            url_model_path:  Custom path to the URL classifier .pkl file.
        """
        self._sms_clf = None
        self._url_clf = None
        self._ml_available = False

        if use_ml:
            self._load_models(sms_model_path, url_model_path)

    def _load_models(
        self,
        sms_model_path: Optional[str],
        url_model_path: Optional[str],
    ) -> None:
        """
        Attempt to load ML classifiers. Gracefully falls back to rules-only
        mode if model files are not found (e.g. before training).

        Args:
            sms_model_path: Optional custom path for SMS model.
            url_model_path: Optional custom path for URL model.
        """
        from .sms_classifier import SMSClassifier
        from .url_classifier import URLClassifier

        sms_loaded = False
        url_loaded = False

        # -- Load SMS Classifier --
        try:
            kwargs = {"model_path": sms_model_path} if sms_model_path else {}
            self._sms_clf = SMSClassifier(**kwargs)
            sms_loaded = True
            logger.info("SMS classifier loaded successfully.")
        except FileNotFoundError:
            logger.warning(
                "SMS model not found. Run train.py to train it. "
                "Falling back to rules-only for SMS scoring."
            )

        # -- Load URL Classifier --
        try:
            kwargs = {"model_path": url_model_path} if url_model_path else {}
            self._url_clf = URLClassifier(**kwargs)
            url_loaded = True
            logger.info("URL classifier loaded successfully.")
        except FileNotFoundError:
            logger.warning(
                "URL model not found. Run train.py to train it. "
                "Falling back to rules-only for URL scoring."
            )

        self._ml_available = sms_loaded or url_loaded

    # -----------------------------------------------------------------------
    # Score Computation
    # -----------------------------------------------------------------------

    def _compute_rule_score(self, text: str, sender: str) -> tuple:
        """
        Run the rule engine and normalise the penalty total to 0-100.

        The raw penalty total is clamped to MAX_RULE_PENALTY before
        normalisation so that a message hitting many rules caps at 100.

        Args:
            text:   The SMS body.
            sender: Optional sender ID string.

        Returns:
            Tuple of (normalised_score, rule_engine_result).
        """
        result = run_rule_engine(text, sender)
        raw_penalty = min(result.total_penalty, MAX_RULE_PENALTY)
        normalised = (raw_penalty / MAX_RULE_PENALTY) * 100
        return round(normalised, 2), result

    def _compute_sms_ml_score(self, text: str) -> float:
        """
        Get the SMS spam probability from the ML classifier and scale to 0-100.

        Args:
            text: Raw SMS message string.

        Returns:
            Score in [0, 100]. Returns 0.0 if model is unavailable.
        """
        if self._sms_clf is None:
            return 0.0
        spam_prob = self._sms_clf.predict_proba(text)  # Returns 0.0–1.0
        return round(spam_prob * 100, 2)

    def _compute_url_ml_score(self, urls: List[str]) -> float:
        """
        Score each detected URL with the URL classifier and return
        the maximum malicious probability across all URLs in the message.

        Using the maximum (rather than average) means one highly
        suspicious URL is enough to raise the overall URL score.

        Args:
            urls: List of URL strings extracted from the message.

        Returns:
            Score in [0, 100]. Returns 0.0 if no URLs or model unavailable.
        """
        if self._url_clf is None or not urls:
            return 0.0

        scores = []
        for url in urls:
            try:
                prob = self._url_clf.predict_proba(url)
                scores.append(prob * 100)
            except Exception as e:
                logger.debug(f"URL scoring failed for '{url}': {e}")

        return round(max(scores), 2) if scores else 0.0

    def _combine_scores(
        self,
        rule_score: float,
        sms_ml_score: float,
        url_ml_score: float,
        has_urls: bool,
    ) -> float:
        """
        Combine individual component scores into one final risk score
        using a weighted average.

        When ML models are unavailable, weights are redistributed to
        give 100% weight to the rule engine.

        When no URLs are detected, the URL ML weight is redistributed
        proportionally between rules and SMS ML.

        Args:
            rule_score:    Normalised rule engine score (0-100).
            sms_ml_score:  SMS classifier score (0-100).
            url_ml_score:  URL classifier score (0-100).
            has_urls:      Whether URLs were detected in the message.

        Returns:
            Combined score in [0, 100].
        """
        if not self._ml_available:
            # No ML models — rules carry full weight
            return rule_score

        if not has_urls:
            # No URLs — redistribute URL weight between rules and SMS ML
            # New weights: rules=0.53, sms_ml=0.47 (proportional to original)
            w_rules = WEIGHT_RULES / (WEIGHT_RULES + WEIGHT_SMS_ML)
            w_sms = WEIGHT_SMS_ML / (WEIGHT_RULES + WEIGHT_SMS_ML)
            combined = (rule_score * w_rules) + (sms_ml_score * w_sms)
        else:
            # Full weighted combination
            combined = (
                (rule_score   * WEIGHT_RULES) +
                (sms_ml_score * WEIGHT_SMS_ML) +
                (url_ml_score * WEIGHT_URL_ML)
            )

        # Clamp to valid range [0, 100]
        return round(min(max(combined, 0.0), 100.0), 2)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def score(self, text: str, sender: str = "") -> ScoreBreakdown:
        """
        Main entry point. Score a message for fraud risk.

        Runs the full pipeline: rule engine → SMS ML → URL ML → combine.

        Args:
            text:   The raw SMS message or URL string to analyse.
            sender: Optional sender ID (e.g. 'VM-HDFCBK' or '+4479...).
                    Enables sender anomaly checks.

        Returns:
            ScoreBreakdown object with final_score (0-100) and full explanation.

        Example:
            >>> scorer = FraudScorer()
            >>> result = scorer.score(
            ...     "URGENT! Your SBI account is blocked. "
            ...     "Verify KYC now: http://sbi-kyc-verify.xyz/login",
            ...     sender="+447911123456"
            ... )
            >>> print(f"Risk Score: {result.final_score}/100 — {result.risk_label}")
            Risk Score: 91.4/100 — Critical Risk 🚨
        """
        breakdown = ScoreBreakdown()

        # -- Step 1: Detect URLs in the message --
        breakdown.detected_urls = URL_DETECTION_REGEX.findall(text)
        has_urls = len(breakdown.detected_urls) > 0

        # -- Step 2: Rule Engine --
        rule_score, rule_result = self._compute_rule_score(text, sender)
        breakdown.rule_score = rule_score
        breakdown.rule_matches = rule_result.matches
        breakdown.triggered_rules = rule_result.triggered_rules

        # -- Step 3: SMS ML Classifier --
        sms_ml_score = 0.0
        if self._sms_clf is not None:
            sms_ml_score = self._compute_sms_ml_score(text)
        else:
            breakdown.warnings.append(
                "SMS ML model not loaded — SMS score uses rules only."
            )
        breakdown.sms_ml_score = sms_ml_score

        # -- Step 4: URL ML Classifier --
        url_ml_score = 0.0
        if has_urls and self._url_clf is not None:
            url_ml_score = self._compute_url_ml_score(breakdown.detected_urls)
        elif has_urls and self._url_clf is None:
            breakdown.warnings.append(
                "URL ML model not loaded — URL score uses rules only."
            )
        breakdown.url_ml_score = url_ml_score

        # -- Step 5: Combine into final score --
        breakdown.ml_available = self._ml_available
        breakdown.final_score = self._combine_scores(
            rule_score, sms_ml_score, url_ml_score, has_urls
        )
        breakdown.risk_label = get_risk_label(breakdown.final_score)

        return breakdown

    def explain(self, breakdown: ScoreBreakdown) -> str:
        """
        Generate a human-readable explanation report for a ScoreBreakdown.

        Useful for displaying results in a UI or printing in a project demo.

        Args:
            breakdown: A ScoreBreakdown object returned by score().

        Returns:
            Formatted multi-line string report.
        """
        lines = [
            "=" * 60,
            f"  CYBERTRACK FRAUD RISK REPORT",
            "=" * 60,
            f"  Final Risk Score : {breakdown.final_score:.1f} / 100",
            f"  Risk Level       : {breakdown.risk_label}",
            "-" * 60,
            "  SCORE BREAKDOWN:",
            f"    Rule Engine Score  : {breakdown.rule_score:.1f}/100  (weight: 40%)",
            f"    SMS ML Score       : {breakdown.sms_ml_score:.1f}/100  (weight: 35%)",
            f"    URL ML Score       : {breakdown.url_ml_score:.1f}/100  (weight: 25%)",
            f"    ML Models Active   : {'Yes' if breakdown.ml_available else 'No (rules only)'}",
        ]

        if breakdown.detected_urls:
            lines.append("-" * 60)
            lines.append(f"  DETECTED URLs ({len(breakdown.detected_urls)}):")
            for url in breakdown.detected_urls:
                lines.append(f"    • {url}")

        if breakdown.rule_matches:
            lines.append("-" * 60)
            lines.append(f"  TRIGGERED RULES ({len(breakdown.rule_matches)}):")
            for match in breakdown.rule_matches:
                lines.append(
                    f"    [{match.penalty:+3d} pts] {match.rule_name}"
                )
                lines.append(f"              → {match.description}")
                if match.matched_text:
                    lines.append(f"              Matched: \"{match.matched_text}\"")

        if breakdown.warnings:
            lines.append("-" * 60)
            lines.append("  WARNINGS:")
            for w in breakdown.warnings:
                lines.append(f"    ⚠ {w}")

        lines.append("=" * 60)
        return "\n".join(lines)
