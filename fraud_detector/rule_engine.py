"""
rule_engine.py — Rule-Based Risk Scoring
=========================================
This module implements deterministic, explainable checks based on:
  - Suspicious URL structural patterns (regex)
  - Urgency language detection
  - Payment / OTP phishing phrasing
  - India-specific scam trigger words (TRAI, Aadhaar, KYC, UPI)
  - Sender anomaly indicators

Each check returns a "hit" (True/False) and an associated penalty weight.
The total rule score is then normalised to a 0-100 range by the scorer.

WHY RULE-BASED?
  Rules are fast (<1 ms), require no training data, and are fully explainable
  — critical for a final year project demonstration.
"""

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RuleMatch:
    """Stores a single rule that fired during analysis."""
    rule_name: str          # Short identifier for the rule
    description: str        # Human-readable explanation (for report output)
    penalty: int            # How much this rule contributes to the risk score
    matched_text: str = ""  # The actual snippet that triggered the rule


@dataclass
class RuleEngineResult:
    """Aggregated output from the rule engine for one input."""
    matches: List[RuleMatch] = field(default_factory=list)

    @property
    def total_penalty(self) -> int:
        """Sum of all fired rule penalties (unbounded)."""
        return sum(m.penalty for m in self.matches)

    @property
    def triggered_rules(self) -> List[str]:
        return [m.rule_name for m in self.matches]


# ---------------------------------------------------------------------------
# URL Structural Pattern Rules
# ---------------------------------------------------------------------------

# Suspicious top-level domains frequently abused in phishing campaigns
SUSPICIOUS_TLDS = re.compile(
    r'\.(xyz|tk|ml|ga|cf|gq|pw|top|click|loan|gdn|win|bid|review|trade|party|date)(\b|/)',
    re.IGNORECASE
)

# Legitimate brand names appearing as a *subdomain* (not the actual domain)
# e.g. sbi.secure-login.xyz  →  "sbi" is the brand, but the real domain is "secure-login.xyz"
BRAND_IN_SUBDOMAIN = re.compile(
    r'(sbi|icici|hdfc|paytm|phonepe|gpay|bhim|irctc|uidai|npci|rbi|amazon|flipkart|'
    r'olx|myntra|zomato|swiggy)\.[a-z0-9\-]+\.(xyz|tk|ml|ga|cf|pw|top|com|in)',
    re.IGNORECASE
)

# IP address used directly instead of a domain name in a URL
IP_IN_URL = re.compile(
    r'https?://(\d{1,3}\.){3}\d{1,3}[/:]'
)

# URL shortener services that can hide the true destination
URL_SHORTENERS = re.compile(
    r'https?://(bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|rb\.gy|'
    r'cutt\.ly|short\.io|is\.gd|tiny\.cc)/',
    re.IGNORECASE
)

# Unusual special characters in a URL path — sign of encoding tricks
URL_SPECIAL_CHARS = re.compile(r'@|%40|//+(?!https?:)')

# Punycode / Internationalized Domain Names used for homograph attacks
# e.g. xn--pymt-8ra.com looks like paytm in certain scripts
PUNYCODE_DOMAIN = re.compile(r'https?://xn--', re.IGNORECASE)

# Excessive subdomains (depth ≥ 4) — legitimate sites rarely need this
EXCESSIVE_SUBDOMAINS = re.compile(
    r'https?://([a-z0-9\-]+\.){4,}[a-z]{2,}',
    re.IGNORECASE
)

# Very long URLs (> 100 chars) are a common phishing indicator
URL_TOO_LONG_THRESHOLD = 100


def check_url_patterns(text: str) -> List[RuleMatch]:
    """
    Run all URL-structural regex rules against the input text.

    Args:
        text: Raw SMS message or URL string.

    Returns:
        List of RuleMatch objects for every rule that fired.
    """
    matches: List[RuleMatch] = []

    # -- Extract all URLs from the message first --
    url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)
    urls = url_pattern.findall(text)

    for url in urls:
        if SUSPICIOUS_TLDS.search(url):
            matches.append(RuleMatch(
                rule_name="SUSPICIOUS_TLD",
                description="URL uses a high-risk TLD (.xyz, .tk, .pw, etc.)",
                penalty=20,
                matched_text=url
            ))

        if BRAND_IN_SUBDOMAIN.search(url):
            matches.append(RuleMatch(
                rule_name="BRAND_IN_SUBDOMAIN",
                description="Known brand name used as subdomain to impersonate a trusted site",
                penalty=25,
                matched_text=url
            ))

        if IP_IN_URL.search(url):
            matches.append(RuleMatch(
                rule_name="IP_IN_URL",
                description="URL uses a raw IP address instead of a domain name",
                penalty=30,
                matched_text=url
            ))

        if URL_SHORTENERS.search(url):
            matches.append(RuleMatch(
                rule_name="URL_SHORTENER",
                description="URL uses a shortening service that hides the real destination",
                penalty=15,
                matched_text=url
            ))

        if URL_SPECIAL_CHARS.search(url):
            matches.append(RuleMatch(
                rule_name="URL_SPECIAL_CHARS",
                description="URL contains suspicious characters (@, encoding tricks)",
                penalty=15,
                matched_text=url
            ))

        if PUNYCODE_DOMAIN.search(url):
            matches.append(RuleMatch(
                rule_name="PUNYCODE_HOMOGRAPH",
                description="URL uses Punycode (xn--) — possible homograph/lookalike attack",
                penalty=30,
                matched_text=url
            ))

        if EXCESSIVE_SUBDOMAINS.search(url):
            matches.append(RuleMatch(
                rule_name="EXCESSIVE_SUBDOMAINS",
                description="URL has 4+ subdomain levels — unusual for legitimate services",
                penalty=10,
                matched_text=url
            ))

        if len(url) > URL_TOO_LONG_THRESHOLD:
            matches.append(RuleMatch(
                rule_name="LONG_URL",
                description=f"URL is excessively long ({len(url)} chars)",
                penalty=10,
                matched_text=url[:60] + "..."
            ))

    return matches


# ---------------------------------------------------------------------------
# Text Keyword / Phrase Rules
# ---------------------------------------------------------------------------

# Urgency language patterns — pressure tactics to force quick action
URGENCY_PHRASES = re.compile(
    r'\b(act now|urgent|urgently|immediately|expires? in|last (chance|warning|reminder)|'
    r'within \d+ (hour|minute|day)|account (suspended|blocked|deactivated|restricted)|'
    r'limited time|do not ignore|your account will be|deadline|'
    r'security alert|unusual activity|suspicious (activity|login|transaction)|'
    r'restore (your )?access|restore access|access (has been|will be) (restricted|blocked|suspended)|'
    r'temporarily (restricted|blocked|suspended|limited)|'
    r'verify (your )?(account|identity|details?) (within|immediately|now)|'
    r'failure to (comply|verify|respond)|action (required|needed)|'
    r'account (will be|has been) (closed|terminated|restricted))\b',
    re.IGNORECASE
)

# Payment / money transfer related scam phrases
PAYMENT_PHRASES = re.compile(
    r'\b(send money|transfer (rs\.?|inr|₹)?\s*[\d,]+|pay now|upi (id|link|payment)|'
    r'wallet|recharge|cashback (credited|pending)|click to (pay|receive)|'
    r'prize money|lottery (won|winner)|lucky draw|redeem now|claim your (reward|prize|cash))\b',
    re.IGNORECASE
)

# OTP and credential harvesting phrases
OTP_HARVEST_PHRASES = re.compile(
    r'\b(share your otp|enter (your )?otp|verify (your )?(otp|pin|account|details?|information)|'
    r'do not share|otp is confidential|confirm (your )?(identity|account|details?)|'
    r'update (your )?kyc|kyc (pending|expired|verification)|'
    r'complete (your )?(verification|kyc)|account verification|'
    r'submit (your )?(aadhaar|pan|bank) (number|details|card)|'
    r'click (here|below|the link) to verify|tap to verify|'
    r're-?verify|re-?activate (your )?account)\b',
    re.IGNORECASE
)

# India-specific authority impersonation — very common in Indian phishing SMS
AUTHORITY_IMPERSONATION = re.compile(
    r'\b(trai|rbi|income tax (dept|department|notice)|it department|uidai|epfo|epf|'
    r'aadhaar (update|link|verify|expired)|pan card (link|verify|block)|'
    r'government of india|ministry of|sebi|irda|npci)\b',
    re.IGNORECASE
)

# Bank / financial account security threat — messages mimicking bank fraud alerts
# that pressure the user into "verifying" their account
ACCOUNT_SECURITY_THREAT = re.compile(
    r'\b(your (bank )?account|account (number|ending|no\.?)|'
    r'(bank|credit|debit) (account|card)|net.?banking|'
    r'transaction (has been|was) (blocked|flagged|declined)|'
    r'(login|sign.?in) attempt|unauthori[sz]ed (access|transaction|activity)|'
    r'account (security|protection)|security (breach|concern|measure)|'
    r'unusual (login|access|transaction)|suspicious (login|access))\b',
    re.IGNORECASE
)

# Prize / lottery bait — classic advance-fee scam trigger
PRIZE_BAIT = re.compile(
    r'\b(congratulations|you (have won|are selected|are a winner)|'
    r'selected as (the )?lucky|prize of rs|won (rs\.?|inr|₹)?\s*[\d,]+|'
    r'gift (card|voucher)|spin (and win|to win)|scratch (and win|card))\b',
    re.IGNORECASE
)

# Excessive caps and exclamation marks — hallmarks of scam messages
EXCESSIVE_CAPS_PATTERN = re.compile(r'\b[A-Z]{4,}\b')
EXCESSIVE_EXCLAMATION = re.compile(r'!{2,}')


def check_text_keywords(text: str) -> List[RuleMatch]:
    """
    Run keyword and phrase matching rules on the message body.

    Args:
        text: Raw SMS or message string.

    Returns:
        List of RuleMatch objects for every rule that fired.
    """
    matches: List[RuleMatch] = []

    if URGENCY_PHRASES.search(text):
        snippet = URGENCY_PHRASES.search(text).group()
        matches.append(RuleMatch(
            rule_name="URGENCY_LANGUAGE",
            description="Message uses urgency/pressure tactics to force quick action",
            penalty=18,
            matched_text=snippet
        ))

    if PAYMENT_PHRASES.search(text):
        snippet = PAYMENT_PHRASES.search(text).group()
        matches.append(RuleMatch(
            rule_name="PAYMENT_REQUEST",
            description="Message contains payment or money-transfer phrasing",
            penalty=20,
            matched_text=snippet
        ))

    if OTP_HARVEST_PHRASES.search(text):
        snippet = OTP_HARVEST_PHRASES.search(text).group()
        matches.append(RuleMatch(
            rule_name="OTP_CREDENTIAL_HARVEST",
            description="Message asks for OTP, KYC details, or Aadhaar/PAN numbers",
            penalty=30,
            matched_text=snippet
        ))

    if AUTHORITY_IMPERSONATION.search(text):
        snippet = AUTHORITY_IMPERSONATION.search(text).group()
        matches.append(RuleMatch(
            rule_name="AUTHORITY_IMPERSONATION",
            description="Message impersonates TRAI, RBI, UIDAI, or other Indian authorities",
            penalty=25,
            matched_text=snippet
        ))

    if ACCOUNT_SECURITY_THREAT.search(text):
        snippet = ACCOUNT_SECURITY_THREAT.search(text).group()
        matches.append(RuleMatch(
            rule_name="ACCOUNT_SECURITY_THREAT",
            description="Message uses bank/account security threat language to trigger panic",
            penalty=20,
            matched_text=snippet
        ))

    if PRIZE_BAIT.search(text):
        snippet = PRIZE_BAIT.search(text).group()
        matches.append(RuleMatch(
            rule_name="PRIZE_LOTTERY_BAIT",
            description="Message uses prize/lottery bait — classic advance-fee scam pattern",
            penalty=22,
            matched_text=snippet
        ))

    # Count ALL-CAPS words — scammers use this to simulate urgency
    caps_count = len(EXCESSIVE_CAPS_PATTERN.findall(text))
    if caps_count >= 3:
        matches.append(RuleMatch(
            rule_name="EXCESSIVE_CAPS",
            description=f"Message has {caps_count} ALL-CAPS words (urgency signalling)",
            penalty=8,
            matched_text=f"{caps_count} caps words"
        ))

    # Multiple consecutive exclamation marks
    if EXCESSIVE_EXCLAMATION.search(text):
        matches.append(RuleMatch(
            rule_name="EXCESSIVE_EXCLAMATIONS",
            description="Multiple consecutive exclamation marks (!! or !!!)",
            penalty=5,
            matched_text=EXCESSIVE_EXCLAMATION.search(text).group()
        ))

    return matches


# ---------------------------------------------------------------------------
# Sender Anomaly Rules  (applied when sender metadata is available)
# ---------------------------------------------------------------------------

# TRAI-registered DLT headers for commercial SMS in India follow the format:
# 2-letter prefix + hyphen + up to 6 alphanumeric chars, e.g. "VM-HDFCBK"
DLT_HEADER_PATTERN = re.compile(r'^[A-Z]{2}-[A-Z0-9]{1,6}$')

# International number prefixes that commonly send fake "Indian bank" alerts
FOREIGN_PREFIX = re.compile(r'^\+(?!91)\d{1,3}')


def check_sender_anomalies(sender: str) -> List[RuleMatch]:
    """
    Evaluate the SMS sender field for anomalies.

    In India, legitimate commercial SMS must be sent through a
    TRAI-registered DLT header (e.g. "VM-SBIBNK"). Personal mobile
    numbers or foreign numbers sending bank/government alerts are
    strong red flags.

    Args:
        sender: The sender ID string (e.g. "VM-HDFCBK" or "+447911123456").

    Returns:
        List of RuleMatch objects.
    """
    matches: List[RuleMatch] = []

    if not sender:
        return matches

    sender = sender.strip()

    # Check if sender is a plain mobile number (not a DLT header)
    if re.match(r'^\d{10}$', sender):
        matches.append(RuleMatch(
            rule_name="SENDER_PLAIN_MOBILE",
            description="SMS sent from a 10-digit mobile number, not a registered DLT header",
            penalty=20,
            matched_text=sender
        ))

    # Check for international prefix (non-Indian country code)
    elif FOREIGN_PREFIX.match(sender):
        matches.append(RuleMatch(
            rule_name="SENDER_FOREIGN_NUMBER",
            description="SMS sent from a non-Indian international number",
            penalty=25,
            matched_text=sender
        ))

    # DLT-formatted header — check it's syntactically correct
    elif not DLT_HEADER_PATTERN.match(sender.upper()):
        matches.append(RuleMatch(
            rule_name="SENDER_MALFORMED_HEADER",
            description="Sender ID does not match TRAI DLT format (XX-XXXXXX)",
            penalty=18,
            matched_text=sender
        ))

    return matches


# ---------------------------------------------------------------------------
# Main Rule Engine Entry Point
# ---------------------------------------------------------------------------

def run_rule_engine(text: str, sender: str = "") -> RuleEngineResult:
    """
    Run all rule-based checks on the given message and sender.

    This is the single public function called by the scorer module.
    It aggregates URL checks, text keyword checks, and sender anomaly
    checks into a unified RuleEngineResult.

    Args:
        text:   The SMS body or URL string to analyse.
        sender: Optional sender ID / phone number string.

    Returns:
        RuleEngineResult with all fired rules and their penalties.
    """
    result = RuleEngineResult()

    # Run each category of rules and collect matches
    result.matches.extend(check_url_patterns(text))
    result.matches.extend(check_text_keywords(text))
    result.matches.extend(check_sender_anomalies(sender))

    return result
