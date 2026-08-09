"""
url_analyzer.py — URL Feature Extraction for ML Classifier
===========================================================
Extracts a fixed-length numerical feature vector from a URL string.
These features are used to train and run a lightweight scikit-learn
classifier for malicious URL detection.

DATASET TARGET: "Malicious URLs Dataset" / "Benign and Malicious URLs" on Kaggle
  - Label column: 'type' (benign / phishing / malware / defacement)
  - Input column: 'url'

WHY FEATURE ENGINEERING INSTEAD OF RAW TEXT?
  Raw URLs are sparse tokens — feature engineering extracts semantic
  signals that generalise well across unseen phishing domains, even
  with a simple Logistic Regression model.
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, List


# ---------------------------------------------------------------------------
# Individual Feature Extraction Functions
# ---------------------------------------------------------------------------

def get_url_length(url: str) -> int:
    """Total character count of the URL. Phishing URLs tend to be longer."""
    return len(url)


def get_hostname_length(url: str) -> int:
    """Length of just the hostname portion (e.g. 'secure.sbi-login.xyz')."""
    try:
        return len(urlparse(url).netloc)
    except Exception:
        return 0


def count_dots_in_url(url: str) -> int:
    """
    Count of dots (.) in the full URL.
    More dots often indicate deeper subdomain nesting.
    """
    return url.count(".")


def count_hyphens_in_domain(url: str) -> int:
    """
    Hyphens in the domain name are a common phishing trick.
    e.g. 'sbi-secure-login.xyz' vs 'sbi.co.in'
    """
    try:
        hostname = urlparse(url).netloc
        return hostname.count("-")
    except Exception:
        return 0


def count_subdomains(url: str) -> int:
    """
    Number of subdomain levels above the registered domain.
    Legitimate sites rarely exceed 2 levels (www.example.com).
    """
    try:
        hostname = urlparse(url).netloc
        # Remove port if present
        hostname = hostname.split(":")[0]
        parts = hostname.split(".")
        # Subtract 2 for the registered domain + TLD
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def has_ip_address(url: str) -> int:
    """
    Binary: 1 if the host is a raw IPv4 address, 0 otherwise.
    e.g. http://192.168.1.1/phish → 1
    """
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    try:
        host = urlparse(url).netloc.split(":")[0]
        return 1 if ip_pattern.match(host) else 0
    except Exception:
        return 0


def uses_https(url: str) -> int:
    """
    Binary: 1 if scheme is https, 0 if http or missing.
    Note: HTTPS alone doesn't mean safe — phishing sites get free TLS certs.
    """
    try:
        return 1 if urlparse(url).scheme.lower() == "https" else 0
    except Exception:
        return 0


def count_special_chars(url: str) -> int:
    """
    Count of suspicious special characters in the URL.
    Characters like @, ~, % are often used in obfuscation attacks.
    """
    suspicious = set("@~%=+$!*,;")
    return sum(1 for c in url if c in suspicious)


def count_digits_in_domain(url: str) -> int:
    """
    Count of digits in the domain name.
    Legitimate domains rarely have many numbers (e.g. 'sbi123456login.xyz').
    """
    try:
        hostname = urlparse(url).netloc.split(":")[0]
        return sum(1 for c in hostname if c.isdigit())
    except Exception:
        return 0


def get_path_length(url: str) -> int:
    """Length of the URL path component (/pay/verify/submit/...)."""
    try:
        return len(urlparse(url).path)
    except Exception:
        return 0


def count_path_segments(url: str) -> int:
    """
    Number of '/' separated segments in the path.
    Deep paths can indicate hidden phishing landing pages.
    """
    try:
        path = urlparse(url).path
        return len([s for s in path.split("/") if s])
    except Exception:
        return 0


def has_query_string(url: str) -> int:
    """Binary: 1 if the URL has a query string (? parameters), 0 otherwise."""
    try:
        return 1 if urlparse(url).query else 0
    except Exception:
        return 0


def count_query_params(url: str) -> int:
    """Number of query parameters — phishing pages often pass tracking tokens."""
    try:
        return len(parse_qs(urlparse(url).query))
    except Exception:
        return 0


def is_url_shortener(url: str) -> int:
    """
    Binary: 1 if domain matches known URL shortener services.
    Shorteners hide the true destination.
    """
    shorteners = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "rb.gy", "cutt.ly", "short.io", "is.gd", "tiny.cc",
        "shorte.st", "adf.ly", "bc.vc"
    }
    try:
        hostname = urlparse(url).netloc.lower().replace("www.", "")
        return 1 if hostname in shorteners else 0
    except Exception:
        return 0


def get_tld_risk_score(url: str) -> int:
    """
    Categorical risk score for the top-level domain:
      0 = Low risk (.in, .com, .org, .gov, .edu, .net)
      1 = Medium risk (.co, .io, .info, .biz)
      2 = High risk (.xyz, .tk, .ml, .ga, .cf, .gq, .pw, .top, .click, etc.)
    """
    high_risk_tlds = {
        "xyz", "tk", "ml", "ga", "cf", "gq", "pw", "top",
        "click", "loan", "gdn", "win", "bid", "review", "trade",
        "party", "date", "faith", "racing", "stream"
    }
    medium_risk_tlds = {"co", "io", "info", "biz", "mobi", "name"}
    try:
        hostname = urlparse(url).netloc.lower()
        tld = hostname.split(".")[-1].split(":")[0]
        if tld in high_risk_tlds:
            return 2
        elif tld in medium_risk_tlds:
            return 1
        else:
            return 0
    except Exception:
        return 0


def contains_brand_keyword(url: str) -> int:
    """
    Binary: 1 if a known Indian/global brand appears in the URL.
    Phishing pages impersonate brands in subdomains or paths.
    e.g. 'sbi.secure-verify.xyz' or 'paytm-kyc.xyz/login'
    """
    brands = [
        "sbi", "icici", "hdfc", "axis", "paytm", "phonepe", "gpay",
        "amazon", "flipkart", "irctc", "aadhaar", "uidai", "rbi",
        "trai", "epfo", "npci", "bhim", "google", "facebook", "whatsapp"
    ]
    url_lower = url.lower()
    return 1 if any(brand in url_lower for brand in brands) else 0


def calculate_url_entropy(url: str) -> float:
    """
    Shannon entropy of the URL string.
    Highly randomised URLs (e.g. auto-generated phishing subdomains)
    have higher entropy. Formula: H = -Σ p(x) * log2(p(x))
    """
    if not url:
        return 0.0
    freq = {}
    for c in url:
        freq[c] = freq.get(c, 0) + 1
    total = len(url)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return round(entropy, 4)


# ---------------------------------------------------------------------------
# Feature Vector Builder
# ---------------------------------------------------------------------------

# Ordered list of feature names — must match the extraction order below.
# Used to label columns when building a DataFrame for training.
URL_FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "dot_count",
    "hyphen_count_domain",
    "subdomain_count",
    "has_ip_address",
    "uses_https",
    "special_char_count",
    "digit_count_domain",
    "path_length",
    "path_segment_count",
    "has_query_string",
    "query_param_count",
    "is_url_shortener",
    "tld_risk_score",
    "contains_brand_keyword",
    "url_entropy",
]


def extract_url_features(url: str) -> List[float]:
    """
    Extract a fixed-length feature vector from a URL string.

    This is the main function called during both training and inference.
    Each element corresponds to one entry in URL_FEATURE_NAMES.

    Args:
        url: A URL string (with or without http/https prefix).

    Returns:
        A list of 17 numeric features.

    Example:
        >>> extract_url_features("http://sbi.secure-verify.xyz/login?user=abc")
        [45, 22, 3, 1, 1, 0, 0, 1, 0, 6, 1, 1, 1, 0, 2, 1, 3.89]
    """
    # Ensure URL has a scheme for urlparse to work correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    features = [
        get_url_length(url),
        get_hostname_length(url),
        count_dots_in_url(url),
        count_hyphens_in_domain(url),
        count_subdomains(url),
        has_ip_address(url),
        uses_https(url),
        count_special_chars(url),
        count_digits_in_domain(url),
        get_path_length(url),
        count_path_segments(url),
        has_query_string(url),
        count_query_params(url),
        is_url_shortener(url),
        get_tld_risk_score(url),
        contains_brand_keyword(url),
        calculate_url_entropy(url),
    ]

    return features


def extract_url_features_dict(url: str) -> Dict[str, float]:
    """
    Same as extract_url_features() but returns a named dictionary.
    Useful for inspecting individual feature values during debugging.

    Args:
        url: A URL string.

    Returns:
        Dict mapping feature name → feature value.
    """
    values = extract_url_features(url)
    return dict(zip(URL_FEATURE_NAMES, values))
