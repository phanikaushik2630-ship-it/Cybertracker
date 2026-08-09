# CyberTrack — SMS & Phishing Fraud Detection Module
**Final Year B.Tech CSE Project** | Rule-Based + ML Risk Scoring for Indian SMEs

---

## Project Structure

```
CyberTrack/
├── fraud_detector/
│   ├── __init__.py          # Package entry point
│   ├── rule_engine.py       # Regex & keyword rule-based checks
│   ├── url_analyzer.py      # 17-feature URL feature extractor
│   ├── sms_classifier.py    # TF-IDF + Naive Bayes for SMS text
│   ├── url_classifier.py    # Logistic Regression for URL features
│   └── scorer.py            # Combined 0–100 risk scorer
├── data/                    # Place downloaded CSV datasets here
├── models/                  # Trained .pkl models saved here
├── train.py                 # Model training CLI script
├── example_usage.py         # Demo with 9 test cases
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Rules-only mode (no dataset needed)
```bash
python example_usage.py
```

### 3. Full ML mode

**Download datasets from Kaggle:**
- [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) → save as `data/spam.csv`
- [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) → save as `data/malicious_phish.csv`

**Train the models:**
```bash
python train.py --sms_data data/spam.csv --url_data data/malicious_phish.csv
```

**Run with ML scoring:**
```bash
python example_usage.py --use_ml
```

---

## Scoring Architecture

```
Input Message (text + optional sender)
        │
   ┌────▼────────────┐
   │  Rule Engine    │  Regex + keyword checks → Rule Score (0–100)
   └────────────────┘
        │
   ┌────▼────────────┐
   │ SMS Classifier  │  TF-IDF + Naive Bayes → SMS ML Score (0–100)
   └────────────────┘
        │
   ┌────▼────────────┐
   │ URL Classifier  │  17 URL features + Logistic Regression → URL ML Score (0–100)
   └────────────────┘
        │
   ┌────▼────────────────────────────────────┐
   │  Weighted Average  →  Final Score 0–100 │
   │  Rule 40% + SMS ML 35% + URL ML 25%     │
   └─────────────────────────────────────────┘
```

### Score Interpretation
| Score Range | Label | Recommended Action |
|---|---|---|
| 0–25 | Low Risk [SAFE] | Allow |
| 26–50 | Moderate Risk [WARNING] | Flag for review |
| 51–75 | High Risk [DANGER] | Block + warn user |
| 76–100 | Critical Risk [PHISHING] | Block immediately |

---

## Rule Categories (rule_engine.py)

| Category | Rules | Max Penalty |
|---|---|---|
| URL Patterns | Suspicious TLD, IP in URL, Brand in subdomain, Shortener, Punycode | 30 pts each |
| Text Keywords | Urgency, Payment request, OTP harvest, Authority impersonation, Prize bait | 18–30 pts each |
| Sender Anomaly | Plain mobile number, Foreign prefix, Malformed DLT header | 10–25 pts each |

---

## In-code Usage
```python
from fraud_detector import FraudScorer

scorer = FraudScorer(use_ml=True)   # use_ml=False for rules only

result = scorer.score(
    "URGENT: Update Aadhaar KYC or account blocked. http://uidai-verify.xyz",
    sender="+447700900123"
)

print(result.final_score)       # e.g. 91.4
print(result.risk_label)        # "Critical Risk [PHISHING]"
print(result.triggered_rules)   # ["SUSPICIOUS_TLD", "URGENCY_LANGUAGE", ...]
print(scorer.explain(result))   # Full formatted report
```

---

## Datasets Used

| Dataset | Source | Use |
|---|---|---|
| SMS Spam Collection | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | SMS text classifier training |
| Malicious URLs Dataset | [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | URL risk classifier training |

---

*Built with scikit-learn, pandas, and Python 3.10+*
