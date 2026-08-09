# -*- coding: utf-8 -*-
"""
example_usage.py — CyberTrack Demo Script
==========================================
Demonstrates the full scoring pipeline with several test cases
covering different risk levels and fraud patterns.

Run this after training:
    python example_usage.py              # rules only (no models needed)
    python example_usage.py --use_ml     # with trained ML models
"""

import argparse
import logging
import sys
import io
from fraud_detector import FraudScorer

# Force UTF-8 output on Windows to handle emoji/special characters
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.WARNING)  # Suppress info logs in demo


def run_demo(use_ml: bool = False) -> None:
    """
    Run the scorer against a curated set of test messages spanning
    different fraud categories and risk levels.
    """
    # Initialise the scorer (loads ML models if use_ml=True and models exist)
    scorer = FraudScorer(use_ml=use_ml)

    test_cases = [
        # ------------------------------------------------------------------
        # CRITICAL RISK — Indian phishing SMS examples
        # ------------------------------------------------------------------
        {
            "label": "Aadhaar KYC Phishing (with suspicious URL)",
            "text": (
                "URGENT: Your Aadhaar KYC is expired! "
                "Update immediately or your bank account will be BLOCKED. "
                "Click: http://uidai-kyc-update.xyz/verify?uid=12345"
            ),
            "sender": "+447700900123",  # Foreign number = anomaly
        },
        {
            "label": "Prize/Lottery Scam",
            "text": (
                "Congratulations!! You have WON Rs.2,50,000 in the Lucky Draw. "
                "Claim your prize NOW before it expires in 24 hours. "
                "Send UPI payment of Rs.199 to claim@upi to receive your reward."
            ),
            "sender": "9876543210",  # Plain mobile = anomaly
        },
        {
            "label": "Fake TRAI Threat",
            "text": (
                "TRAI Notice: Your mobile number will be disconnected by tonight "
                "due to illegal activity. Call 9988776655 immediately to avoid action. "
                "Visit: http://trai-support.tk/resolve"
            ),
            "sender": "",
        },

        # ------------------------------------------------------------------
        # HIGH RISK — URL-heavy phishing
        # ------------------------------------------------------------------
        {
            "label": "SBI Brand Impersonation URL",
            "text": "Dear customer, verify your SBI account at http://sbi.secure-login.xyz",
            "sender": "VM-FAKBK",  # Malformed DLT header
        },
        {
            "label": "URL Shortener Hiding Destination",
            "text": "Your HDFC account needs verification. Click: https://bit.ly/hdfc-verify",
            "sender": "VM-HDFCBK",
        },

        # ------------------------------------------------------------------
        # MODERATE RISK — Suspicious but incomplete signals
        # ------------------------------------------------------------------
        {
            "label": "Vague Urgency with No URL",
            "text": "Your account will be blocked. Contact us immediately to resolve.",
            "sender": "VM-MYBANKK",
        },
        {
            "label": "OTP Request (No URL)",
            "text": "Please share your OTP to complete KYC verification for your wallet.",
            "sender": "VM-PAYTMK",
        },

        # ------------------------------------------------------------------
        # LOW RISK — Legitimate messages
        # ------------------------------------------------------------------
        {
            "label": "Legitimate OTP SMS",
            "text": "Your OTP for login is 483921. Valid for 10 minutes. Do not share.",
            "sender": "VM-SBIBNK",
        },
        {
            "label": "Legitimate Promotional SMS",
            "text": (
                "Hi! Your order #12345 has been shipped and will arrive by tomorrow. "
                "Track at: https://www.flipkart.com/track/12345"
            ),
            "sender": "VM-FKRTAI",
        },
    ]

    print("\n" + "[**]  CYBERTRACK -- Fraud Risk Detection Demo  [**]".center(60))
    print("=" * 60)
    print(f"Mode: {'ML + Rules' if use_ml else 'Rules Only (run with --use_ml for ML)'}")
    print("=" * 60)

    for case in test_cases:
        result = scorer.score(case["text"], sender=case.get("sender", ""))
        print(f"\n📌 {case['label']}")
        msg_preview = case['text'][:80] + ('...' if len(case['text']) > 80 else '')
        print(f"   Message: \"{msg_preview}\"")
        print(f"   Sender : {case.get('sender', '(none)')}")
        # Strip emoji from risk_label for Windows console safety
        safe_label = result.risk_label.encode('ascii', 'ignore').decode('ascii').strip()
        print(f"   >> Score : {result.final_score:.1f}/100  |  {safe_label}")
        if result.triggered_rules:
            print(f"   >> Rules : {', '.join(result.triggered_rules)}")

    print("\n" + "=" * 60)
    print("For a full report, call: scorer.explain(result)")
    print("=" * 60 + "\n")

    # -- Full Explain output for the first test case --
    first_result = scorer.score(test_cases[0]["text"], sender=test_cases[0]["sender"])
    print("\nFULL REPORT FOR FIRST TEST CASE:")
    print(scorer.explain(first_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberTrack Demo")
    parser.add_argument(
        "--use_ml",
        action="store_true",
        help="Use trained ML models for scoring (requires running train.py first)"
    )
    args = parser.parse_args()
    run_demo(use_ml=args.use_ml)
