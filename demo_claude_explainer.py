"""
demo_claude_explainer.py — Prompt Template Demo
=================================================
Shows the exact prompt that would be sent to Claude for several
fraud scenarios, then demonstrates the mock_claude_response()
for offline development (no API key needed).

Run:
    python demo_claude_explainer.py
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fraud_detector import FraudScorer
from fraud_detector.claude_explainer import (
    build_claude_prompt,
    parse_claude_response,
    mock_claude_response,
    SYSTEM_PROMPT,
)

DIVIDER = "=" * 70
THIN    = "-" * 70


def run_scenario(label: str, message: str, sender: str = "") -> None:
    """Score one message and print the prompt + mock explanation."""
    scorer = FraudScorer(use_ml=False)
    breakdown = scorer.score(message, sender=sender)

    print(f"\n{DIVIDER}")
    print(f"  SCENARIO: {label}")
    print(f"  Risk Score: {breakdown.final_score:.1f}/100  |  {breakdown.risk_label}")
    print(f"  Triggered : {', '.join(breakdown.triggered_rules) or 'none'}")
    print(THIN)

    # --- Show the formatted prompt Claude would receive ---
    prompt = build_claude_prompt(breakdown, raw_message=message)
    print("[SYSTEM PROMPT sent to Claude]")
    print(THIN)
    print(prompt["system"])
    print(THIN)
    print("[USER PROMPT sent to Claude]")
    print(THIN)
    print(prompt["user"])
    print(THIN)

    # --- Simulate what Claude would return (no API key needed) ---
    explanation = mock_claude_response(breakdown)
    print("[MOCK CLAUDE RESPONSE (offline simulation)]")
    print(THIN)
    print(f"VERDICT     : {explanation.verdict}")
    print(f"EXPLANATION : {explanation.explanation}")
    print(f"ACTION      : {explanation.action}")
    print(DIVIDER)


if __name__ == "__main__":
    print(f"\n{'CYBERTRACK — Claude Prompt Template Demo'.center(70)}")
    print(f"{'(shows prompt + offline mock response for 3 scenarios)'.center(70)}\n")

    run_scenario(
        label="Aadhaar KYC Phishing (Critical Risk)",
        message=(
            "URGENT: Your Aadhaar KYC is expired! "
            "Update immediately or your bank account will be BLOCKED. "
            "Click: http://uidai-kyc-update.xyz/verify?uid=12345"
        ),
        sender="+447700900123",
    )

    run_scenario(
        label="OTP Harvest + Urgency (Moderate Risk)",
        message="Please share your OTP to complete your wallet KYC before your account is suspended.",
        sender="VM-PAYTMK",
    )

    run_scenario(
        label="Legitimate Order Confirmation (Low Risk)",
        message=(
            "Your Flipkart order #98765 has been shipped. "
            "Expected delivery: tomorrow by 6 PM. Track: https://www.flipkart.com/track/98765"
        ),
        sender="VM-FKRTAI",
    )

    print("\n[TIP] To call the real Claude API, replace mock_claude_response() with:")
    print("""
  import anthropic
  client = anthropic.Anthropic(api_key="YOUR_KEY")
  prompt = build_claude_prompt(breakdown, raw_message=sms_text)

  response = client.messages.create(
      model="claude-sonnet-4-5",
      max_tokens=300,
      system=prompt["system"],
      messages=[{"role": "user", "content": prompt["user"]}]
  )
  explanation = parse_claude_response(response.content[0].text)
  print(explanation)
""")
