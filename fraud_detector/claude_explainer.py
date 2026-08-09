"""
claude_explainer.py — Claude Prompt Builder for Fraud Explanations
===================================================================
Takes a scored fraud analysis result (ScoreBreakdown from scorer.py)
and produces a structured prompt to send to Claude (via the Anthropic
API or Firebase AI Logic). Claude returns a calm, plain-language
explanation suitable for a non-technical SME user.

WHY A SEPARATE PROMPT MODULE?
  The ML/rule engine produces *signals* (numbers and rule names).
  Claude's job is to translate those signals into *meaning* — a
  verdict and explanation that a shop owner or accountant can act on
  without understanding what "TF-IDF" or "Logistic Regression" means.

INTEGRATION POINTS:
  - Input  : ScoreBreakdown (from scorer.py) + optional raw message
  - Output : A formatted prompt string (send this to Claude)
  - Parsing: parse_claude_response() extracts the 3 structured fields

USAGE:
    from fraud_detector.scorer import FraudScorer
    from fraud_detector.claude_explainer import build_claude_prompt, parse_claude_response

    scorer = FraudScorer()
    result = scorer.score("Win Rs 50,000 now! http://prize-claim.xyz", sender="+4479...")

    prompt = build_claude_prompt(result, raw_message="Win Rs 50,000 now! ...")
    # → send `prompt` to Claude API, get back `response_text`

    explanation = parse_claude_response(response_text)
    print(explanation.verdict)        # "High Risk"
    print(explanation.explanation)    # 2–3 plain sentences
    print(explanation.action)         # "Do not click any links..."
"""

import textwrap
from dataclasses import dataclass
from typing import Optional

from .scorer import ScoreBreakdown


# ---------------------------------------------------------------------------
# Human-readable signal label map
# ---------------------------------------------------------------------------
# Maps internal rule names (from rule_engine.py) to plain-language phrases
# that Claude can weave into its explanation naturally.

SIGNAL_LABELS = {
    # URL rules
    "SUSPICIOUS_TLD":        "the link uses an unusual web address ending (.xyz, .tk, etc.)",
    "BRAND_IN_SUBDOMAIN":    "a trusted brand name is used in a misleading part of the link",
    "IP_IN_URL":             "the link goes to a raw IP address instead of a named website",
    "URL_SHORTENER":         "the link is hidden behind a URL shortener",
    "URL_SPECIAL_CHARS":     "the link contains unusual characters that can hide its true destination",
    "PUNYCODE_HOMOGRAPH":    "the link uses special Unicode characters to impersonate a known website",
    "EXCESSIVE_SUBDOMAINS":  "the link has an unusually complex web address structure",
    "LONG_URL":              "the web address is unusually long",
    # Text keyword rules
    "URGENCY_LANGUAGE":      "the message tries to create a sense of urgency or panic",
    "PAYMENT_REQUEST":       "the message requests a payment or money transfer",
    "OTP_CREDENTIAL_HARVEST":"the message asks for an OTP, KYC details, or personal ID numbers",
    "AUTHORITY_IMPERSONATION":"the message claims to be from a government body or official institution",
    "PRIZE_LOTTERY_BAIT":    "the message offers a prize, lottery win, or unexpected reward",
    "EXCESSIVE_CAPS":        "the message uses aggressive all-capital-letter words",
    "EXCESSIVE_EXCLAMATIONS":"the message overuses exclamation marks to create urgency",
    # Sender rules
    "SENDER_PLAIN_MOBILE":   "the message was sent from a personal mobile number, not a registered sender ID",
    "SENDER_FOREIGN_NUMBER": "the message was sent from an international phone number",
    "SENDER_MALFORMED_HEADER":"the sender ID does not match the standard format used by legitimate businesses",
}


def _humanise_signals(triggered_rules: list[str]) -> str:
    """
    Convert a list of internal rule names into a numbered plain-English list.

    Used inside the prompt so Claude has readable context rather than
    cryptic codes like 'SUSPICIOUS_TLD'.

    Args:
        triggered_rules: List of rule name strings from ScoreBreakdown.

    Returns:
        A formatted numbered string, one signal per line.
    """
    lines = []
    for i, rule in enumerate(triggered_rules, start=1):
        label = SIGNAL_LABELS.get(rule, rule.replace("_", " ").lower())
        lines.append(f"  {i}. {label}")
    return "\n".join(lines) if lines else "  (no specific signals detected)"


def _score_to_tier(score: float) -> str:
    """
    Map a numeric risk score to one of three clean tier labels.
    Matches the verdict options Claude is asked to choose from.
    """
    if score <= 30:
        return "Safe"
    elif score <= 60:
        return "Suspicious"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

# The system prompt establishes Claude's role, tone, and constraints.
# This is sent as the `system` parameter in the Anthropic Messages API.
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a fraud detection assistant helping small business owners and
    everyday users in India understand whether a text message or link they
    received is safe or potentially dangerous.

    Your communication style:
    - Write in clear, simple English that anyone can understand — avoid
      technical jargon (no terms like "TLD", "regex", "ML model", "phishing").
    - Stay calm and measured. Do NOT use alarming language like "you are being
      attacked" or "hackers". Instead, be factual and reassuring.
    - Be concise. Users are busy. Two to three sentences is enough.
    - You may refer to common Indian context: UPI payments, Aadhaar, KYC,
      TRAI, bank SMS headers, etc., since the users are Indian SME owners.
    - Never fabricate specific details about the message you were not given.
    - Always output your response in EXACTLY the structured format below.
      Do not add any extra prose, headings, or commentary outside it.

    OUTPUT FORMAT (follow this exactly, including the labels):
    VERDICT: <one of: Safe | Suspicious | High Risk>
    EXPLANATION: <2–3 calm, plain-language sentences explaining why this
    message received this rating, mentioning 1–2 of the most important signals
    without using technical names for them>
    ACTION: <one specific, actionable instruction the user should take right now>
""")


# The user-turn prompt template. Placeholders are filled by build_claude_prompt().
USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    A user submitted the following message for fraud risk analysis.

    --- MESSAGE (may be truncated for privacy) ---
    {message_preview}
    --- END MESSAGE ---

    Our automated system analysed this message and produced:

    RISK SCORE : {score:.0f} out of 100  ({tier})
    SIGNALS DETECTED:
    {signals_list}

    Based on the risk score and the signals above, please provide your
    structured response in the exact format specified.
""")


def build_claude_prompt(
    breakdown: ScoreBreakdown,
    raw_message: str = "",
    preview_length: int = 200,
) -> dict:
    """
    Build a ready-to-send Claude API payload from a ScoreBreakdown.

    Returns a dict with 'system' and 'user' keys, matching the structure
    expected by the Anthropic Messages API and Firebase AI Logic SDK.

    Args:
        breakdown:      ScoreBreakdown returned by FraudScorer.score().
        raw_message:    The original SMS text (used for context preview).
                        If empty, a placeholder is used.
        preview_length: Max characters of the message shown to Claude.
                        Keeps the prompt short and respects user privacy.

    Returns:
        Dict with two keys:
          'system' → the SYSTEM_PROMPT string (role + tone instructions)
          'user'   → the formatted user-turn prompt string

    Example:
        >>> prompt = build_claude_prompt(breakdown, raw_message=sms_text)
        >>> # Anthropic SDK:
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-5",
        ...     max_tokens=300,
        ...     system=prompt["system"],
        ...     messages=[{"role": "user", "content": prompt["user"]}]
        ... )
    """
    # Truncate the message to protect privacy and keep the prompt short
    if raw_message:
        preview = raw_message[:preview_length]
        if len(raw_message) > preview_length:
            preview += "... [truncated]"
    else:
        preview = "(message text not provided)"

    tier = _score_to_tier(breakdown.final_score)
    signals_list = _humanise_signals(breakdown.triggered_rules)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        message_preview=preview,
        score=breakdown.final_score,
        tier=tier,
        signals_list=signals_list,
    )

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------

@dataclass
class FraudExplanation:
    """
    Structured output parsed from Claude's response.

    Attributes:
        verdict:     One of 'Safe', 'Suspicious', or 'High Risk'.
        explanation: 2–3 plain-language sentences for the user.
        action:      One clear recommended action.
        raw:         The full unmodified response text from Claude.
    """
    verdict: str
    explanation: str
    action: str
    raw: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for JSON API responses)."""
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "action": self.action,
        }

    def __str__(self) -> str:
        return (
            f"Verdict     : {self.verdict}\n"
            f"Explanation : {self.explanation}\n"
            f"Action      : {self.action}"
        )


def parse_claude_response(response_text: str) -> FraudExplanation:
    """
    Parse Claude's structured text response into a FraudExplanation object.

    Handles minor formatting variations (extra spaces, different line
    endings) that can occur in real API responses.

    Args:
        response_text: The raw text content from Claude's API response.

    Returns:
        FraudExplanation with verdict, explanation, and action fields.
        Falls back gracefully if the format is unexpected.

    Example:
        >>> raw = '''
        ... VERDICT: High Risk
        ... EXPLANATION: This message claims to be from UIDAI and asks you to
        ... update your Aadhaar details urgently. It also contains a link that
        ... leads to an unofficial website not associated with the government.
        ... ACTION: Do not click the link. If you are worried about your Aadhaar
        ... status, visit the official uidai.gov.in website directly.
        ... '''
        >>> result = parse_claude_response(raw)
        >>> result.verdict
        'High Risk'
    """
    verdict = "Unknown"
    explanation = ""
    action = ""

    # Split into lines and extract each labelled field
    lines = response_text.strip().splitlines()
    current_field = None
    field_buffer = []

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("VERDICT:"):
            # Flush any previous buffer
            if current_field == "EXPLANATION":
                explanation = " ".join(field_buffer).strip()
            elif current_field == "ACTION":
                action = " ".join(field_buffer).strip()

            current_field = "VERDICT"
            field_buffer = []
            # Extract value on same line as label
            value = stripped[len("VERDICT:"):].strip()
            if value:
                verdict = value

        elif stripped.upper().startswith("EXPLANATION:"):
            if current_field == "VERDICT":
                pass  # verdict was already captured inline
            current_field = "EXPLANATION"
            field_buffer = []
            value = stripped[len("EXPLANATION:"):].strip()
            if value:
                field_buffer.append(value)

        elif stripped.upper().startswith("ACTION:"):
            if current_field == "EXPLANATION":
                explanation = " ".join(field_buffer).strip()
            current_field = "ACTION"
            field_buffer = []
            value = stripped[len("ACTION:"):].strip()
            if value:
                field_buffer.append(value)

        else:
            # Continuation line of current field
            if current_field in ("EXPLANATION", "ACTION") and stripped:
                field_buffer.append(stripped)

    # Flush the final buffer
    if current_field == "EXPLANATION" and field_buffer:
        explanation = " ".join(field_buffer).strip()
    elif current_field == "ACTION" and field_buffer:
        action = " ".join(field_buffer).strip()

    # Normalise verdict to the three canonical values
    verdict_lower = verdict.lower()
    if "high" in verdict_lower:
        verdict = "High Risk"
    elif "suspicious" in verdict_lower:
        verdict = "Suspicious"
    elif "safe" in verdict_lower:
        verdict = "Safe"

    return FraudExplanation(
        verdict=verdict,
        explanation=explanation,
        action=action,
        raw=response_text,
    )


# ---------------------------------------------------------------------------
# Convenience: simulate Claude response for offline testing
# ---------------------------------------------------------------------------

def mock_claude_response(breakdown: ScoreBreakdown) -> FraudExplanation:
    """
    Generate a deterministic simulated explanation WITHOUT calling the
    Claude API. Useful for unit tests, demos, and offline development.

    The response is generated using rule-based templates keyed on the
    most severe signals detected, matching Claude's expected output format.

    Args:
        breakdown: ScoreBreakdown from FraudScorer.score().

    Returns:
        FraudExplanation with plausible verdict, explanation, and action.
    """
    score = breakdown.final_score
    rules = set(breakdown.triggered_rules)

    # -- Verdict: must match scorer.py's label boundaries exactly --
    # Safe: 0-25 | Moderate: 26-50 | High Risk: 51-75 | Critical: 76-100
    if score <= 25:
        verdict = "Safe"
    elif score <= 55:
        verdict = "Suspicious"
    else:
        verdict = "High Risk"

    # -- Pick the most prominent signals to build the explanation --
    explanation_parts = []

    if "AUTHORITY_IMPERSONATION" in rules:
        explanation_parts.append(
            "This message claims to be from a government authority or official body, "
            "which is a common tactic used in scams."
        )
    if "OTP_CREDENTIAL_HARVEST" in rules:
        explanation_parts.append(
            "It asks for sensitive information like an OTP or personal ID — "
            "no legitimate bank or government office will ever ask for this over SMS."
        )
    if "SUSPICIOUS_TLD" in rules or "BRAND_IN_SUBDOMAIN" in rules:
        explanation_parts.append(
            "The link in the message leads to an unofficial website "
            "that is not associated with the organisation it claims to represent."
        )
    if "URGENCY_LANGUAGE" in rules:
        explanation_parts.append(
            "The message tries to pressure you into acting quickly — "
            "scammers use urgency to stop you from thinking carefully."
        )
    if "ACCOUNT_SECURITY_THREAT" in rules:
        explanation_parts.append(
            "The message uses bank security alert language (account restriction, unusual activity) "
            "to create panic. Genuine banks never ask you to verify your account by clicking an SMS link."
        )
    if "PAYMENT_REQUEST" in rules:
        explanation_parts.append(
            "It asks you to make a payment or transfer money, "
            "which is a very common sign of a financial scam."
        )
    if "SENDER_FOREIGN_NUMBER" in rules or "SENDER_PLAIN_MOBILE" in rules:
        explanation_parts.append(
            "This message was sent from a personal or international phone number "
            "rather than a registered business sender ID."
        )
    if "PRIZE_LOTTERY_BAIT" in rules:
        explanation_parts.append(
            "Offers of unexpected prizes or lottery winnings are almost always scams — "
            "you cannot win a contest you never entered."
        )

    # Default fallback explanation
    if not explanation_parts:
        if verdict == "Safe":
            explanation_parts = [
                "This message does not show any significant warning signs. "
                "It appears to be a routine message."
            ]
        else:
            explanation_parts = [
                "This message shows some unusual patterns that are worth being cautious about."
            ]

    # Use at most 2 sentences for the explanation
    explanation = " ".join(explanation_parts[:2])

    # -- Action --
    if verdict == "High Risk":
        action = (
            "Do not click any links or share any information. "
            "Delete this message immediately and, if it claims to be from your bank, "
            "call your bank directly using the number on their official website."
        )
    elif verdict == "Suspicious":
        action = (
            "Be cautious. Verify the message by contacting the organisation "
            "directly through their official website or customer care number "
            "before taking any action."
        )
    else:
        action = (
            "No action needed — this message appears safe. "
            "Always stay alert and never share OTPs or passwords over messages."
        )

    return FraudExplanation(
        verdict=verdict,
        explanation=explanation,
        action=action,
        raw="[mock response — Claude API not called]",
    )
