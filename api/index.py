import logging
import os
import sys
import time
import re
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Optional, Any, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# EMBEDDED RULE ENGINE & SCORER (Zero external file import dependencies)
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = re.compile(
    r'\.(xyz|top|club|site|online|tech|info|work|live|icu|online|tk|ml|ga|cf|gq|pw|cc|bid|loan|win|date|review|stream|download)\b',
    re.IGNORECASE
)

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

PAYMENT_PHRASES = re.compile(
    r'\b(send money|transfer (rs\.?|inr|₹)?\s*[\d,]+|pay now|upi (id|link|payment)|'
    r'wallet|recharge|cashback (credited|pending)|click to (pay|receive)|'
    r'prize money|lottery (won|winner)|lucky draw|redeem now|claim your (reward|prize|cash))\b',
    re.IGNORECASE
)

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

AUTHORITY_IMPERSONATION = re.compile(
    r'\b(trai|rbi|income tax (dept|department|notice)|it department|uidai|epfo|epf|'
    r'aadhaar (update|link|verify|expired)|pan card (link|verify|block)|'
    r'government of india|ministry of|sebi|irda|npci)\b',
    re.IGNORECASE
)

ACCOUNT_SECURITY_THREAT = re.compile(
    r'\b(your (bank )?account|account (number|ending|no\.?)|'
    r'(bank|credit|debit) (account|card)|net.?banking|'
    r'transaction (has been|was) (blocked|flagged|declined)|'
    r'(login|sign.?in) attempt|unauthori[sz]ed (access|transaction|activity)|'
    r'account (security|protection)|security (breach|concern|measure)|'
    r'unusual (login|access|transaction)|suspicious (login|access))\b',
    re.IGNORECASE
)

DLT_HEADER_PATTERN = re.compile(r'^[A-Z]{2}-[A-Z0-9]{1,6}$')
FOREIGN_PREFIX = re.compile(r'^\+(?!91)\d{1,3}')
URL_PATTERN = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)


@dataclass
class SimpleScoreBreakdown:
    rule_score: float = 0.0
    sms_ml_score: float = 0.0
    url_ml_score: float = 0.0
    final_score: float = 0.0
    risk_label: str = "Low Risk [SAFE]"
    triggered_rules: List[str] = field(default_factory=list)
    detected_urls: List[str] = field(default_factory=list)
    ml_models_active: bool = False


def calculate_risk(text: str, sender: Optional[str] = None) -> SimpleScoreBreakdown:
    penalty = 0
    rules = []
    urls = URL_PATTERN.findall(text)

    for url in urls:
        if SUSPICIOUS_TLDS.search(url):
            penalty += 20
            rules.append("SUSPICIOUS_TLD")

    if URGENCY_PHRASES.search(text):
        penalty += 18
        rules.append("URGENCY_LANGUAGE")

    if PAYMENT_PHRASES.search(text):
        penalty += 20
        rules.append("PAYMENT_REQUEST")

    if OTP_HARVEST_PHRASES.search(text):
        penalty += 30
        rules.append("OTP_CREDENTIAL_HARVEST")

    if AUTHORITY_IMPERSONATION.search(text):
        penalty += 25
        rules.append("AUTHORITY_IMPERSONATION")

    if ACCOUNT_SECURITY_THREAT.search(text):
        penalty += 20
        rules.append("ACCOUNT_SECURITY_THREAT")

    if sender:
        sender = sender.strip()
        if re.match(r'^\d{10}$', sender):
            penalty += 20
            rules.append("SENDER_PLAIN_MOBILE")
        elif FOREIGN_PREFIX.match(sender):
            penalty += 25
            rules.append("SENDER_FOREIGN_NUMBER")
        elif not DLT_HEADER_PATTERN.match(sender.upper()):
            penalty += 18
            rules.append("SENDER_MALFORMED_HEADER")

    final_score = round(min(penalty / 150.0, 1.0) * 100.0, 2)

    if final_score <= 25:
        label = "Low Risk [SAFE]"
    elif final_score <= 50:
        label = "Moderate Risk [WARNING]"
    elif final_score <= 75:
        label = "High Risk [DANGER]"
    else:
        label = "Critical Risk [PHISHING]"

    return SimpleScoreBreakdown(
        rule_score=final_score,
        final_score=final_score,
        risk_label=label,
        triggered_rules=rules,
        detected_urls=urls,
        ml_models_active=False
    )

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CyberTrack — SMS & Phishing Fraud Detection API",
    description="Scores text messages and URLs for fraud risk.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
@app.get("/api", include_in_schema=False)
@app.get("/api/index.py", include_in_schema=False)
async def root():
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "static", "index.html"),
        os.path.join(os.path.dirname(__file__), "..", "index.html"),
        os.path.join(os.getcwd(), "index.html"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception:
                pass
    return HTMLResponse(content="<h1>CyberTrack</h1><p>UI loading error. Check static files.</p>")


class CheckMessageRequest(BaseModel):
    message: str
    sender: Optional[str] = None


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/check-message")
@app.post("/api/check-message")
async def check_message(payload: CheckMessageRequest):
    t_start = time.perf_counter()
    b = calculate_risk(payload.message, payload.sender)

    if b.final_score > 50:
        verdict = "High Risk"
        explanation = "This message contains strong fraud signals such as urgent account restriction threats or suspicious link domain patterns."
        action = "Do not click any links or share information. Delete this message immediately."
    elif b.final_score > 25:
        verdict = "Suspicious"
        explanation = "This message contains warning signs like urgency pressure or unverified sender details."
        action = "Exercise caution. Verify with your bank or provider via their official website."
    else:
        verdict = "Safe"
        explanation = "No major suspicious signals detected in this message."
        action = "No action needed. Stay vigilant."

    t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "verdict": verdict,
        "explanation": explanation,
        "action": action,
        "risk_score": b.final_score,
        "risk_label": b.risk_label,
        "signals": b.triggered_rules,
        "detected_urls": b.detected_urls,
        "score_breakdown": {
            "rule_score": b.rule_score,
            "sms_ml_score": 0.0,
            "url_ml_score": 0.0,
            "ml_active": False,
        },
        "gemini_used": False,
        "processing_time_ms": t_elapsed_ms,
    }
