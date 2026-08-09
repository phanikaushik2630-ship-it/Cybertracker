"""
api/index.py — Vercel Serverless Entry Point for CyberTrack FastAPI
===================================================================
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional, Any

# Ensure project root is in sys.path for relative/absolute imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except Exception as exc:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

from fraud_detector import FraudScorer
from fraud_detector.claude_explainer import (
    build_claude_prompt,
    parse_claude_response,
    mock_claude_response,
    FraudExplanation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cybertrack.api")

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
USE_ML_MODELS  = os.getenv("USE_ML_MODELS", "false").lower() == "true"


class AppState:
    scorer: Optional[FraudScorer] = None
    gemini_model: Any = None
    gemini_available: bool = False


app_state = AppState()


def get_scorer() -> FraudScorer:
    if app_state.scorer is None:
        try:
            app_state.scorer = FraudScorer(use_ml=USE_ML_MODELS)
        except Exception as exc:
            logger.error(f"Error creating FraudScorer: {exc}")
            app_state.scorer = FraudScorer(use_ml=False)
    return app_state.scorer


def get_gemini_client():
    if not GOOGLE_API_KEY or not GENAI_AVAILABLE:
        app_state.gemini_available = False
        return None

    if app_state.gemini_model is None and not getattr(app_state, "_gemini_failed", False):
        try:
            app_state.gemini_model = genai.Client(api_key=GOOGLE_API_KEY)
            app_state.gemini_available = True
            logger.info("Gemini client initialised successfully.")
        except Exception as exc:
            logger.error(f"Gemini initialisation failed: {exc}")
            app_state._gemini_failed = True
            app_state.gemini_available = False
            return None

    return app_state.gemini_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CyberTrack API on Vercel Serverless...")
    get_scorer()
    get_gemini_client()
    yield
    logger.info("Shutting down CyberTrack API...")


app = FastAPI(
    title="CyberTrack — SMS & Phishing Fraud Detection API",
    description="Scores text messages and URLs for fraud risk and provides Gemini explanations.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    html_file = os.path.join(ROOT_DIR, "static", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return RedirectResponse(url="/docs")


class CheckMessageRequest(BaseModel):
    message: str = Field(..., description="Message text or URL to analyse")
    sender: Optional[str] = Field(None, description="Optional sender ID")

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("message parameter cannot be empty or whitespace.")
        return s


class CheckMessageResponse(BaseModel):
    verdict: str
    explanation: str
    action: str
    risk_score: float
    risk_label: str
    signals: list
    detected_urls: list
    score_breakdown: dict
    gemini_used: bool
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    gemini_available: bool
    ml_models_active: bool
    model: str
    version: str


def call_gemini(prompt_user: str) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable")

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt_user,
            config=genai_types.GenerateContentConfig(
                system_instruction="You are CyberTrack AI fraud expert.",
                temperature=0.2,
                max_output_tokens=300,
            ),
        )
        return response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    get_gemini_client()
    return HealthResponse(
        status="ok",
        gemini_available=app_state.gemini_available,
        ml_models_active=USE_ML_MODELS,
        model=GEMINI_MODEL,
        version=app.version,
    )


@app.post("/check-message", response_model=CheckMessageResponse, tags=["Detection"])
async def check_message(payload: CheckMessageRequest) -> CheckMessageResponse:
    t_start = time.perf_counter()
    scorer = get_scorer()
    breakdown = scorer.score(payload.message, payload.sender)

    gemini_used = False
    explanation_obj = None

    if app_state.gemini_available:
        try:
            prompt_user = build_claude_prompt(breakdown, payload.message)
            raw_text = call_gemini(prompt_user)
            explanation_obj = parse_claude_response(raw_text)
            gemini_used = True
        except Exception as exc:
            logger.warning(f"Gemini unavailable, fallback to mock: {exc}")

    if explanation_obj is None:
        explanation_obj = mock_claude_response(breakdown)

    t_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return CheckMessageResponse(
        verdict=explanation_obj.verdict,
        explanation=explanation_obj.explanation,
        action=explanation_obj.action,
        risk_score=breakdown.final_score,
        risk_label=breakdown.risk_label,
        signals=breakdown.triggered_rules,
        detected_urls=breakdown.detected_urls,
        score_breakdown={
            "rule_score": breakdown.rule_score,
            "sms_ml_score": breakdown.sms_ml_score,
            "url_ml_score": breakdown.url_ml_score,
            "ml_active": breakdown.ml_models_active,
        },
        gemini_used=gemini_used,
        processing_time_ms=t_elapsed_ms,
    )
