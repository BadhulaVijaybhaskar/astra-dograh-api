"""Astra: structured, platform-wide TTS failure logging for runtime calls.

When a TTS service reports an error (for example Sarvam rejecting a speaker
that the selected model does not offer), log ONE structured line with the
employee, workflow, call id, provider, model, language, speaker, regional
style, a stable error code, HTTP status, a truncated provider error and a
timestamp. The provider error text is for logs only. A customer-safe message
is included separately. No API keys are ever logged.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from loguru import logger

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "od": "Odia", "or": "Odia",
}
MODEL_LABELS = {
    "bulbul:v4-flash": "Bulbul v4 Flash",
    "bulbul:v3": "Bulbul v3",
    "bulbul:v3-beta": "Bulbul v3 Beta",
    "bulbul:v2": "Bulbul v2",
}
_SECRET_RE = re.compile(r"(api[-_ ]?key|subscription[-_ ]?key|authorization|bearer)\S*", re.I)


def classify_tts_error(message: str) -> str:
    m = str(message or "")
    if re.search(r"not compatible with model|speaker .*(invalid|not (available|supported))", m, re.I):
        return "INCOMPATIBLE_SPEAKER"
    if re.search(r"language", m, re.I) and re.search(r"not supported|invalid|unsupported", m, re.I):
        return "UNSUPPORTED_LANGUAGE"
    if re.search(r"\b401\b|\b403\b|unauthori[sz]ed|forbidden", m, re.I):
        return "AUTH_FAILED"
    if re.search(r"\b429\b|rate limit", m, re.I):
        return "RATE_LIMITED"
    if re.search(r"timeout|timed out|too long", m, re.I):
        return "TIMEOUT"
    return "TTS_FAILED"


def http_status_from(message: str):
    m = re.search(r"(?:TTS Error:|API error:)?\s*\b([45]\d\d)\b\s*:", str(message or ""))
    return int(m.group(1)) if m else None


def _lang_name(lang) -> str:
    base = str(lang or "").split("-")[0].lower()
    return LANGUAGE_NAMES.get(base, str(lang or "this language"))


def _speaker_name(voice) -> str:
    v = str(voice or "").split("_")[0]
    return v[:1].upper() + v[1:] if v else "This voice"


def customer_message(code: str, voice, model, language) -> str:
    if code == "INCOMPATIBLE_SPEAKER":
        lname = _lang_name(language)
        return (
            f"Voice {_speaker_name(voice)} isn't compatible with {lname} / "
            f"{MODEL_LABELS.get(str(model or ''), str(model or 'this model'))}. "
            f"Choose one of the available {lname} voices."
        )
    return "The voice could not be generated right now. Please try again or choose another voice."


def build_record(ctx: dict, message: str) -> dict:
    code = classify_tts_error(message)
    return {
        "source": "dograh_runtime",
        "employee_id": ctx.get("employee_id"),
        "workflow_id": ctx.get("workflow_id"),
        "astra_workflow_id": ctx.get("astra_workflow_id"),
        "call_id": ctx.get("call_id"),
        "provider": ctx.get("provider"),
        "model": ctx.get("model"),
        "language": ctx.get("language"),
        "speaker": ctx.get("speaker"),
        "regional_style": ctx.get("regional_style"),
        "error_code": code,
        "http_status": http_status_from(message),
        "provider_error": _SECRET_RE.sub("[redacted]", str(message or ""))[:300],
        "customer_message": customer_message(code, ctx.get("speaker"), ctx.get("model"), ctx.get("language")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def tag_tts_service(svc, *, user_config=None, workflow_id=None, workflow_run_id=None, call_context=None):
    """Attach call context to a TTS service and log its errors (best effort)."""
    if svc is None or getattr(svc, "_astra_tts_error_tagged", False):
        return svc
    cc = call_context if isinstance(call_context, dict) else {}
    tts = getattr(user_config, "tts", None)
    ctx = {
        "employee_id": cc.get("employee_id"),
        "workflow_id": workflow_id,
        "astra_workflow_id": cc.get("astra_workflow_id") or cc.get("workflow_id"),
        "call_id": workflow_run_id,
        "provider": getattr(tts, "provider", None) or cc.get("astra_tts_provider"),
        "model": getattr(tts, "model", None) or cc.get("astra_tts_model"),
        "language": getattr(tts, "language", None) or cc.get("astra_tts_language"),
        "speaker": getattr(tts, "voice", None) or getattr(tts, "speaker", None) or cc.get("astra_tts_voice"),
        "regional_style": cc.get("regional_language_style") or cc.get("astra_regional_language_style"),
    }
    orig = getattr(svc, "push_error_frame", None)
    if orig is None:
        return svc
    state = {"last": None, "at": 0.0}

    async def push_error_frame(error, *args, **kwargs):  # noqa: ANN001
        try:
            msg = str(getattr(error, "error", "") or "")
            now = time.monotonic()
            if not (state["last"] == msg and now - state["at"] < 2.0):
                state["last"], state["at"] = msg, now
                logger.error("ASTRA_TTS_FAILURE " + json.dumps(build_record(ctx, msg), ensure_ascii=False))
        except Exception as exc:  # pragma: no cover
            logger.debug(f"ASTRA_TTS_FAILURE logging skipped: {exc}")
        return await orig(error, *args, **kwargs)

    try:
        svc.push_error_frame = push_error_frame
        svc._astra_tts_error_tagged = True
        svc._astra_tts_ctx = ctx
    except Exception as exc:  # pragma: no cover
        logger.debug(f"ASTRA_TTS_FAILURE tagging skipped: {exc}")
    return svc
