import asyncio
import json

from api.services.pipecat import tts_error_log as t


class _Err:
    def __init__(self, error):
        self.error = error


class _Svc:
    def __init__(self):
        self.pushed = []

    async def push_error_frame(self, error):
        self.pushed.append(error)


class _TTS:
    provider = "sarvam"
    model = "bulbul:v4-flash"
    voice = "kavya"
    language = "te-IN"


class _UC:
    tts = _TTS()


RAW = "TTS Error: 400: Speaker 'kavya' is not compatible with model bulbul:v4-flash. Available speakers for bulbul:v4-flash are: a, b"


def test_classify_and_status():
    assert t.classify_tts_error(RAW) == "INCOMPATIBLE_SPEAKER"
    assert t.http_status_from(RAW) == 400
    assert t.classify_tts_error("something else") == "TTS_FAILED"


def test_customer_message():
    assert t.customer_message("INCOMPATIBLE_SPEAKER", "kavya", "bulbul:v4-flash", "te-IN") == (
        "Voice Kavya isn't compatible with Telugu / Bulbul v4 Flash. Choose one of the available Telugu voices."
    )


def test_tag_logs_structured_record_once_and_passes_through(monkeypatch):
    lines = []
    monkeypatch.setattr(t.logger, "error", lambda m: lines.append(m))
    svc = _Svc()
    t.tag_tts_service(svc, user_config=_UC(), workflow_id=9, workflow_run_id=255,
                      call_context={"employee_id": "emp_x", "regional_language_style": "telangana"})
    asyncio.run(svc.push_error_frame(_Err(RAW)))
    asyncio.run(svc.push_error_frame(_Err(RAW)))
    assert len(svc.pushed) == 2
    assert len(lines) == 1
    rec = json.loads(lines[0].split(" ", 1)[1])
    assert rec["error_code"] == "INCOMPATIBLE_SPEAKER"
    assert rec["http_status"] == 400
    assert rec["employee_id"] == "emp_x" and rec["workflow_id"] == 9 and rec["call_id"] == 255
    assert rec["model"] == "bulbul:v4-flash" and rec["speaker"] == "kavya" and rec["language"] == "te-IN"
    assert rec["regional_style"] == "telangana"
    assert len(rec["provider_error"]) <= 300
    assert "timestamp" in rec
