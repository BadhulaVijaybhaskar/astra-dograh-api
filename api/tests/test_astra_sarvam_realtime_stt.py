import json
import urllib.parse

import pytest

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language

from api.services.pipecat.sarvam_realtime_stt import (
    SarvamRealtimeSTTService,
    build_realtime_query,
    is_realtime_model,
)


def test_realtime_models_routed():
    assert is_realtime_model("saaras:v4")
    assert is_realtime_model("saaras:v3-realtime")
    assert not is_realtime_model("saaras:v3")
    assert not is_realtime_model("saarika:v2.5")
    assert not is_realtime_model(None)


def test_query_codemix_no_keyterms():
    q = dict(build_realtime_query(model="saaras:v4", language_code="te-IN", sample_rate=16000,
                                  mode="codemix", silence_duration_ms=350))
    assert q["model"] == "saaras:v4" and q["language_code"] == "te-IN"
    assert q["mode"] == "codemix" and q["silence_duration_ms"] == "350"
    assert q["endpointing"] == "vad" and q["encoding"] == "linear16" and q["sample_rate"] == "16000"
    assert "keyterms" not in q


def test_query_defaults_omit_optional():
    q = dict(build_realtime_query(model="saaras:v4", language_code="unknown", sample_rate=8000))
    assert "language_code" not in q and "mode" not in q and "silence_duration_ms" not in q


def test_query_keyterms_only_when_given():
    q = dict(build_realtime_query(model="saaras:v4", language_code="te-IN", sample_rate=16000,
                                  keyterms=["Zoho CRM"]))
    assert json.loads(q["keyterms"]) == ["Zoho CRM"]


def _svc(**kw):
    s = SarvamRealtimeSTTService(api_key="test", model="saaras:v4", language_code="te-IN",
                                 mode="codemix", silence_duration_ms=350, sample_rate=16000, **kw)
    pushed = []

    async def fake_push(frame, direction=None):
        pushed.append(frame)

    async def noop(*a, **k):
        return None

    s.push_frame = fake_push
    s.emit_stt_usage_metrics = noop
    s.stop_processing_metrics = noop
    return s, pushed


def test_url_contains_settings_not_key():
    s, _ = _svc()
    s._sample_rate = 16000
    url = s._build_url()
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert q["silence_duration_ms"] == "350" and q["mode"] == "codemix"
    assert "test" not in url  # api key goes in a header, never the URL


def test_ttfs_tracks_silence():
    s, _ = _svc()
    assert abs(s._ttfs_p99_latency - 1.25) < 1e-6


@pytest.mark.asyncio
async def test_final_is_finalized_transcription():
    s, pushed = _svc()
    await s.handle_message({"event": "transcript.final", "text": "మా Zoho CRM తో integrate అవుతుందా?"})
    assert len(pushed) == 1 and isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].finalized is True and pushed[0].language == Language.TE_IN
    assert pushed[0].text.startswith("మా Zoho CRM")


@pytest.mark.asyncio
async def test_partial_is_interim_and_empty_ignored():
    s, pushed = _svc()
    await s.handle_message({"event": "transcript.partial", "text": "మా Zoho"})
    await s.handle_message({"event": "transcript.final", "text": "  "})
    await s.handle_message({"event": "vad.speech_start"})
    assert len(pushed) == 1 and isinstance(pushed[0], InterimTranscriptionFrame)
    assert not isinstance(pushed[0], TranscriptionFrame)


def test_registry_accepts_realtime_fields():
    from api.services.configuration.registry import SarvamSTTConfiguration

    c = SarvamSTTConfiguration(model="saaras:v4", language="te-IN", api_key="x",
                               mode="codemix", silence_duration_ms=350)
    assert c.mode == "codemix" and c.silence_duration_ms == 350
    with pytest.raises(Exception):
        SarvamSTTConfiguration(model="saaras:v4", language="te-IN", api_key="x", silence_duration_ms=10)


def test_interim_finalizer_skipped_for_guaranteed_finals():
    from api.services.pipecat.pipeline_builder import interim_finalizer_needed

    s, _ = _svc()
    assert interim_finalizer_needed(s) is False

    class Other:
        pass

    assert interim_finalizer_needed(Other()) is True


@pytest.mark.asyncio
async def test_reassert_after_non_speech_vad_blip():
    import asyncio as _a
    s, pushed = _svc(); s.reassert_grace_s = 0.01
    s.on_vad_started()
    await s.handle_message({"event": "transcript.final", "text": "మా Zoho CRM తో integrate అవుతుందా?"})
    s.on_vad_stopped()            # normal stop: STT produced text in this segment -> no marker
    await _a.sleep(0.05)
    assert len(pushed) == 1
    s.on_vad_started(); s.on_vad_stopped()   # blip with no STT output
    await _a.sleep(0.05)
    assert len(pushed) == 2 and pushed[1].finalized and pushed[1].text.strip() == ""


@pytest.mark.asyncio
async def test_no_reassert_when_caller_resumes_or_turn_closed():
    import asyncio as _a
    s, pushed = _svc(); s.reassert_grace_s = 0.05
    await s.handle_message({"event": "transcript.final", "text": "మా sales team"})
    s.on_vad_started(); s.on_vad_stopped()
    await s.handle_message({"event": "transcript.partial", "text": "Salesforce"})  # speech arrived in grace
    await _a.sleep(0.1)
    assert [type(f).__name__ for f in pushed] == ["TranscriptionFrame", "InterimTranscriptionFrame"]
    s.on_user_turn_stopped(); s.on_vad_started(); s.on_vad_stopped()   # new turn, no final yet
    await _a.sleep(0.1)
    assert len(pushed) == 2
