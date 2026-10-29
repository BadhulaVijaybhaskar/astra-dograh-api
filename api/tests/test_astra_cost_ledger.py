"""Astra per-call cost ledger: recorded/mock provider responses only (no network)."""

import asyncio
import json
from datetime import datetime, UTC

import pytest

from api.services import call_cost as cc
from api.services.pipecat import cost_ledger as cl

RATE_CARDS = [
    {"id": 1, "provider": "openrouter", "service": "llm", "model": "openai/gpt-4.1-mini", "variant": "*",
     "currency": "USD", "rates": {"input_per_1m": 0.40, "cached_input_per_1m": 0.10, "output_per_1m": 1.60},
     "effective_from": "2026-01-01", "is_active": True},
    {"id": 2, "provider": "deepgram", "service": "stt", "model": "nova-3*", "variant": "multi", "currency": "USD",
     "rates": {"per_minute": 0.0058}, "billing_increment_seconds": 1, "effective_from": "2026-01-01", "is_active": True},
    {"id": 3, "provider": "deepgram", "service": "stt", "model": "nova-3*", "variant": "*", "currency": "USD",
     "rates": {"per_minute": 0.0048}, "billing_increment_seconds": 1, "effective_from": "2026-01-01", "is_active": True},
    {"id": 4, "provider": "sarvam", "service": "tts", "model": "bulbul:v3", "variant": "*", "currency": "INR",
     "rates": {"per_1k_chars": 3.0}, "effective_from": "2026-01-01", "is_active": True},
    {"id": 5, "provider": "sarvam", "service": "tts", "model": "bulbul:*", "variant": "*", "currency": "INR",
     "rates": {"per_1k_chars": 3.0}, "effective_from": "2026-01-01", "is_active": True},
    {"id": 6, "provider": "vobiz", "service": "telephony", "model": "*", "variant": "*", "currency": "INR",
     "rates": {"per_minute": None}, "billing_increment_seconds": 60, "effective_from": "2026-01-01", "is_active": True},
    {"id": 7, "provider": "fx", "service": "fx", "model": "USD_INR", "variant": "*", "currency": "INR",
     "rates": {"rate": 96.78}, "effective_from": "2026-10-08", "is_active": True},
]

# Recorded shape of OpenRouter streaming usage chunk (usage accounting always on).
OR_USAGE_CHUNK = {
    "id": "gen-1760000000-abcDEF", "object": "chat.completion.chunk", "created": 1760000000,
    "model": "openai/gpt-4.1-mini", "provider": "OpenAI", "choices": [],
    "usage": {"prompt_tokens": 4156, "completion_tokens": 127, "total_tokens": 4283,
              "prompt_tokens_details": {"cached_tokens": 1024, "audio_tokens": 0},
              "completion_tokens_details": {"reasoning_tokens": 0},
              "cost": 0.0018456, "is_byok": False, "cost_details": {"upstream_inference_cost": None}},
}
OR_TEXT_CHUNK = {
    "id": "gen-1760000000-abcDEF", "object": "chat.completion.chunk", "created": 1760000000,
    "model": "openai/gpt-4.1-mini",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
}
OR_NONSTREAM = {
    "id": "gen-1760000001-xyz", "object": "chat.completion", "created": 1760000001, "model": "openai/gpt-4.1-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"name\": \"Ravi\"}"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920,
              "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.000392},
}


def _chunk(d):
    from openai.types.chat import ChatCompletionChunk
    return ChatCompletionChunk.model_validate(d)


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False
        self.iter_closed = False

    def __aiter__(self):
        outer = self

        class It:
            def __init__(self):
                self.i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.i >= len(outer._chunks):
                    raise StopAsyncIteration
                self.i += 1
                return outer._chunks[self.i - 1]

            async def aclose(self):
                outer.iter_closed = True

        return It()

    async def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, stream_chunks=None, nonstream=None, exc=None):
        self.stream_chunks, self.nonstream, self.exc = stream_chunks, nonstream, exc
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        if kwargs.get("stream"):
            self.last_stream = FakeAsyncStream(self.stream_chunks)
            return self.last_stream
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(self.nonstream)


class FakeClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {})()
        self.chat.completions = completions


class FakeLLM:
    name = "OpenRouterLLMService#0"

    def __init__(self, completions):
        self._client = FakeClient(completions)


@pytest.fixture(autouse=True)
def _clean_registry():
    cl._runs.clear()
    yield
    cl._runs.clear()


def _with_run(run_id):
    from pipecat.utils.run_context import set_current_run_id
    set_current_run_id(run_id)


async def test_llm_stream_tap_records_provider_cost_and_passes_chunks_unchanged():
    _with_run("9001")
    comp = FakeCompletions(stream_chunks=[_chunk(OR_TEXT_CHUNK), _chunk(OR_USAGE_CHUNK)])
    svc = cl.instrument_llm(FakeLLM(comp), "openrouter", "openai/gpt-4.1-mini")
    params = {"model": "openai/gpt-4.1-mini", "stream": True, "stream_options": {"include_usage": True},
              "messages": [{"role": "user", "content": "hi"}]}
    stream = await svc._client.chat.completions.create(**params)
    # request is NOT modified by the tap
    assert comp.calls[0] == params
    got = []
    it = stream.__aiter__()
    async for ch in it:
        got.append(ch)
    await it.aclose()
    await stream.close()
    assert [c.id for c in got] == ["gen-1760000000-abcDEF"] * 2
    assert got[0].choices[0].delta.content == "Hello"
    assert comp.last_stream.closed and comp.last_stream.iter_closed
    ev = cl.collector_for("9001").llm_requests
    assert len(ev) == 1
    e = ev[0]
    assert e["kind"] == "stream" and e["generation_id"] == "gen-1760000000-abcDEF"
    assert e["prompt_tokens"] == 4156 and e["cached_tokens"] == 1024 and e["completion_tokens"] == 127
    assert e["reported_cost"] == pytest.approx(0.0018456) and e["reported_cost_currency"] == "USD"
    assert e["upstream_provider"] == "OpenAI"


async def test_llm_nonstream_inference_is_recorded_as_out_of_band():
    _with_run("9002")
    comp = FakeCompletions(nonstream=OR_NONSTREAM)
    svc = cl.instrument_llm(FakeLLM(comp), "openrouter", "openai/gpt-4.1-mini")
    resp = await svc._client.chat.completions.create(model="openai/gpt-4.1-mini", stream=False, messages=[])
    assert resp.choices[0].message.content.startswith("{")
    e = cl.collector_for("9002").llm_requests[0]
    assert e["kind"] == "inference" and e["reported_cost"] == pytest.approx(0.000392)


async def test_provider_error_propagates_unchanged_402():
    _with_run("9003")
    err = RuntimeError("402 Payment Required")
    svc = cl.instrument_llm(FakeLLM(FakeCompletions(exc=err)), "openrouter", "m")
    with pytest.raises(RuntimeError, match="402"):
        await svc._client.chat.completions.create(model="m", stream=True, messages=[])
    assert cl.collector_for("9003").llm_requests == []


async def test_tap_callback_failure_never_breaks_stream(monkeypatch):
    _with_run("9004")
    monkeypatch.setattr(cl, "usage_event_from_response", lambda *a, **k: 1 / 0)
    comp = FakeCompletions(stream_chunks=[_chunk(OR_TEXT_CHUNK), _chunk(OR_USAGE_CHUNK)])
    svc = cl.instrument_llm(FakeLLM(comp), "openrouter", "m")
    stream = await svc._client.chat.completions.create(model="m", stream=True, messages=[])
    n = 0
    async for _ in stream:
        n += 1
    assert n == 2


async def test_instrument_is_idempotent_and_survives_bad_service():
    _with_run("9005")
    comp = FakeCompletions(nonstream=OR_NONSTREAM)
    svc = FakeLLM(comp)
    cl.instrument_llm(svc, "openrouter", "m")
    first = svc._client.chat.completions.create
    cl.instrument_llm(svc, "openrouter", "m")
    assert svc._client.chat.completions.create is first
    assert cl.instrument_llm(object(), "x", "y") is not None
    assert cl.instrument_llm(None, "x", "y") is None


async def test_real_openrouter_service_run_inference_and_stream_are_tapped():
    from pipecat.processors.aggregators.llm_context import LLMContext
    from api.services.pipecat import service_factory as sf

    _with_run("9010")
    svc = sf.create_llm_service_from_provider("openrouter", "openai/gpt-4.1-mini", "test-key-not-used")
    assert type(svc).__name__ == "OpenRouterLLMService"
    col = cl.collector_for("9010")
    assert col.services[svc.name]["provider"] == "openrouter"
    # swap the *underlying* SDK create (beneath the tap) for a recorded response
    tapped = svc._client.chat.completions.create
    fake = FakeCompletions(stream_chunks=[_chunk(OR_TEXT_CHUNK), _chunk(OR_USAGE_CHUNK)], nonstream=OR_NONSTREAM)
    cl._patch_completions  # noqa
    # rebuild tap around the fake to emulate the real client call path
    svc._client.chat.completions.create = fake.create
    assert cl._patch_completions(svc, "openrouter")
    ctx = LLMContext()
    ctx.set_messages([{"role": "user", "content": "hi"}])
    out = await svc.run_inference(ctx)
    assert out == "{\"name\": \"Ravi\"}"
    stream = await svc.get_chat_completions(ctx)
    async for _ in stream:
        pass
    kinds = [e["kind"] for e in col.llm_requests]
    assert kinds == ["inference", "stream"]
    assert "usage" not in fake.calls[0]  # no request parameter added
    assert callable(tapped)


async def test_factory_wrapper_returns_service_even_if_instrumentation_fails(monkeypatch):
    from api.services.pipecat import service_factory as sf

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(cl, "instrument_llm", boom)
    svc = sf.create_llm_service_from_provider("openrouter", "openai/gpt-4.1-mini", "k")
    assert type(svc).__name__ == "OpenRouterLLMService"


async def test_deepgram_offsets_and_metadata_are_captured():
    from deepgram.listen.v1.types import ListenV1Metadata, ListenV1Results

    class DeepgramSTTService:  # name-matched dummy (real class needs a socket)
        name = "DeepgramSTTService#0"

        def __init__(self):
            self.seen = 0

        async def _on_message(self, message):
            self.seen += 1

    _with_run("9020")
    svc = cl.instrument_stt(DeepgramSTTService(), "deepgram", "nova-3-general", "multi")

    def res(start, dur, rid="req-A"):
        return ListenV1Results.model_validate({
            "type": "Results", "channel_index": [0, 1], "duration": dur, "start": start, "is_final": True,
            "speech_final": True, "channel": {"alternatives": [{"transcript": "", "confidence": 0.0, "words": []}]},
            "metadata": {"request_id": rid, "model_info": {"name": "x", "version": "1", "arch": "a"}, "model_uuid": "u"},
        })

    await svc._on_message(res(0.0, 2.5))
    await svc._on_message(res(2.5, 3.0))
    await svc._on_message(res(0.0, 1.0, "req-B"))
    await svc._on_message(ListenV1Metadata.model_validate({
        "type": "Metadata", "transaction_key": "t", "request_id": "req-B", "sha256": "s",
        "created": "2026-10-09T00:00:00Z", "duration": 1.75, "channels": 1}))
    assert svc.seen == 4
    st = cl.collector_for("9020").stt_streams
    assert st["req-A"]["max_result_end"] == pytest.approx(5.5)
    assert st["req-B"]["metadata_duration"] == pytest.approx(1.75)


async def test_metrics_aggregator_collects_stt_seconds_and_tts_segments():
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData, TTSUsageMetricsData
    from api.services.pipecat.pipeline_metrics_aggregator import PipelineMetricsAggregator

    agg = PipelineMetricsAggregator()
    agg.push_frame = _noop_push  # type: ignore
    from pipecat.processors.frame_processor import FrameDirection
    import pipecat.processors.frame_processor as fp

    async def _sup(self, frame, direction):
        return None

    orig = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = _sup
    try:
        for s in (1.5, 2.25):
            await agg.process_frame(MetricsFrame(data=[STTUsageMetricsData(
                processor="DeepgramSTTService#0", model="nova-3-general", value=STTUsage(audio_seconds=s))]),
                FrameDirection.DOWNSTREAM)
        for n in (100, 40):
            await agg.process_frame(MetricsFrame(data=[TTSUsageMetricsData(
                processor="SarvamTTSService#0", model="bulbul:v3", value=n)]), FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = orig
    u = agg.get_all_usage_metrics_serialized()
    assert u["stt"] == {"DeepgramSTTService#0|||nova-3-general": 3.75}
    assert u["tts"] == {"SarvamTTSService#0|||bulbul:v3": 140}
    assert agg.get_tts_request_counts() == {"SarvamTTSService#0|||bulbul:v3": 2}


async def _noop_push(*a, **k):
    return None


# ---------------------------------------------------------------- compute


def _run237(**over):
    run = {
        "id": 237, "workflow_id": 8, "mode": "vobiz", "organization_id": 1,
        "created_at": datetime(2026, 10, 8, 16, 55, tzinfo=UTC),
        "usage_info": {"llm": {"OpenRouterLLMService#0|||openai/gpt-4.1-mini": {
            "prompt_tokens": 4156, "completion_tokens": 127, "total_tokens": 4283,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}},
            "tts": {"SarvamTTSService#0|||bulbul:v3": 540},
            "stt": {"DeepgramSTTService#0|||nova-3-general": 86.2},
            "call_duration_seconds": 85},
        "logs": {"telephony_status_callbacks": [{
            "status": "completed", "call_id": "84a6", "duration": "87", "BillDuration": "120", "BillRate": "0",
            "Duration": "87", "Event": "Hangup", "TotalCost": "0.00000", "CallUUID": "84a6"}]},
    }
    run.update(over)
    return run


def _raw_live():
    return {"runtime": {
        "services": {
            "OpenRouterLLMService#0": {"service": "llm", "provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "DeepgramSTTService#0": {"service": "stt", "provider": "deepgram", "model": "nova-3-general", "language": "multi"},
            "SarvamTTSService#0": {"service": "tts", "provider": "sarvam", "model": "bulbul:v3", "voice": "priya"}},
        "llm_requests": [
            {"kind": "stream", "processor": "OpenRouterLLMService#0", "provider": "openrouter",
             "requested_model": "openai/gpt-4.1-mini", "model": "openai/gpt-4.1-mini", "generation_id": "gen-1",
             "prompt_tokens": 2000, "cached_tokens": 0, "completion_tokens": 60, "reported_cost": 0.000896,
             "reported_cost_currency": "USD"},
            {"kind": "stream", "processor": "OpenRouterLLMService#0", "provider": "openrouter",
             "requested_model": "openai/gpt-4.1-mini", "model": "openai/gpt-4.1-mini", "generation_id": "gen-2",
             "prompt_tokens": 2156, "cached_tokens": 1024, "completion_tokens": 67, "reported_cost": 0.0006592,
             "reported_cost_currency": "USD"},
            {"kind": "inference", "processor": "OpenRouterLLMService#0", "provider": "openrouter",
             "requested_model": "openai/gpt-4.1-mini", "model": "openai/gpt-4.1-mini", "generation_id": "gen-3",
             "prompt_tokens": 900, "cached_tokens": 0, "completion_tokens": 20, "reported_cost": 0.000392,
             "reported_cost_currency": "USD"}],
        "stt_streams": {"req-A": {"processor": "DeepgramSTTService#0", "max_result_end": 85.9}}},
        "tts_segments": {"SarvamTTSService#0|||bulbul:v3": 9}}


def test_compute_live_call_actuals_and_telephony_not_configured():
    b = cc.compute_breakdown(run=_run237(), raw=_raw_live(), rate_cards=RATE_CARDS)
    llm = b["llm"]
    assert (llm["input_tokens"], llm["cached_tokens"], llm["output_tokens"], llm["requests"]) == (4156, 1024, 127, 2)
    assert llm["cost_source"] == "provider_reported" and llm["currency"] == "USD"
    assert llm["cost"] == pytest.approx(0.0015552)
    assert llm["cost_inr"] == pytest.approx(0.0015552 * 96.78, abs=1e-5)
    assert llm["generation_ids"] == ["gen-1", "gen-2"]
    post = b["post_call_processing"]
    assert post["cost"] == pytest.approx(0.000392) and post["cost_source"] == "provider_reported"
    assert post["items"][0]["purpose"] == "variable_extraction_or_summary"
    stt = b["stt"]
    assert stt["seconds"] == pytest.approx(85.9) and stt["usage_source"] == "provider_reported"
    assert stt["measured_seconds"] == pytest.approx(86.2)
    assert stt["cost"] == pytest.approx(85.9 / 60 * 0.0058) and stt["cost_source"] == "rate_card"
    tts = b["tts"]
    assert tts["characters"] == 540 and tts["segments"] == 9
    assert tts["cost"] == pytest.approx(1.62) and tts["currency"] == "INR" and tts["cost_source"] == "rate_card"
    tel = b["telephony"]
    assert tel["billable_minutes"] == pytest.approx(2.0) and tel["usage_source"] == "provider_reported"
    assert tel["cost"] is None and tel["cost_inr"] is None and tel["cost_source"] == "not_configured"
    t = b["totals"]
    assert t["excludes_telephony"] is True and "excluding telephony" in t["label"]
    expected = (0.0015552 + 0.000392 + 85.9 / 60 * 0.0058) * 96.78 + 1.62
    assert t["total_infrastructure_cost_inr"] == pytest.approx(expected, rel=1e-4)
    assert t["actual_cost_per_minute_inr"] == pytest.approx(expected / (87 / 60), rel=1e-3)
    assert t["native"]["INR"] == pytest.approx(1.62) and t["native"]["USD"] > 0
    assert b["fx"]["rate"] == 96.78 and b["fx"]["as_of"] == "2026-10-08"
    json.dumps(b, default=str)


def test_telephony_rate_configured_and_provider_cost_paths():
    cards = [dict(c) for c in RATE_CARDS]
    for c in cards:
        if c["service"] == "telephony":
            c["rates"] = {"per_minute": 0.45}
    b = cc.compute_breakdown(run=_run237(), raw=_raw_live(), rate_cards=cards)
    assert b["telephony"]["cost"] == pytest.approx(0.90) and b["telephony"]["cost_source"] == "rate_card"
    assert b["totals"]["excludes_telephony"] is False
    run = _run237()
    run["logs"]["telephony_status_callbacks"][0]["TotalCost"] = "1.10"
    b2 = cc.compute_breakdown(run=run, raw=_raw_live(), rate_cards=RATE_CARDS)
    assert b2["telephony"]["cost"] == pytest.approx(1.10) and b2["telephony"]["cost_source"] == "provider_reported"
    assert b2["telephony"]["currency"] == "INR"


def test_missing_provider_cost_falls_back_to_rate_card_then_not_configured():
    raw = _raw_live()
    for r in raw["runtime"]["llm_requests"]:
        r["reported_cost"] = None
    b = cc.compute_breakdown(run=_run237(), raw=raw, rate_cards=RATE_CARDS)
    exp = ((2000 * 0.40) + (2156 - 1024) * 0.40 + 1024 * 0.10 + (60 + 67) * 1.60) / 1e6
    assert b["llm"]["cost"] == pytest.approx(exp) and b["llm"]["cost_source"] == "rate_card"
    cards = [c for c in RATE_CARDS if c["service"] != "llm"]
    b2 = cc.compute_breakdown(run=_run237(), raw=raw, rate_cards=cards)
    assert b2["llm"]["cost"] is None and b2["llm"]["cost_source"] == "not_configured"
    assert "llm" in b2["totals"]["incomplete_components"] and "incomplete" in b2["totals"]["label"]


def test_backfill_is_marked_estimated_never_actual():
    run = _run237(usage_info={**_run237()["usage_info"], "stt": {}})
    run["workflow_configurations"] = {"model_configuration_v2_override": {"x": {"stt": {
        "provider": "deepgram", "model": "nova-3-general", "language": "multi", "api_key": "SHOULD-NOT-LEAK"}}}}
    b = cc.compute_breakdown(run=run, raw={}, rate_cards=RATE_CARDS, backfill=True)
    assert "SHOULD-NOT-LEAK" not in json.dumps(b, default=str)
    assert b["stt"]["seconds"] == 85 and b["stt"]["cost"] == pytest.approx(85 / 60 * 0.0058)
    assert b["backfill"] is True
    assert b["llm"]["cost_source"] == "estimated"
    assert all(i["cost_source"] == "estimated" for i in b["llm"]["items"])
    assert b["stt"]["usage_source"] == "estimated" and b["stt"]["cost_source"] == "estimated"
    assert b["tts"]["cost_source"] == "estimated"
    assert b["telephony"]["cost_source"] == "not_configured"
    # only VoBiz BillDuration (a genuine provider value) may be provider_reported in a backfill
    assert b["telephony"]["usage_source"] == "provider_reported"


def test_telugu_monolingual_stt_and_v4flash_tts_use_generic_cards():
    raw = _raw_live()
    raw["runtime"]["services"]["DeepgramSTTService#0"]["language"] = "te"
    raw["runtime"]["services"]["SarvamTTSService#0"]["model"] = "bulbul:v4-flash"
    run = _run237(workflow_id=9)
    run["usage_info"] = {**run["usage_info"], "tts": {"SarvamTTSService#0|||bulbul:v4-flash": 300}}
    b = cc.compute_breakdown(run=run, raw=raw, rate_cards=RATE_CARDS)
    assert b["stt"]["rate_card"]["variant"] == "*" and b["stt"]["cost"] == pytest.approx(85.9 / 60 * 0.0048)
    assert b["tts"]["items"][0]["rate_card"]["model"] == "bulbul:*" and b["tts"]["cost"] == pytest.approx(0.9)


def test_web_call_has_no_telephony_and_no_label():
    b = cc.compute_breakdown(run=_run237(mode="smallwebrtc", logs={}), raw=_raw_live(), rate_cards=RATE_CARDS)
    assert b["telephony"]["applicable"] is False
    assert b["totals"]["excludes_telephony"] is False and "excluding" not in b["totals"]["label"]


def test_qa_tokens_from_usage_info_are_post_call_not_conversation():
    run = _run237()
    run["usage_info"]["llm"]["QAAnalysis|||openai/gpt-4.1-mini"] = {"prompt_tokens": 1000, "completion_tokens": 50}
    b = cc.compute_breakdown(run=run, raw=_raw_live(), rate_cards=RATE_CARDS)
    assert b["llm"]["requests"] == 2
    qa = [i for i in b["post_call_processing"]["items"] if i["purpose"] == "qa_analysis"]
    assert qa and qa[0]["cost_source"] == "rate_card" and qa[0]["provider"] == "openrouter"
    assert b["post_call_processing"]["cost_source"] == "mixed"


def test_rate_card_effective_dates_and_specificity():
    cards = RATE_CARDS + [{"id": 99, "provider": "fx", "service": "fx", "model": "USD_INR", "variant": "*",
                           "currency": "INR", "rates": {"rate": 100.0}, "effective_from": "2026-11-01", "is_active": True}]
    from datetime import date
    assert cc.fx_from_rate_cards(cards, date(2026, 10, 9))["rate"] == 96.78
    assert cc.fx_from_rate_cards(cards, date(2026, 11, 2))["rate"] == 100.0
    assert cc.find_rate(cards, service="stt", provider="deepgram", model="nova-3-general", variant="multi")["id"] == 2


async def test_record_entry_points_never_raise(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(cc, "_merge_raw", boom)
    monkeypatch.setattr(cc, "_fetch", boom)
    await cc.record_pipeline_end(1, None)
    await cc.record_post_call(1)
    await cc.recompute_if_exists(1)
    assert await cc.recompute(1) is None

    async def slow():
        await asyncio.sleep(5)

    t0 = asyncio.get_event_loop().time()
    await cc.safe(slow(), timeout=0.05)
    assert asyncio.get_event_loop().time() - t0 < 1.0


async def test_record_pipeline_end_persists_runtime_snapshot(monkeypatch):
    _with_run("9100")
    comp = FakeCompletions(stream_chunks=[_chunk(OR_USAGE_CHUNK)])
    svc = cl.instrument_llm(FakeLLM(comp), "openrouter", "openai/gpt-4.1-mini")
    s = await svc._client.chat.completions.create(model="openai/gpt-4.1-mini", stream=True, messages=[])
    async for _ in s:
        pass
    saved = {}

    async def merge(run_id, patch):
        saved.update(patch)

    async def rec(run_id, final=False):
        saved["recomputed"] = final

    monkeypatch.setattr(cc, "_merge_raw", merge)
    monkeypatch.setattr(cc, "recompute", rec)
    await cc.record_pipeline_end(9100, None)
    assert saved["runtime"]["llm_requests"][0]["generation_id"] == "gen-1760000000-abcDEF"
    assert saved["recomputed"] is False
    assert cl.collector_for("9100", create=False) is None  # popped


async def test_read_reprices_when_rate_card_changed_after_pricing(monkeypatch):
    calls = {"recompute": 0}

    async def fetch(sql, params):
        assert "stale" in sql
        return [{"breakdown": {"totals": {}}, "status": "final", "computed_at": None, "is_backfill": False,
                 "stale": True}]

    async def rec(run_id, final=False):
        calls["recompute"] += 1
        assert final is True
        return {"totals": {"excludes_telephony": False}}

    monkeypatch.setattr(cc, "_fetch", fetch)
    monkeypatch.setattr(cc, "recompute", rec)
    b = await cc.get_call_cost(5)
    assert calls["recompute"] == 1 and b["stored"] is True and b["totals"]["excludes_telephony"] is False
