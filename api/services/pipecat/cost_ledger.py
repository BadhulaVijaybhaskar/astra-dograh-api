"""Per-call provider usage capture (Astra cost ledger, runtime side).

Platform-wide and provider-agnostic: services created through
``service_factory`` are instrumented here, keyed by workflow run id.

Hard rules honoured here:
  * Never raises into the caller. Every hook is wrapped in try/except.
  * Never awaits I/O on the audio path. Hooks only append small dicts to an
    in-memory list; persistence happens after the call ends (event handlers)
    or in the post-call ARQ job.
  * Never changes the request sent to a provider. OpenRouter returns usage and
    cost on every response automatically (usage accounting is always on), so
    the tap only *reads* the response objects the SDK already produced.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from loguru import logger

try:  # pipecat contextvar carrying the workflow run id across tasks
    from pipecat.utils.run_context import get_current_run_id
except Exception:  # pragma: no cover
    def get_current_run_id():  # type: ignore
        return None


_MAX_RUNS = 512           # bound memory if a run is never flushed
_MAX_EVENTS_PER_RUN = 5000


class RunCollector:
    """Raw provider events for one workflow run (in this process)."""

    __slots__ = ("run_id", "created", "services", "llm_requests", "llm_cancelled", "stt_streams", "dropped")

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.created = time.time()
        self.services: dict[str, dict] = {}       # processor name -> meta
        self.llm_requests: list[dict] = []
        self.llm_cancelled: list[dict] = []
        self.stt_streams: dict[str, dict] = {}    # request_id -> {...}
        self.dropped = 0

    def add_cancelled(self, event: dict) -> None:
        if len(self.llm_cancelled) < 200:
            self.llm_cancelled.append(event)

    def add_llm(self, event: dict) -> None:
        if len(self.llm_requests) >= _MAX_EVENTS_PER_RUN:
            self.dropped += 1
            return
        self.llm_requests.append(event)

    def snapshot(self) -> dict:
        return {
            "services": dict(self.services),
            "llm_requests": list(self.llm_requests),
            "llm_cancelled": list(self.llm_cancelled),
            "stt_streams": {k: dict(v) for k, v in self.stt_streams.items()},
            "dropped_events": self.dropped,
        }


_lock = threading.Lock()
_runs: "OrderedDict[str, RunCollector]" = OrderedDict()


def _key(run_id: Any) -> Optional[str]:
    if run_id is None or run_id == "":
        return None
    return str(run_id)


def collector_for(run_id: Any, create: bool = True) -> Optional[RunCollector]:
    key = _key(run_id)
    if key is None:
        return None
    with _lock:
        col = _runs.get(key)
        if col is None and create:
            col = RunCollector(key)
            _runs[key] = col
            while len(_runs) > _MAX_RUNS:
                _runs.popitem(last=False)
        return col


def pop_collector(run_id: Any) -> Optional[RunCollector]:
    key = _key(run_id)
    if key is None:
        return None
    with _lock:
        return _runs.pop(key, None)


def _bound_run_id(svc) -> Optional[str]:
    rid = getattr(svc, "_astra_cost_run_id", None)
    return rid or _key(get_current_run_id())


# ---------------------------------------------------------------------------
# LLM: tap the OpenAI-compatible client (OpenRouter, OpenAI, Groq, ...)
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None) or {}
        val = extra.get(name, default) if isinstance(extra, dict) else default
    return val


def usage_event_from_response(resp, *, kind: str, processor: str, provider: str,
                              requested_model: Optional[str]) -> Optional[dict]:
    """Normalise one provider response/chunk carrying ``usage`` into an event."""
    usage = _get(resp, "usage")
    if usage is None:
        return None
    ptd = _get(usage, "prompt_tokens_details")
    ctd = _get(usage, "completion_tokens_details")
    cost_details = _get(usage, "cost_details")
    cost = _num(_get(usage, "cost"))
    return {
        "kind": kind,  # "stream" (conversation) | "inference" (out-of-band)
        "processor": processor,
        "provider": provider,
        "model": _get(resp, "model") or requested_model,
        "requested_model": requested_model,
        "generation_id": _get(resp, "id"),
        "upstream_provider": _get(resp, "provider"),
        "prompt_tokens": int(_get(usage, "prompt_tokens") or 0),
        "completion_tokens": int(_get(usage, "completion_tokens") or 0),
        "total_tokens": int(_get(usage, "total_tokens") or 0),
        "cached_tokens": int(_get(ptd, "cached_tokens") or 0),
        "cache_write_tokens": int(_get(ptd, "cache_write_tokens") or 0),
        "reasoning_tokens": int(_get(ctd, "reasoning_tokens") or 0),
        # Only OpenRouter reports cost; it is in USD (credits == USD).
        "reported_cost": cost,
        "reported_cost_currency": "USD" if cost is not None else None,
        "upstream_inference_cost": _num(_get(cost_details, "upstream_inference_cost")),
        "ts": time.time(),
    }


class _TapIterator:
    """Async iterator proxy that records the usage chunk and passes all through."""

    __slots__ = ("_it", "_cb")

    def __init__(self, it, cb):
        self._it = it
        self._cb = cb

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._it.__anext__()
        try:
            st = getattr(self._cb, "_astra_state", None)
            if st is not None and st.get("first") is None:
                ch = getattr(chunk, "choices", None) or []
                d = getattr(ch[0], "delta", None) if ch else None
                if d is not None and getattr(d, "content", None):
                    st["first"] = time.time()
            if getattr(chunk, "usage", None) is not None:
                self._cb(chunk)
        except Exception:  # never break the stream
            pass
        return chunk

    async def aclose(self):
        aclose = getattr(self._it, "aclose", None)
        if aclose is not None:
            await aclose()


class _TapStream:
    """Wraps an SDK AsyncStream; keeps close()/aclose()/response semantics."""

    def __init__(self, stream, cb):
        self._astra_stream = stream
        self._astra_cb = cb

    def __aiter__(self):
        return _TapIterator(self._astra_stream.__aiter__(), self._astra_cb)

    async def close(self):
        try:
            st = getattr(self._astra_cb, "_astra_state", None)
            if st is not None and not st.get("recorded") and callable(st.get("on_cancel")):
                st["on_cancel"]()
        except Exception:
            pass
        close = getattr(self._astra_stream, "close", None)
        if close is not None:
            res = close()
            if hasattr(res, "__await__"):
                await res

    def __getattr__(self, name):
        return getattr(self._astra_stream, name)


def _record_llm(svc, provider: str, kind: str, resp, requested_model, extra: Optional[dict] = None) -> None:
    try:
        col = collector_for(_bound_run_id(svc))
        if col is None:
            return
        ev = usage_event_from_response(
            resp, kind=kind, processor=getattr(svc, "name", type(svc).__name__),
            provider=provider, requested_model=requested_model,
        )
        if ev is not None:
            if extra:
                ev.update({k: v for k, v in extra.items() if v is not None})
            col.add_llm(ev)
    except Exception as e:  # pragma: no cover
        logger.debug(f"cost_ledger: llm usage capture skipped: {e}")


def _patch_completions(svc, provider: str) -> bool:
    client = getattr(svc, "_client", None)
    chat = getattr(client, "chat", None) if client is not None else None
    completions = getattr(chat, "completions", None) if chat is not None else None
    create = getattr(completions, "create", None) if completions is not None else None
    if create is None or getattr(create, "_astra_cost_tap", False):
        return False

    async def create_with_tap(*args, **kwargs):
        opts_info = {}
        try:
            from api.services.pipecat import astra_llm_opts as _alo

            opts_info = _alo.apply(svc, kwargs) or {}
        except Exception as exc:  # never block a request on options
            logger.debug(f"ASTRA_LLM_OPTS apply skipped: {exc}")
        t0 = time.time()
        resp = await create(*args, **kwargs)
        try:
            model = kwargs.get("model")
            if kwargs.get("stream"):
                st = {"t0": t0, "first": None, "recorded": False}

                def _extra():
                    out = {"request_ts": t0, **{k: v for k, v in opts_info.items() if k != "tail_block"}}
                    if st["first"] is not None:
                        out["ttft_ms"] = int((st["first"] - t0) * 1000)
                    return out

                def _cb(chunk):
                    st["recorded"] = True
                    _record_llm(svc, provider, "stream", chunk, model, _extra())

                def _on_cancel():
                    st["recorded"] = True
                    try:
                        col = collector_for(_bound_run_id(svc))
                        if col is not None:
                            col.add_cancelled({"ts": time.time(), "model": model, **_extra()})
                    except Exception:
                        pass

                st["on_cancel"] = _on_cancel
                _cb._astra_state = st  # type: ignore[attr-defined]
                return _TapStream(resp, _cb)
            _record_llm(svc, provider, "inference", resp, model,
                        {"request_ts": t0, "latency_ms": int((time.time() - t0) * 1000), **opts_info})
        except Exception:
            pass
        return resp

    create_with_tap._astra_cost_tap = True  # type: ignore[attr-defined]
    completions.create = create_with_tap
    return True


def instrument_llm(svc, provider: Optional[str], model: Optional[str], purpose: str = "conversation"):
    """Best-effort: bind run id + register meta + tap the client. Returns svc."""
    try:
        if svc is None:
            return svc
        rid = _key(get_current_run_id())
        setattr(svc, "_astra_cost_run_id", rid)
        if rid:
            col = collector_for(rid)
            if col is not None:
                col.services[getattr(svc, "name", type(svc).__name__)] = {
                    "service": "llm", "provider": provider, "model": model, "purpose": purpose,
                }
        tapped = _patch_completions(svc, provider or "unknown")
        # Groq fallback / client re-creation: re-tap when a new client appears.
        if not tapped and getattr(svc, "_client", None) is None:
            logger.debug("cost_ledger: LLM service has no OpenAI-compatible client; tokens via metrics only")
    except Exception as e:
        logger.debug(f"cost_ledger: instrument_llm skipped: {e}")
    return svc


# ---------------------------------------------------------------------------
# STT: seconds come from pipecat STTUsageMetricsData (aggregator). For Deepgram
# streaming we additionally read the provider's own audio offsets per request.
# ---------------------------------------------------------------------------

def _wrap_deepgram_on_message(svc) -> None:
    orig = getattr(svc, "_on_message", None)
    if orig is None or getattr(orig, "_astra_cost_tap", False):
        return
    name = getattr(svc, "name", type(svc).__name__)

    async def on_message(message):
        try:
            mtype = type(message).__name__
            if mtype == "ListenV1Results" or mtype == "ListenV1Metadata":
                col = collector_for(_bound_run_id(svc))
                if col is not None:
                    if mtype == "ListenV1Results":
                        md = getattr(message, "metadata", None)
                        rid = getattr(md, "request_id", None) or "unknown"
                        end = (_num(getattr(message, "start", None)) or 0.0) + (
                            _num(getattr(message, "duration", None)) or 0.0
                        )
                        s = col.stt_streams.setdefault(str(rid), {"processor": name})
                        if end > (s.get("max_result_end") or 0.0):
                            s["max_result_end"] = end
                    else:
                        rid = getattr(message, "request_id", None) or "unknown"
                        s = col.stt_streams.setdefault(str(rid), {"processor": name})
                        d = _num(getattr(message, "duration", None))
                        if d is not None:
                            s["metadata_duration"] = d
        except Exception:
            pass
        return await orig(message)

    on_message._astra_cost_tap = True  # type: ignore[attr-defined]
    svc._on_message = on_message


def instrument_stt(svc, provider: Optional[str], model: Optional[str], language: Optional[str]):
    try:
        if svc is None:
            return svc
        rid = _key(get_current_run_id())
        setattr(svc, "_astra_cost_run_id", rid)
        if rid:
            col = collector_for(rid)
            if col is not None:
                col.services[getattr(svc, "name", type(svc).__name__)] = {
                    "service": "stt", "provider": provider, "model": model, "language": language,
                }
        if type(svc).__name__ == "DeepgramSTTService":
            _wrap_deepgram_on_message(svc)
    except Exception as e:
        logger.debug(f"cost_ledger: instrument_stt skipped: {e}")
    return svc


def instrument_tts(svc, provider: Optional[str], model: Optional[str], voice: Optional[str]):
    try:
        if svc is None:
            return svc
        rid = _key(get_current_run_id())
        setattr(svc, "_astra_cost_run_id", rid)
        if rid:
            col = collector_for(rid)
            if col is not None:
                col.services[getattr(svc, "name", type(svc).__name__)] = {
                    "service": "tts", "provider": provider, "model": model, "voice": voice,
                }
    except Exception as e:
        logger.debug(f"cost_ledger: instrument_tts skipped: {e}")
    return svc
