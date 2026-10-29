"""Astra per-call ACTUAL provider cost ledger (compute + persistence + read).

Platform-wide: works for every workflow / employee. Nothing here is specific to
any one agent.

Value provenance (every number carries a ``source``):
  provider_reported  the provider returned the value itself (OpenRouter usage.cost,
                     Deepgram result offsets / metadata duration, VoBiz BillDuration,
                     VoBiz TotalCost when non-zero)
  measured           counted by our runtime (characters sent to TTS, audio seconds
                     submitted to STT, tokens from Dograh usage metrics)
  rate_card          usage x configurable rate card (provider gave no cost)
  estimated          backfill / reconstruction for calls recorded before the ledger
  not_configured     no rate configured -> value is NULL, never guessed

Telephony cost is NEVER estimated: if the provider reports no cost and the
rate-card slot is empty, it stays NULL and totals are labelled as excluding
telephony.

All functions are best-effort and never raise to callers of the ``record_*`` /
``recompute`` entry points.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import math
import re
from datetime import UTC, date, datetime
from typing import Any, Iterable, Optional

from loguru import logger

LEDGER_VERSION = 2  # v2: telephony 60s round-up of billed seconds + configurable GST
TELEPHONY_MODES = {"ari", "plivo", "twilio", "vonage", "vobiz", "cloudonix", "telnyx"}
DEFAULT_TELEPHONY_CURRENCY = {"vobiz": "INR", "plivo": "USD", "twilio": "USD", "telnyx": "USD", "vonage": "USD"}

RATE_TABLE = "astra_provider_rate_cards"
LEDGER_TABLE = "astra_call_cost_ledger"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _r(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(x, nd)


def _as_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).date()
    except Exception:
        return None


def _loads(v, default):
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return default


_PROVIDER_BY_CLASS = [
    (re.compile(r"^openrouter", re.I), "openrouter"),
    (re.compile(r"^groq", re.I), "groq"),
    (re.compile(r"^openai", re.I), "openai"),
    (re.compile(r"^deepgram", re.I), "deepgram"),
    (re.compile(r"^sarvam", re.I), "sarvam"),
    (re.compile(r"^elevenlabs", re.I), "elevenlabs"),
    (re.compile(r"^cartesia", re.I), "cartesia"),
    (re.compile(r"^google|^dograhgoogle", re.I), "google"),
    (re.compile(r"^rumik", re.I), "rumik"),
    (re.compile(r"^dograh", re.I), "dograh"),
]


def provider_from_processor(processor: str) -> Optional[str]:
    for rx, name in _PROVIDER_BY_CLASS:
        if rx.search(processor or ""):
            return name
    return None


def split_key(key: str) -> tuple[str, Optional[str]]:
    if "|||" in key:
        p, m = key.split("|||", 1)
        return p, m
    return key, None


# ---------------------------------------------------------------------------
# rate cards
# ---------------------------------------------------------------------------


def find_rate(rate_cards: Iterable[dict], *, service: str, provider: Optional[str],
              model: Optional[str] = None, variant: Optional[str] = None,
              on: Optional[date] = None, organization_id: Optional[int] = None) -> Optional[dict]:
    """Most specific active rate card row for (service, provider, model, variant) on date."""
    best, best_score = None, None
    on = on or datetime.now(UTC).date()
    for rc in rate_cards or []:
        if not rc.get("is_active", True):
            continue
        if rc.get("service") != service or (rc.get("provider") or "") != (provider or ""):
            continue
        org = rc.get("organization_id")
        if org is not None and org != organization_id:
            continue
        ef, et = _as_date(rc.get("effective_from")), _as_date(rc.get("effective_to"))
        if ef and ef > on:
            continue
        if et and et <= on:
            continue
        score = []
        for want, have in ((model, rc.get("model") or "*"), (variant, rc.get("variant") or "*")):
            if have == "*":
                score.append(0)
            elif want is not None and have == want:
                score.append(2)
            elif want is not None and fnmatch.fnmatchcase(want, have):
                score.append(1)
            else:
                score = None
                break
        if score is None:
            continue
        key = (score[0], score[1], 1 if org is not None else 0, ef or date.min)
        if best_score is None or key > best_score:
            best, best_score = rc, key
    return best


def fx_from_rate_cards(rate_cards, on: Optional[date] = None) -> Optional[dict]:
    rc = find_rate(rate_cards, service="fx", provider="fx", model="USD_INR", on=on)
    if not rc:
        return None
    rate = _f((rc.get("rates") or {}).get("rate"))
    if rate is None:
        return None
    return {
        "pair": "USD_INR", "rate": rate,
        "as_of": str(_as_date(rc.get("effective_from")) or ""),
        "source": rc.get("source_url") or rc.get("notes"),
        "rate_card_id": rc.get("id"),
    }


def to_inr(amount: Optional[float], currency: Optional[str], fx: Optional[dict]) -> Optional[float]:
    if amount is None or currency is None:
        return None
    if currency == "INR":
        return amount
    if currency == "USD" and fx and fx.get("rate"):
        return amount * fx["rate"]
    return None


def gst_from_rate_cards(rate_cards, on: Optional[date] = None, organization_id: Optional[int] = None) -> Optional[dict]:
    """Optional GST config row: provider='tax', service='gst', model='*'.

    rates = {"rate": 0.18, "providers": {"sarvam": true, "vobiz": false, ...}, "default": false}
    No row (or rate missing) -> no GST: behaviour is identical to ledger v1.
    """
    rc = find_rate(rate_cards, service="gst", provider="tax", model="*", on=on, organization_id=organization_id)
    if not rc:
        return None
    rates = rc.get("rates") or {}
    rate = _f(rates.get("rate"))
    if rate is None or rate <= 0:
        return None
    return {"rate": rate, "providers": dict(rates.get("providers") or {}), "default": bool(rates.get("default", False)),
            "as_of": str(_as_date(rc.get("effective_from")) or ""), "rate_card_id": rc.get("id"),
            "note": rc.get("notes")}


def gst_rate_for(gst: Optional[dict], provider: Optional[str]) -> float:
    if not gst:
        return 0.0
    applies = gst["providers"].get(provider or "", gst["default"])
    return gst["rate"] if applies else 0.0


def apply_gst(sec: dict, gst: Optional[dict], provider: Optional[str] = None) -> dict:
    """Make sec['cost_inr'] GST-inclusive; keep the pre-tax value in cost_inr_ex_gst.

    Per item when the section has priced items (providers can differ), else by section provider.
    """
    if not gst or sec.get("cost_inr") is None:
        if sec.get("cost_inr") is not None:
            sec.setdefault("cost_inr_ex_gst", sec["cost_inr"])
            sec.setdefault("gst_inr", 0.0)
        return sec
    ex = float(sec["cost_inr"])
    items = [i for i in (sec.get("items") or []) if i.get("cost") is not None]
    if items:
        tax = 0.0
        for i in items:
            v = to_inr(i.get("cost"), i.get("currency"), sec.get("_fx"))
            if v is None:
                continue
            r = gst_rate_for(gst, i.get("provider"))
            i["gst_rate"] = r
            tax += v * r
    else:
        r = gst_rate_for(gst, provider or sec.get("provider"))
        sec["gst_rate"] = r
        tax = ex * r
    sec["cost_inr_ex_gst"] = _r(ex, 6)
    sec["gst_inr"] = _r(tax, 6)
    sec["cost_inr"] = _r(ex + tax, 6)
    return sec


def _rc_ref(rc: Optional[dict]) -> Optional[dict]:
    if not rc:
        return None
    return {k: rc.get(k) for k in ("id", "provider", "service", "model", "variant", "currency",
                                    "rates", "billing_increment_seconds")} | {
        "effective_from": str(_as_date(rc.get("effective_from")) or "")}


# ---------------------------------------------------------------------------
# section builders (pure)
# ---------------------------------------------------------------------------


def _combine_sources(items: list[dict], key: str = "cost_source") -> Optional[str]:
    s = {i.get(key) for i in items if i.get(key)}
    if not s:
        return None
    return s.pop() if len(s) == 1 else "mixed"


def _sum_native(items: list[dict]) -> dict:
    out: dict[str, float] = {}
    for i in items:
        if i.get("cost") is not None and i.get("currency"):
            out[i["currency"]] = out.get(i["currency"], 0.0) + i["cost"]
    return {k: _r(v, 8) for k, v in out.items()}


def _section_money(items: list[dict], fx) -> dict:
    native = _sum_native(items)
    inr_parts = [to_inr(i.get("cost"), i.get("currency"), fx) for i in items if i.get("cost") is not None]
    complete = all(i.get("cost") is not None for i in items)
    inr = sum(x for x in inr_parts if x is not None) if inr_parts else (0.0 if not items else None)
    if any(x is None for x in inr_parts):
        complete = False
    single = list(native.items())[0] if len(native) == 1 else None
    return {
        "cost_native": native,
        "cost": _r(single[1], 8) if single else (None if native else (0.0 if not items else None)),
        "currency": single[0] if single else ("MIXED" if len(native) > 1 else None),
        "cost_inr": _r(inr, 6) if inr is not None else None,
        "complete": complete,
        "cost_source": _combine_sources(items) or ("not_configured" if items else None),
    }


def llm_request_cost(ev: dict, rate_cards, on, org) -> dict:
    """One LLM request -> cost item (provider_reported preferred)."""
    item = {
        "provider": ev.get("provider"), "model": ev.get("requested_model") or ev.get("model"),
        "generation_id": ev.get("generation_id"),
        "input_tokens": int(ev.get("prompt_tokens") or 0),
        "cached_tokens": int(ev.get("cached_tokens") or 0),
        "output_tokens": int(ev.get("completion_tokens") or 0),
        "usage_source": "provider_reported",
    }
    if ev.get("reported_cost") is not None:
        item.update(cost=_f(ev["reported_cost"]), currency=ev.get("reported_cost_currency") or "USD",
                    cost_source="provider_reported")
        return item
    return _llm_rate_card(item, rate_cards, on, org, ev.get("model"))


def _llm_rate_card(item: dict, rate_cards, on, org, alt_model=None) -> dict:
    rc = find_rate(rate_cards, service="llm", provider=item.get("provider"), model=item.get("model"), on=on,
                   organization_id=org)
    if rc is None and alt_model and alt_model != item.get("model"):
        rc = find_rate(rate_cards, service="llm", provider=item.get("provider"), model=alt_model, on=on,
                       organization_id=org)
    rates = (rc or {}).get("rates") or {}
    pin, pout = _f(rates.get("input_per_1m")), _f(rates.get("output_per_1m"))
    pcache = _f(rates.get("cached_input_per_1m"))
    if rc is None or pin is None or pout is None:
        item.update(cost=None, currency=None, cost_source="not_configured")
        return item
    cached = item["cached_tokens"]
    uncached = max(0, item["input_tokens"] - cached)
    cost = (uncached * pin + cached * (pcache if pcache is not None else pin) + item["output_tokens"] * pout) / 1e6
    item.update(cost=cost, currency=rc.get("currency") or "USD", cost_source="rate_card", rate_card=_rc_ref(rc))
    return item


def _token_totals(items: list[dict]) -> dict:
    return {
        "input_tokens": sum(i.get("input_tokens", 0) for i in items),
        "cached_tokens": sum(i.get("cached_tokens", 0) for i in items),
        "output_tokens": sum(i.get("output_tokens", 0) for i in items),
        "requests": len(items),
    }


def build_llm_sections(usage_info: dict, runtime: dict, post_call_runtime: list, rate_cards, fx, on, org):
    services = (runtime or {}).get("services") or {}
    reqs = (runtime or {}).get("llm_requests") or []
    stream_reqs = [r for r in reqs if r.get("kind") == "stream"]
    oob_reqs = [r for r in reqs if r.get("kind") != "stream"]
    tapped_processors = {r.get("processor") for r in stream_reqs}

    conv_items = [llm_request_cost(r, rate_cards, on, org) for r in stream_reqs]
    qa_from_usage = []
    for key, u in ((usage_info or {}).get("llm") or {}).items():
        processor, model = split_key(key)
        u = u or {}
        if processor == "QAAnalysis":
            qa_from_usage.append((model, u))
            continue
        if processor in tapped_processors:
            continue  # provider-reported per-request data already covers this service
        meta = services.get(processor) or {}
        item = {
            "provider": meta.get("provider") or provider_from_processor(processor),
            "model": meta.get("model") or model,
            "input_tokens": int(u.get("prompt_tokens") or 0),
            "cached_tokens": int(u.get("cache_read_input_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
            "usage_source": "measured",
            "processor": processor,
        }
        conv_items.append(_llm_rate_card(item, rate_cards, on, org))

    post_items = []
    for r in oob_reqs:
        it = llm_request_cost(r, rate_cards, on, org)
        it["purpose"] = "variable_extraction_or_summary"
        post_items.append(it)
    qa_reqs = list(post_call_runtime or [])
    for r in qa_reqs:
        it = llm_request_cost(r, rate_cards, on, org)
        it["purpose"] = "qa_analysis"
        post_items.append(it)
    if not qa_reqs:
        for model, u in qa_from_usage:
            item = {
                "provider": None, "model": model, "purpose": "qa_analysis",
                "input_tokens": int(u.get("prompt_tokens") or 0),
                "cached_tokens": int(u.get("cache_read_input_tokens") or 0),
                "output_tokens": int(u.get("completion_tokens") or 0),
                "usage_source": "measured",
            }
            # QA provider is not in usage_info; try the conversation provider.
            provs = {i.get("provider") for i in conv_items if i.get("provider")}
            item["provider"] = provs.pop() if len(provs) == 1 else None
            post_items.append(_llm_rate_card(item, rate_cards, on, org))

    llm = {**_token_totals(conv_items), **_section_money(conv_items, fx),
           "usage_source": _combine_sources(conv_items, "usage_source"),
           "generation_ids": [i["generation_id"] for i in conv_items if i.get("generation_id")][:500],
           "items": conv_items[:200]}
    post = {**_token_totals(post_items), **_section_money(post_items, fx),
            "usage_source": _combine_sources(post_items, "usage_source"),
            "items": post_items[:200]}
    return llm, post


def config_hints(workflow_configurations) -> dict:
    """provider/model/language/voice per service from a workflow config (never keys)."""
    out: dict[str, dict] = {}
    wc = _loads(workflow_configurations, {})

    def walk(node, depth=0):
        if depth > 6 or not isinstance(node, dict):
            return
        for k, v in node.items():
            if k in ("stt", "tts", "llm") and isinstance(v, dict) and v.get("provider") and k not in out:
                out[k] = {f: v.get(f) for f in ("provider", "model", "language", "voice") if v.get(f) is not None}
            elif isinstance(v, dict):
                walk(v, depth + 1)

    walk(wc)
    return out


def build_stt_section(usage_info, runtime, rate_cards, fx, on, org, call_duration_s, backfill: bool,
                      hint: Optional[dict] = None):
    services = (runtime or {}).get("services") or {}
    stt_meta = [m for m in services.values() if m.get("service") == "stt"]
    meta = stt_meta[0] if stt_meta else (hint or {})
    measured = {k: float(v or 0) for k, v in ((usage_info or {}).get("stt") or {}).items()}
    measured_s = sum(measured.values()) if measured else None
    provider = meta.get("provider")
    model = meta.get("model")
    if not provider and measured:
        p, m = split_key(next(iter(measured)))
        provider, model = provider_from_processor(p), model or m
    streams = (runtime or {}).get("stt_streams") or {}
    prov_s = 0.0
    for s in streams.values():
        d = _f(s.get("metadata_duration"))
        if d is None:
            d = _f(s.get("max_result_end"))
        prov_s += d or 0.0
    if prov_s > 0:
        seconds, usage_source = prov_s, "provider_reported"
    elif measured_s:
        seconds, usage_source = measured_s, "measured"
    elif call_duration_s:
        seconds, usage_source = float(call_duration_s), "estimated"
    else:
        seconds, usage_source = None, None
    language = meta.get("language")
    variant = "multi" if language == "multi" else (language or None)
    rc = find_rate(rate_cards, service="stt", provider=provider, model=model, variant=variant, on=on,
                   organization_id=org) if provider else None
    per_min = _f(((rc or {}).get("rates") or {}).get("per_minute"))
    cost = currency = None
    cost_source = "not_configured"
    billable_s = seconds
    if seconds is not None and rc is not None and per_min is not None:
        inc = int(rc.get("billing_increment_seconds") or 1)
        billable_s = math.ceil(seconds / inc) * inc if inc > 1 else seconds
        cost, currency = billable_s / 60.0 * per_min, rc.get("currency") or "USD"
        cost_source = "estimated" if backfill else "rate_card"
    if seconds is None:
        cost_source = None
    return {
        "provider": provider, "model": model, "language": language,
        "seconds": _r(seconds, 3), "billable_seconds": _r(billable_s, 3),
        "usage_source": usage_source,
        "provider_reported_seconds": _r(prov_s, 3) if prov_s > 0 else None,
        "measured_seconds": _r(measured_s, 3) if measured_s else None,
        "deepgram_request_ids": [k for k in streams.keys() if k != "unknown"][:50],
        "cost": _r(cost, 8), "currency": currency, "cost_inr": _r(to_inr(cost, currency, fx)),
        "cost_source": cost_source, "rate_card": _rc_ref(rc),
    }


def build_tts_section(usage_info, runtime, tts_segments, rate_cards, fx, on, org, backfill: bool):
    services = (runtime or {}).get("services") or {}
    items = []
    for key, chars in ((usage_info or {}).get("tts") or {}).items():
        processor, model = split_key(key)
        meta = services.get(processor) or {}
        provider = meta.get("provider") or provider_from_processor(processor)
        mdl = meta.get("model") or model
        chars = int(chars or 0)
        rc = find_rate(rate_cards, service="tts", provider=provider, model=mdl, on=on, organization_id=org)
        per_1k = _f(((rc or {}).get("rates") or {}).get("per_1k_chars"))
        it = {"provider": provider, "model": mdl, "voice": meta.get("voice"), "characters": chars,
              "segments": (tts_segments or {}).get(key), "usage_source": "measured"}
        if rc is not None and per_1k is not None:
            it.update(cost=chars / 1000.0 * per_1k, currency=rc.get("currency") or "INR",
                      cost_source="estimated" if backfill else "rate_card", rate_card=_rc_ref(rc))
        else:
            it.update(cost=None, currency=None, cost_source="not_configured")
        items.append(it)
    money = _section_money(items, fx)
    return {"characters": sum(i["characters"] for i in items),
            "segments": sum(i["segments"] or 0 for i in items) if any(i.get("segments") for i in items) else None,
            **money, "usage_source": "measured" if items else None, "items": items}


def _callback_fields(cbs: list) -> dict:
    """Pick the final billing record out of provider status callbacks."""
    out: dict[str, Any] = {}
    for cb in cbs or []:
        if not isinstance(cb, dict):
            continue
        for src, dst in (("BillDuration", "bill_s"), ("bill_duration", "bill_s"), ("billed_duration", "bill_s"),
                         ("Duration", "duration_s"), ("duration", "duration_s"), ("CallDuration", "duration_s"),
                         ("TotalCost", "total_cost"), ("total_cost", "total_cost"),
                         ("BillRate", "bill_rate"), ("bill_rate", "bill_rate"),
                         ("Currency", "currency"), ("currency", "currency"),
                         ("CallUUID", "call_id"), ("call_id", "call_id")):
            v = cb.get(src)
            if v not in (None, ""):
                out[dst] = v
    return out


def build_telephony_section(mode, cbs, rate_cards, fx, on, org, call_duration_s):
    provider = (mode or "").lower() or None
    if provider not in TELEPHONY_MODES:
        return {"applicable": False, "provider": provider}
    f = _callback_fields(cbs)
    bill_s, dur_s = _f(f.get("bill_s")), _f(f.get("duration_s"))
    rc = find_rate(rate_cards, service="telephony", provider=provider, on=on, organization_id=org,
                   variant=None)
    rates = (rc or {}).get("rates") or {}
    per_min = _f(rates.get("per_minute"))
    inc = int((rc or {}).get("billing_increment_seconds") or 60)
    if bill_s is not None:
        # Carriers bill in pulses (VoBiz: full 60 s minutes, verified against CDR charges).
        # Round the provider-reported billed seconds up to the rate card's increment.
        billed = math.ceil(bill_s / inc) * inc if (inc > 1 and bill_s > 0) else bill_s
        minutes, usage_source = billed / 60.0, "provider_reported"
    elif dur_s is not None:
        minutes, usage_source = math.ceil(dur_s / inc) * inc / 60.0 if dur_s > 0 else 0.0, "derived_from_provider_duration"
    elif _f(call_duration_s) is not None:
        # Inbound VoBiz calls get no status callback (no BillDuration). Fall back to the
        # pipeline-measured call duration, rounded up to the billing increment. Labelled so
        # it is never mistaken for a provider-reported figure.
        cs = _f(call_duration_s)
        minutes, usage_source = math.ceil(cs / inc) * inc / 60.0 if cs > 0 else 0.0, "derived_from_call_duration"
    else:
        minutes, usage_source = None, None
    currency = (rc or {}).get("currency") or f.get("currency") or DEFAULT_TELEPHONY_CURRENCY.get(provider)
    total_cost, bill_rate = _f(f.get("total_cost")), _f(f.get("bill_rate"))
    cost, cost_source = None, "not_configured"
    note = None
    if total_cost is not None and total_cost > 0:
        cost, cost_source = total_cost, "provider_reported"
    elif bill_rate is not None and bill_rate > 0 and minutes is not None:
        cost, cost_source = bill_rate * minutes, "provider_reported"
    elif per_min is not None and minutes is not None:
        cost, cost_source = per_min * minutes, "rate_card"
    else:
        note = (f"{provider} reported no cost (TotalCost={f.get('total_cost')!r}, BillRate={f.get('bill_rate')!r}) "
                "and no telephony rate is configured; telephony cost is not estimated.")
    return {
        "applicable": True, "provider": provider, "call_id": f.get("call_id"),
        "provider_duration_seconds": dur_s, "billable_seconds": bill_s,
        "billable_minutes": _r(minutes, 4), "usage_source": usage_source,
        "provider_total_cost": total_cost, "provider_bill_rate": bill_rate,
        "cost": _r(cost, 6), "currency": currency if cost is not None else currency,
        "cost_inr": _r(to_inr(cost, currency, fx)), "cost_source": cost_source,
        "rate_card": _rc_ref(rc), "note": note,
    }


def compute_breakdown(*, run: dict, raw: dict, rate_cards: list[dict], backfill: bool = False) -> dict:
    """Pure function: run row + raw captured usage + rate cards -> cost record."""
    usage_info = _loads(run.get("usage_info"), {}) or {}
    logs = _loads(run.get("logs"), {}) or {}
    runtime = (raw or {}).get("runtime") or {}
    post_rt = (raw or {}).get("post_call_llm_requests") or []
    tts_segments = (raw or {}).get("tts_segments") or {}
    on = _as_date(run.get("created_at")) or datetime.now(UTC).date()
    org = run.get("organization_id")
    fx = fx_from_rate_cards(rate_cards, on)
    call_s = _f(usage_info.get("call_duration_seconds"))

    llm, post = build_llm_sections(usage_info, runtime, post_rt, rate_cards, fx, on, org)
    if backfill:
        for sec in (llm, post):
            for it in sec["items"]:
                if it.get("cost_source") == "rate_card":
                    it["cost_source"] = "estimated"
            if sec.get("cost_source") == "rate_card":
                sec["cost_source"] = "estimated"
    hints = config_hints(run.get("workflow_configurations"))
    stt = build_stt_section(usage_info, runtime, rate_cards, fx, on, org, call_s, backfill, hints.get("stt"))
    tts = build_tts_section(usage_info, runtime, tts_segments, rate_cards, fx, on, org, backfill)
    tel = build_telephony_section(run.get("mode"), logs.get("telephony_status_callbacks") or [],
                                  rate_cards, fx, on, org, call_s)

    gst = gst_from_rate_cards(rate_cards, on, org)
    for sec, prov in ((llm, None), (post, None), (tts, None), (stt, stt.get("provider")), (tel, tel.get("provider"))):
        if isinstance(sec, dict) and (sec.get("applicable", True)):
            sec["_fx"] = fx
            apply_gst(sec, gst, prov)
            sec.pop("_fx", None)
    parts = {"llm": llm, "stt": stt, "tts": tts, "post_call_processing": post}
    if tel.get("applicable"):
        parts["telephony"] = tel
    total_inr = 0.0
    total_ex_gst = 0.0
    total_gst = 0.0
    missing = []
    native_totals: dict[str, float] = {}
    for name, sec in parts.items():
        has_usage = not (name in ("llm", "post_call_processing") and sec.get("requests", 0) == 0) and not (
            name == "tts" and not sec.get("characters")) and not (name == "stt" and sec.get("seconds") is None)
        if name == "telephony":
            has_usage = True
        if not has_usage:
            continue
        if sec.get("cost_inr") is None:
            missing.append(name)
            continue
        if sec.get("complete") is False:
            missing.append(name)  # partially priced: provider-reported parts still counted
        total_inr += sec["cost_inr"]
        total_ex_gst += sec.get("cost_inr_ex_gst", sec["cost_inr"]) or 0.0
        total_gst += sec.get("gst_inr") or 0.0
        if isinstance(sec.get("cost_native"), dict):
            for c, v in sec["cost_native"].items():
                native_totals[c] = native_totals.get(c, 0.0) + (v or 0.0)
        elif sec.get("cost") is not None and sec.get("currency"):
            native_totals[sec["currency"]] = native_totals.get(sec["currency"], 0.0) + sec["cost"]

    excludes_tel = bool(tel.get("applicable")) and tel.get("cost_inr") is None
    minutes_basis = tel.get("provider_duration_seconds") if tel.get("applicable") else None
    duration_s = _f(minutes_basis) or call_s
    per_min = (total_inr / (duration_s / 60.0)) if duration_s and duration_s > 0 else None
    label = "Total infrastructure cost"
    if gst:
        label += f" (incl. {round(gst['rate'] * 100)}% GST)"
    if excludes_tel:
        label += f" (excluding telephony: {tel.get('provider')} rate not configured)"
    other_missing = [m for m in missing if m != "telephony"]
    if other_missing:
        label += f" (incomplete: no rate for {', '.join(other_missing)})"

    return {
        "version": LEDGER_VERSION,
        "workflow_run_id": run.get("id"), "workflow_id": run.get("workflow_id"), "mode": run.get("mode"),
        "call_started_at": str(run.get("created_at") or ""),
        "call_duration_seconds": call_s, "duration_basis_seconds": duration_s,
        "backfill": backfill,
        "fx": fx,
        "gst": gst,
        "llm": llm, "stt": stt, "tts": tts, "telephony": tel, "post_call_processing": post,
        "totals": {
            "total_infrastructure_cost_inr": _r(total_inr, 4),
            "total_ex_gst_inr": _r(total_ex_gst, 4),
            "gst_inr": _r(total_gst, 4),
            "native": {k: _r(v, 8) for k, v in native_totals.items()},
            "excludes_telephony": excludes_tel,
            "incomplete_components": missing,
            "label": label,
            "actual_cost_per_minute_inr": _r(per_min, 4),
            "cost_per_minute_label": ("per minute of call (" + ("excluding telephony" if excludes_tel else
                                                                "including telephony") + ")"),
        },
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# persistence (raw SQL; tables created by deploy SQL, not alembic)
# ---------------------------------------------------------------------------


async def _session():
    from api.db import db_client
    return db_client.async_session()


async def _fetch(sql: str, params: dict) -> list[dict]:
    from sqlalchemy import text
    async with await _session() as s:
        res = await s.execute(text(sql), params)
        rows = res.mappings().all()
        return [dict(r) for r in rows]


async def _exec(sql: str, params: dict) -> None:
    from sqlalchemy import text
    async with await _session() as s:
        await s.execute(text(sql), params)
        await s.commit()


async def load_rate_cards() -> list[dict]:
    rows = await _fetch(f"select * from {RATE_TABLE} where is_active", {})
    for r in rows:
        r["rates"] = _loads(r.get("rates"), {})
    return rows


async def load_run(run_id: int) -> Optional[dict]:
    rows = await _fetch(
        "select r.id, r.workflow_id, r.mode, r.created_at, r.usage_info, r.logs, w.organization_id, "
        "d.workflow_configurations from workflow_runs r join workflows w on w.id = r.workflow_id "
        "left join workflow_definitions d on d.id = r.definition_id where r.id = :id",
        {"id": int(run_id)},
    )
    return rows[0] if rows else None


async def load_raw(run_id: int) -> Optional[dict]:
    rows = await _fetch(f"select raw_usage from {LEDGER_TABLE} where workflow_run_id = :id", {"id": int(run_id)})
    return _loads(rows[0]["raw_usage"], {}) if rows else None


async def _merge_raw(run_id: int, patch: dict) -> None:
    await _exec(
        f"insert into {LEDGER_TABLE} (workflow_run_id, raw_usage, status) values (:id, cast(:p as jsonb), 'partial') "
        f"on conflict (workflow_run_id) do update set raw_usage = {LEDGER_TABLE}.raw_usage || excluded.raw_usage, "
        "updated_at = now()",
        {"id": int(run_id), "p": json.dumps(patch, default=str)},
    )


def _flat(b: dict) -> dict:
    def sec(name):
        return b.get(name) or {}
    llm, stt, tts, tel, post, tot = sec("llm"), sec("stt"), sec("tts"), sec("telephony"), sec("post_call_processing"), sec("totals")
    return {
        "workflow_id": b.get("workflow_id"),
        "llm_input_tokens": llm.get("input_tokens"), "llm_cached_tokens": llm.get("cached_tokens"),
        "llm_output_tokens": llm.get("output_tokens"), "llm_cost_native": llm.get("cost"),
        "llm_cost_currency": llm.get("currency"), "llm_cost_inr": llm.get("cost_inr"), "llm_cost_source": llm.get("cost_source"),
        "stt_seconds": stt.get("seconds"), "stt_usage_source": stt.get("usage_source"), "stt_cost_native": stt.get("cost"),
        "stt_cost_currency": stt.get("currency"), "stt_cost_inr": stt.get("cost_inr"), "stt_cost_source": stt.get("cost_source"),
        "tts_characters": tts.get("characters"), "tts_cost_native": tts.get("cost"), "tts_cost_currency": tts.get("currency"),
        "tts_cost_inr": tts.get("cost_inr"), "tts_cost_source": tts.get("cost_source"),
        "telephony_billable_minutes": tel.get("billable_minutes") if tel.get("applicable") else None,
        "telephony_cost_native": tel.get("cost") if tel.get("applicable") else None,
        "telephony_cost_currency": tel.get("currency") if tel.get("applicable") else None,
        "telephony_cost_inr": tel.get("cost_inr") if tel.get("applicable") else None,
        "telephony_cost_source": tel.get("cost_source") if tel.get("applicable") else "not_applicable",
        "post_call_cost_native": post.get("cost"), "post_call_cost_currency": post.get("currency"),
        "post_call_cost_inr": post.get("cost_inr"), "post_call_cost_source": post.get("cost_source"),
        "total_infra_cost_inr": tot.get("total_infrastructure_cost_inr"),
        "total_excludes_telephony": tot.get("excludes_telephony"),
        "cost_per_minute_inr": tot.get("actual_cost_per_minute_inr"),
        "call_duration_seconds": b.get("call_duration_seconds"),
        "fx_usd_inr": (b.get("fx") or {}).get("rate"), "fx_as_of": (b.get("fx") or {}).get("as_of") or None,
    }


_FLAT_COLS = list(_flat({}).keys())


async def recompute(run_id: int, *, final: bool = False) -> Optional[dict]:
    """Recompute and store the breakdown from stored raw usage + run row. Idempotent."""
    try:
        run = await load_run(run_id)
        if not run:
            return None
        raw = await load_raw(run_id) or {}
        rate_cards = await load_rate_cards()
        b = compute_breakdown(run=run, raw=raw, rate_cards=rate_cards, backfill=False)
        flat = _flat(b)
        if flat["fx_as_of"]:
            flat["fx_as_of"] = _as_date(flat["fx_as_of"])
        cols = ", ".join(_FLAT_COLS)
        vals = ", ".join(f":{c}" for c in _FLAT_COLS)
        upd = ", ".join(f"{c} = excluded.{c}" for c in _FLAT_COLS)
        params = {**flat, "id": int(run_id), "b": json.dumps(b, default=str), "org": run.get("organization_id"),
                  "status": "final" if final else "partial"}
        await _exec(
            f"insert into {LEDGER_TABLE} (workflow_run_id, organization_id, breakdown, status, computed_at, {cols}) "
            f"values (:id, :org, cast(:b as jsonb), :status, now(), {vals}) "
            f"on conflict (workflow_run_id) do update set organization_id = excluded.organization_id, "
            f"breakdown = excluded.breakdown, computed_at = now(), updated_at = now(), "
            f"status = case when {LEDGER_TABLE}.status = 'final' then 'final' else excluded.status end, {upd}",
            params,
        )
        return b
    except Exception as e:
        logger.warning(f"cost_ledger: recompute failed for run {run_id}: {e}")
        return None


async def record_pipeline_end(workflow_run_id: int, pipeline_metrics_aggregator=None) -> None:
    """Called once after the call ends (pipeline process). Best-effort, never raises."""
    try:
        from api.services.pipecat import cost_ledger
        col = cost_ledger.pop_collector(workflow_run_id)
        patch: dict = {"version": LEDGER_VERSION, "captured_at": datetime.now(UTC).isoformat()}
        if col is not None:
            patch["runtime"] = col.snapshot()
        try:
            if pipeline_metrics_aggregator is not None and hasattr(pipeline_metrics_aggregator, "get_tts_request_counts"):
                patch["tts_segments"] = dict(pipeline_metrics_aggregator.get_tts_request_counts())
        except Exception:
            pass
        await _merge_raw(workflow_run_id, patch)
        await recompute(workflow_run_id)
    except Exception as e:
        logger.warning(f"cost_ledger: record_pipeline_end failed for run {workflow_run_id}: {e}")


async def record_post_call(workflow_run_id: int) -> None:
    """Called by the post-call job after QA/integrations. Best-effort, never raises."""
    try:
        from api.services.pipecat import cost_ledger
        col = cost_ledger.pop_collector(workflow_run_id)
        if col is not None and col.llm_requests:
            existing = (await load_raw(workflow_run_id) or {}).get("post_call_llm_requests") or []
            await _merge_raw(workflow_run_id, {"post_call_llm_requests": existing + col.snapshot()["llm_requests"]})
        await recompute(workflow_run_id, final=True)
    except Exception as e:
        logger.warning(f"cost_ledger: record_post_call failed for run {workflow_run_id}: {e}")


async def safe(coro, timeout: float = 8.0) -> None:
    try:
        await asyncio.wait_for(coro, timeout=timeout)
    except Exception as e:
        logger.warning(f"cost_ledger: skipped ({type(e).__name__}: {e})")


_LLM_REQ_KEYS = ("ts", "request_ts", "kind", "model", "prompt_tokens", "cached_tokens", "completion_tokens",
                 "reasoning_tokens", "ttft_ms", "latency_ms", "generation_id", "reported_cost",
                 "reasoning_effort", "prompt_cache_key", "temperature")


async def _attach_llm_requests(run_id: int, b: dict) -> None:
    """Per-generation usage (no secrets) for the console's Last call panel."""
    try:
        raw = await load_raw(run_id) or {}
        rt = raw.get("runtime") or {}
        b["llm_requests"] = [{k: r.get(k) for k in _LLM_REQ_KEYS if k in r} for r in (rt.get("llm_requests") or [])]
        b["llm_cancelled"] = [{k: r.get(k) for k in _LLM_REQ_KEYS if k in r} for r in (rt.get("llm_cancelled") or [])]
    except Exception as e:
        logger.debug(f"cost_ledger: llm_requests attach skipped for run {run_id}: {e}")


async def get_call_cost(run_id: int) -> Optional[dict]:
    """Read API: stored ledger if present, else an on-the-fly *estimated* backfill (not stored)."""
    rows = await _fetch(
        f"select breakdown, status, computed_at, is_backfill, "
        f"(computed_at is null or computed_at < (select max(updated_at) from {RATE_TABLE})) as stale "
        f"from {LEDGER_TABLE} where workflow_run_id = :id",
        {"id": int(run_id)},
    )
    if rows and rows[0].get("stale") and not rows[0].get("is_backfill"):
        # Rate card edited after this call was priced (e.g. VoBiz rate configured):
        # re-price from the stored raw provider usage. Provider-reported values are unchanged.
        fresh = await recompute(run_id, final=rows[0].get("status") == "final")
        if fresh:
            fresh["ledger_status"] = rows[0].get("status")
            fresh["stored"] = True
            await _attach_llm_requests(run_id, fresh)
            return fresh
    if rows and _loads(rows[0].get("breakdown"), {}):
        b = _loads(rows[0]["breakdown"], {})
        b["ledger_status"] = rows[0].get("status")
        b["stored"] = True
        await _attach_llm_requests(run_id, b)
        return b
    run = await load_run(run_id)
    if not run:
        return None
    b = compute_breakdown(run=run, raw={}, rate_cards=await load_rate_cards(), backfill=True)
    b["ledger_status"] = "not_recorded"
    b["stored"] = False
    b["note"] = ("Call predates the cost ledger: usage from Dograh usage_info, costs are estimated from the rate "
                 "card (not provider-reported).")
    return b


async def recompute_if_exists(run_id: int) -> None:
    """Telephony callback hook: refresh only if the call already has a ledger row."""
    try:
        rows = await _fetch(f"select status from {LEDGER_TABLE} where workflow_run_id = :id", {"id": int(run_id)})
        if rows:
            await recompute(run_id, final=rows[0].get("status") == "final")
    except Exception as e:
        logger.warning(f"cost_ledger: telephony refresh failed for run {run_id}: {e}")
