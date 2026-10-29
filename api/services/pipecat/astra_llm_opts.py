"""Astra per-employee LLM request options (thinking effort, prompt caching).

Set from the definition's model_configuration_v2_override.byok.pipeline.llm
(`reasoning_effort`, `prompt_cache`) and applied to every OpenAI-compatible
chat.completions request made by that LLM service (see cost_ledger tap).

- Thinking: OpenRouter `reasoning: {effort}` for reasoning-capable models only.
  Out-of-band calls (memory/variable extraction, non-streaming) always use
  `minimal` (or `none` when the employee chose none), unless the After-call card
  sets `summary_reasoning_effort` / `summary_temperature`.
- Prompt caching: a stable `prompt_cache_key` derived from the static head of
  the system prompt, and the per-call memory block is sent as a trailing
  system message instead of being rewritten into the system prompt, so the
  cached prefix (system prompt + history) stays identical turn to turn.

Nothing here runs unless the definition carries one of the two fields, so
definitions published before this change behave exactly as before.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Optional

from loguru import logger

EFFORTS = ("none", "minimal", "low", "medium", "high")
_REASONING_RE = re.compile(
    r"(gpt-5|(^|/)o[134](-|$)|deepseek-r1|reasoner|thinking|gpt-oss|claude-(sonnet|opus)-4|"
    r"claude-[34]\.7|gemini-2\.5|gemini-3|grok-[34]|qwen3|magistral)",
    re.I,
)
_CACHE_HEAD_CHARS = 3000


def is_reasoning_model(model: Optional[str]) -> bool:
    return bool(model) and bool(_REASONING_RE.search(str(model)))


def attach(svc: Any, llm_cfg: Any) -> None:
    """Read the two Astra fields off the LLM config and remember them on the service."""
    try:
        effort = getattr(llm_cfg, "reasoning_effort", None)
        cache = getattr(llm_cfg, "prompt_cache", None)
        temp = getattr(llm_cfg, "temperature", None)
        try:
            temp = None if temp is None else max(0.0, min(1.5, round(float(temp), 2)))
        except (TypeError, ValueError):
            temp = None
        s_eff = getattr(llm_cfg, "summary_reasoning_effort", None)
        s_temp = getattr(llm_cfg, "summary_temperature", None)
        try:
            s_temp = None if s_temp is None else max(0.0, min(1.5, round(float(s_temp), 2)))
        except (TypeError, ValueError):
            s_temp = None
        if effort is None and cache is None and temp is None and s_eff is None and s_temp is None:
            return
        effort = str(effort).strip().lower() if effort is not None else None
        if effort not in EFFORTS:
            effort = None
        s_eff = str(s_eff).strip().lower() if s_eff is not None else None
        if s_eff not in EFFORTS:
            s_eff = None
        svc._astra_llm_opts = {"reasoning_effort": effort, "prompt_cache": bool(cache), "temperature": temp,
                               "summary_reasoning_effort": s_eff, "summary_temperature": s_temp}
        logger.info(f"ASTRA_LLM_OPTS reasoning_effort={effort} prompt_cache={bool(cache)} temperature={temp} "
                    f"summary_reasoning_effort={s_eff} summary_temperature={s_temp}")
    except Exception as exc:  # never break service creation
        logger.debug(f"ASTRA_LLM_OPTS attach skipped: {exc}")


def opts(svc: Any) -> Optional[dict]:
    o = getattr(svc, "_astra_llm_opts", None)
    return o if isinstance(o, dict) else None


def tail_mode(svc: Any) -> bool:
    o = opts(svc)
    return bool(o and o.get("prompt_cache"))


def set_tail_block(svc: Any, fn: Callable[[], str]) -> None:
    svc._astra_tail_block = fn


def _system_text(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def apply(svc: Any, kwargs: dict) -> dict:
    """Mutate request kwargs in place; returns a small summary for the ledger."""
    o = opts(svc)
    if not o:
        return {}
    info: dict = {}
    stream = bool(kwargs.get("stream"))
    model = str(kwargs.get("model") or "")
    extra_body = dict(kwargs.get("extra_body") or {})
    # Temperature: employee value on replies only; extraction (non-stream) keeps the service default (0.1).
    if stream and o.get("temperature") is not None:
        kwargs["temperature"] = o["temperature"]
    # After-call card: background (non-stream) calls use the summary temperature only when set.
    if not stream and o.get("summary_temperature") is not None:
        kwargs["temperature"] = o["summary_temperature"]
    if isinstance(kwargs.get("temperature"), (int, float)):
        info["temperature"] = kwargs.get("temperature")
    if is_reasoning_model(model):
        eff = o.get("reasoning_effort")
        if not stream:
            # Forced minimal (or none) only when the After-call card has no Thinking set.
            eff = o.get("summary_reasoning_effort") or ("none" if eff == "none" else "minimal")
        if eff:
            extra_body["reasoning"] = {"effort": eff}
            info["reasoning_effort"] = eff
    if o.get("prompt_cache"):
        msgs = list(kwargs.get("messages") or [])
        head = _system_text(msgs)[:_CACHE_HEAD_CHARS]
        if head:
            key = "astra-" + hashlib.sha1(head.encode("utf-8")).hexdigest()[:24]
            extra_body["prompt_cache_key"] = key
            info["prompt_cache_key"] = key
        fn = getattr(svc, "_astra_tail_block", None)
        if stream and callable(fn):
            try:
                block = fn()
            except Exception as exc:
                block = ""
                logger.debug(f"ASTRA_LLM_OPTS tail block skipped: {exc}")
            if block:
                msgs.append({"role": "system", "content": block})
                kwargs["messages"] = msgs
                info["tail_block"] = True
    if extra_body:
        kwargs["extra_body"] = extra_body
    return info
