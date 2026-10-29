"""Per-call caller-facts store + AI-employee identity (platform-wide, config driven).

Opt-in per workflow via ``workflow_configurations.caller_facts`` (written by the
console when an employee is activated):

    "caller_facts": {"enabled": true, "fields": [{"key", "label", "type", "question"}...]}
    "astra_employee": {"id": "...", "name": "<employee.name from the console>"}

* The store keeps facts the caller states (name, company, lead volume, team
  size, ... plus every configured Data Collection field). After each caller
  turn a small out-of-band LLM extraction updates the store in the background.
* The store lives on the engine for the whole call, so node transitions never
  erase it. Every node's system prompt gets a CALLER FACTS block rebuilt from
  the store (the engine re-applies it on every node change and whenever the
  facts change), so the LLM sees the facts on every turn in every step.
* ``employee_name`` (the AI employee) and ``caller_name`` (the person calling)
  are separate keys and are never merged.

Nothing here is employee- or workflow-specific: field list, names and language
come from configuration. Workflows without ``caller_facts`` are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from loguru import logger

CONFIG_KEY = "caller_facts"
EMPLOYEE_KEY = "astra_employee"
GATHERED_KEY = "caller_facts"
MARKER = "# CALLER FACTS (CALL MEMORY)"

# Generic business-conversation facts every employee keeps.
DEFAULT_FIELDS: list[dict] = [
    {"key": "caller_name", "label": "Caller name", "hint": "The caller's own name (the person calling). Never the AI employee's name."},
    {"key": "company_name", "label": "Company", "hint": "The caller's company, brand or business name."},
    {"key": "lead_volume", "label": "Lead volume", "hint": "How many leads / enquiries / calls they get, with the number and period exactly as the caller said it (for example '40 per week'). Never convert the period."},
    {"key": "sales_team_size", "label": "Sales team size", "hint": "How many people are in their sales / calling team (for example '12 people')."},
    {"key": "business_type", "label": "Business type", "hint": "Industry or type of business."},
    {"key": "requirement", "label": "Requirement", "hint": "What the caller wants help with, in a short phrase."},
    {"key": "languages_required", "label": "Languages required", "hint": "Languages their customers need."},
    {"key": "location", "label": "Location", "hint": "City / region of the business."},
    {"key": "budget", "label": "Budget", "hint": "Budget the caller mentioned."},
    {"key": "crm", "label": "CRM / tools", "hint": "CRM or software they use."},
    {"key": "preferred_demo_time", "label": "Preferred demo time", "hint": "Day / time the caller prefers for a demo or callback."},
    {"key": "email", "label": "Email", "hint": "Email address the caller gave."},
    {"key": "phone", "label": "Phone", "hint": "Phone number the caller gave."},
]

# How to address the caller by name, per reply language (generic).
_ADDRESS_FORMS = {
    "te": "{name} గారు",
    "hi": "{name} जी",
    "ta": "{name} அவர்களே",
    "kn": "{name} ಅವರೇ",
    "ml": "{name}",
    "mr": "{name} जी",
    "en": "{name}",
}

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(value: Any) -> str:
    return _SLUG.sub("_", str(value or "").strip().lower()).strip("_")[:40]


def config(run_configs: Optional[dict]) -> Optional[dict]:
    cfg = (run_configs or {}).get(CONFIG_KEY) if isinstance(run_configs, dict) else None
    if not isinstance(cfg, dict) or cfg.get("enabled") is False:
        return None
    return cfg


def is_enabled(run_configs: Optional[dict]) -> bool:
    return config(run_configs) is not None


def resolve_fields(run_configs: Optional[dict]) -> list[dict]:
    """Default fields + the employee's configured Data Collection fields."""
    out: list[dict] = [dict(f) for f in DEFAULT_FIELDS]
    seen = {f["key"] for f in out}
    cfg = config(run_configs) or {}
    for raw in cfg.get("fields") or []:
        if not isinstance(raw, dict):
            continue
        key = _slug(raw.get("key") or raw.get("label"))
        if not key or key in seen:
            continue
        label = str(raw.get("label") or key).strip()[:120]
        hint = str(raw.get("question") or raw.get("hint") or label).strip()[:300]
        ftype = str(raw.get("type") or "text")
        opts = [str(o) for o in (raw.get("options") or []) if str(o).strip()][:20]
        if opts:
            hint += " (one of: " + ", ".join(opts) + ")"
        out.append({"key": key, "label": label, "hint": hint, "type": ftype, "data_collection": True})
        seen.add(key)
    return out


def employee_identity(run_configs: Optional[dict], call_context_vars: Optional[dict]) -> dict:
    """employee.name from the activated config, else the session context."""
    emp = (run_configs or {}).get(EMPLOYEE_KEY) if isinstance(run_configs, dict) else None
    emp = emp if isinstance(emp, dict) else {}
    ctx = call_context_vars or {}
    name = str(emp.get("name") or ctx.get("employee_name") or "").strip()
    emp_id = str(emp.get("id") or ctx.get("employee_id") or "").strip()
    return {"name": name, "id": emp_id}


def identity_context_vars(run_configs: Optional[dict], call_context_vars: Optional[dict]) -> dict:
    ident = employee_identity(run_configs, call_context_vars)
    out = {}
    if ident["name"]:
        out["employee_name"] = ident["name"]
    if ident["id"]:
        out["employee_id"] = ident["id"]
    return out


def _lang_base(run_configs: Optional[dict]) -> str:
    lang = str((run_configs or {}).get("primary_language") or "").strip().lower()
    return lang.split("-")[0] if lang else "en"


def address_form(run_configs: Optional[dict], name: str) -> str:
    tpl = _ADDRESS_FORMS.get(_lang_base(run_configs), "{name}")
    try:  # Astra: speak the name in the call language's script (runtime only)
        from api.services.pipecat.spoken_name import to_spoken

        name = to_spoken(name, (run_configs or {}).get("primary_language")) or name
    except Exception:
        pass
    return tpl.format(name=name)


def _clean_value(v: Any) -> Optional[str]:
    if v is None or isinstance(v, bool) and v is False:
        return None
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v if str(x).strip())
    s = str(v).strip()
    if not s or s.lower() in {"null", "none", "unknown", "n/a", "na", "not mentioned", "not provided", "-", "''", '""'}:
        return None
    return s[:200]


_SENT_END = re.compile(r"[?？.!।\n]")
_TOKEN = re.compile(r"[\w\u0900-\u0DFF]+", re.UNICODE)


def _questions_in(text: str) -> list[str]:
    out = []
    for part in re.split(r"(?<=[?？])", text or ""):
        q = part.strip()
        if q.endswith(("?", "？")) and len(q) > 3:
            q = re.split(r"(?<=[.!।])\s+", q)[-1].strip()[:160]
            if q and q not in out:
                out.append(q)
    return out


def _tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    # tolerate inflected endings (e.g. Telugu verb/case suffixes): long common prefix
    n = min(len(a), len(b))
    if n < 4:
        return False
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k >= 4 and k >= 0.75 * n


def is_repeat_question(sentence: str, previous: list, threshold: float = 0.7, ignore=None) -> bool:
    """True when ``sentence`` is a question that (nearly) repeats one already asked.

    ``ignore``: words that carry no meaning for the comparison (e.g. the caller's name).
    """
    s = (sentence or "").strip()
    if not s.endswith(("?", "？")):
        return False
    skip = {w.lower() for w in (ignore or ())}
    a = _tokens(s) - skip
    if not a:
        return False
    for q in previous or []:
        b = _tokens(q) - skip
        if not b:
            continue
        if a == b:  # same question, even a short one
            return True
        small, big = (a, b) if len(a) <= len(b) else (b, a)
        if len(small) < 3:
            continue
        hits = sum(1 for w in small if any(_same_word(w, x) for x in big))
        if hits / len(small) >= threshold:
            return True
    return False


class CallerFactsStore:
    def __init__(self, fields: list[dict]):
        self.fields = fields
        self._keys = [f["key"] for f in fields]
        self.facts: dict[str, str] = {}
        self.history: list[dict] = []
        self.asked: list[str] = []  # questions the AI employee already asked this call
        self.earlier_asked: list[str] = []  # asked before the latest reply (repeat guard)

    def update(self, data: dict, source: str = "llm") -> dict:
        changed = {}
        if not isinstance(data, dict):
            return changed
        for key in self._keys:
            if key not in data:
                continue
            val = _clean_value(data.get(key))
            if val is None:
                continue  # never erase a known fact with an empty extraction
            if self.facts.get(key) != val:
                self.facts[key] = val
                changed[key] = val
        if changed:
            self.history.append({"source": source, "changed": changed})
        return changed

    def snapshot(self) -> dict:
        return dict(self.facts)

    def note_assistant_questions(self, messages: list, limit: int = 8) -> None:
        asked: list[str] = []
        per_turn: list[list[str]] = []
        current: list[str] = []  # questions already spoken since the caller's last message
        last_user = ""
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                current = []
                c = m.get("content")
                last_user = c if isinstance(c, str) else ""
                continue
            if not (m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip()):
                continue
            turn_qs = _questions_in(m["content"])
            per_turn.append(turn_qs)  # one entry per spoken reply, with or without questions
            current = current + turn_qs
            for q in turn_qs:
                if q not in asked:
                    asked.append(q)
        self.asked = asked[-limit:]
        # Every question already asked counts. Exception: when the caller's latest message is
        # very short (unclear / did not hear), re-asking the question of the previous reply is
        # allowed. Questions already spoken in the current turn always count.
        if current or len(_tokens(last_user)) >= 2:
            earlier = [q for qs in per_turn for q in qs]
        else:
            last = set(per_turn[-1]) if per_turn else set()
            earlier = [q for qs in per_turn[:-1] for q in qs if q not in last]
        self.earlier_asked = earlier[-(limit * 2):]

    def label(self, key: str) -> str:
        for f in self.fields:
            if f["key"] == key:
                return f["label"]
        return key


def build_facts_block(store: CallerFactsStore, run_configs: Optional[dict], employee_name: str) -> str:
    lines = [MARKER]
    lines.append(
        "This block is the memory of THIS call. It is kept across every step of the conversation "
        "(welcome, understanding needs, booking, closing); it is one continuous conversation."
    )
    if employee_name:
        lines.append(
            f"Your own name (the AI employee) is {employee_name}. The caller is a different person: "
            "never use your own name as the caller's name, and never call yourself by the caller's name."
        )
    known = [(k, v) for k, v in store.facts.items()]
    if known:
        lines.append("Facts the caller has already told you (treat as true):")
        for k, v in known:
            lines.append(f"- {store.label(k)}: {v}")
    else:
        lines.append("Facts the caller has already told you: none yet.")
    lines.append("RULES:")
    lines.append(
        "- Never ask again for a fact listed above. If the caller asks about one of them, answer from this list "
        "exactly (same number, same unit and period; never convert per day into per week or similar)."
    )
    lines.append(
        "- Use the known facts in your answers (for example relate their lead volume to their team size) "
        "instead of asking questionnaire-style questions. Listen, answer naturally, then continue."
    )
    lines.append(
        "- Ask about something not yet known only when it is really needed for the next step, one thing at a time, "
        "at most one question per reply."
    )
    lines.append(
        "- Never ask the same question twice in this call. If the caller did not answer a question, drop it and "
        "continue with what they said."
    )
    lines.append(
        "- When the caller asks you to recall something they said, answer just that in one short sentence, "
        "without adding a new question."
    )
    lines.append(
        "- Do not end replies with a generic 'anything else?' / 'how else can I help?' question; ask something "
        "only when it moves the conversation forward."
    )
    name = store.facts.get("caller_name")
    if name:
        form = address_form(run_configs, name)
        lines.append(
            f"- Address the caller as \"{form}\" naturally: right after they introduce themselves, when confirming "
            "details, during booking, and occasionally otherwise. Not in every reply."
        )
        if _lang_base(run_configs) != "en":
            lines.append(
                f"- Write the caller's name in the script of the language you are speaking (for example \"{form}\"; "
                "a Latin name like Ram is written రామ్ in Telugu, राम in Hindi). Never in another language's script."
            )
    if store.asked:
        lines.append("Questions you already asked in this call (do not ask them again): " + " | ".join(store.asked))
    return "\n".join(lines)


_EXTRACT_SYSTEM = (
    "You extract facts that the CALLER stated about themselves and their business from a phone conversation "
    "between an AI employee (assistant) and a caller (user). Return ONLY a JSON object. Include only keys "
    "whose value the caller clearly stated or confirmed; omit everything else. Keep numbers as digits with the "
    "unit/period exactly as stated by the caller (e.g. \"40 per week\", \"12 people\"). Transcripts may be "
    "speech-recognition text in any language or script: normalise spoken numbers to digits when unambiguous. "
    "Write names in Latin letters as the caller would spell them. Never use the AI employee's name as the caller's name."
)


def _render_history(messages: list, limit: int = 24) -> str:
    rows = []
    for m in messages[-limit:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            rows.append(f"{role}: {content.strip()}")
    return "\n".join(rows)


class CallerFactsTracker:
    """Owns the store for one call and keeps the engine's prompt in sync."""

    def __init__(self, engine: Any, run_configs: Optional[dict]):
        self.engine = engine
        self.run_configs = run_configs or {}
        self.store = CallerFactsStore(resolve_fields(run_configs))
        ident = employee_identity(run_configs, getattr(engine, "_call_context_vars", {}) or {})
        self.employee_name = ident["name"]
        self._last_user_seen: Optional[str] = None
        self._pending: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self.dropped_questions: list[str] = []  # repeated questions the guard removed

    # -- prompt block -----------------------------------------------------
    def prompt_block(self) -> str:
        self._merge_gathered()
        return build_facts_block(self.store, self.run_configs, self.employee_name)

    def _merge_gathered(self) -> None:
        gathered = getattr(self.engine, "_gathered_context", None)
        if isinstance(gathered, dict):
            data = {k: gathered.get(k) for k in self.store._keys if k in gathered and k not in self.store.facts}
            if data:
                self.store.update(data, source="variable_extraction")
            gathered[GATHERED_KEY] = self.store.snapshot()

    # -- turn hooks -------------------------------------------------------
    async def before_generation(self, messages: list) -> None:
        """Awaited right before each generation: sync the prompt, start extraction."""
        try:
            before = list(self.store.asked)
            self.store.note_assistant_questions(messages)
            if self.store.asked != before:
                refresh = getattr(self.engine, "refresh_caller_facts_prompt", None)
                if refresh is not None:
                    await refresh()
        except Exception as exc:
            logger.debug(f"CALLER_FACTS prompt sync skipped: {exc}")
        self.on_context(messages)

    def on_context(self, messages: list) -> None:
        """Start a background extraction for a new caller message."""
        last_user = None
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                last_user = m["content"]
                break
        if not last_user or last_user == self._last_user_seen:
            return
        self._last_user_seen = last_user
        snapshot = list(messages)
        try:
            task = asyncio.create_task(self._extract(snapshot), name="caller-facts-extract")
        except RuntimeError:
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def wait_idle(self, timeout: float = 20.0) -> None:
        if self._pending:
            await asyncio.wait(list(self._pending), timeout=timeout)

    async def _extract(self, messages: list) -> None:
        async with self._lock:
            try:
                from pipecat.processors.aggregators.llm_context import LLMContext

                fields_desc = "\n".join(f"- {f['key']}: {f['hint']}" for f in self.store.fields)
                known = json.dumps(self.store.snapshot(), ensure_ascii=False)
                user_prompt = (
                    f"AI employee name (NOT the caller): {self.employee_name or '(unknown)'}\n"
                    f"Facts already known (update only if the caller corrected them): {known}\n\n"
                    f"Keys:\n{fields_desc}\n\nConversation:\n{_render_history(messages)}"
                )
                ctx = LLMContext()
                ctx.add_message({"role": "user", "content": user_prompt})
                llm = getattr(self.engine, "variable_extraction_llm", None) or getattr(self.engine, "inference_llm", None)
                if llm is None:
                    return
                raw = await llm.run_inference(ctx, system_instruction=_EXTRACT_SYSTEM)
                if not raw:
                    return
                try:
                    from api.services.gen_ai.json_parser import parse_llm_json

                    data = parse_llm_json(raw)
                except Exception:
                    data = json.loads(raw)
                changed = self.store.update(data if isinstance(data, dict) else {}, source="llm")
                gathered = getattr(self.engine, "_gathered_context", None)
                if isinstance(gathered, dict):
                    gathered[GATHERED_KEY] = self.store.snapshot()
                if changed:
                    logger.info(f"CALLER_FACTS updated: {sorted(changed)}")
                    refresh = getattr(self.engine, "refresh_caller_facts_prompt", None)
                    if refresh is not None:
                        await refresh()
            except Exception as exc:  # never break a call on memory extraction
                logger.warning(f"CALLER_FACTS extraction skipped: {exc}")


def build_tracker_processor(tracker: CallerFactsTracker):
    """FrameProcessor placed right before the LLM: observes each LLMContextFrame."""
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class CallerFactsProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.tracker = tracker

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                try:
                    await self.tracker.before_generation(frame.context.get_messages())
                except Exception as exc:
                    logger.debug(f"CALLER_FACTS observe skipped: {exc}")
            await self.push_frame(frame, direction)

    return CallerFactsProcessor()


def build_repeat_question_guard(tracker: CallerFactsTracker):
    """FrameProcessor placed right after the LLM.

    Drops a question sentence that repeats a question already asked earlier in the
    call (before the previous reply), as long as the reply says something else.
    Sentence-level buffering only; everything else passes through unchanged.
    """
    from pipecat.frames.frames import (
        InterruptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class RepeatQuestionGuard(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.tracker = tracker
            self._reset()

        def _reset(self):
            self._buf = ""
            self._emitted = False
            self._held: Optional[str] = None
            self._src = None

        def _ignore_words(self) -> set:
            words: set = set()
            try:
                name = self.tracker.store.facts.get("caller_name")
                if name:
                    words |= _tokens(name)
                    words |= _tokens(address_form(getattr(self.tracker, "run_configs", {}), name))
            except Exception:
                pass
            return words

        def _sync_from_context(self):
            # Generations after a tool call or node change do not pass the pre-LLM
            # observer, so re-read what was already said from the live context.
            try:
                ctx = getattr(getattr(self.tracker, "engine", None), "context", None)
                if ctx is not None:
                    self.tracker.store.note_assistant_questions(ctx.get_messages())
            except Exception as exc:
                logger.debug(f"CALLER_FACTS guard sync skipped: {exc}")

        def _frame(self, text: str):
            f = LLMTextFrame(text)
            for attr in ("skip_tts", "includes_inter_frame_spaces"):
                if self._src is not None and hasattr(self._src, attr):
                    try:
                        setattr(f, attr, getattr(self._src, attr))
                    except Exception:
                        pass
            return f

        async def _emit(self, text: str, direction):
            if text:
                if text.strip():
                    self._emitted = True
                await self.push_frame(self._frame(text), direction)

        async def _sentence(self, text: str, direction):
            if text.strip() and is_repeat_question(
                text, getattr(self.tracker.store, "earlier_asked", []), ignore=self._ignore_words()
            ):
                logger.info("CALLER_FACTS dropped repeated question")
                dropped = getattr(self.tracker, "dropped_questions", None)
                if isinstance(dropped, list):
                    dropped.append(text.strip())
                if not self._emitted and self._held is None:
                    self._held = text  # spoken only if the reply would otherwise be empty
                return
            await self._emit(text, direction)

        async def _drain(self, direction, final: bool):
            while True:
                m = _SENT_END.search(self._buf)
                if not m:
                    break
                end = m.end()
                while end < len(self._buf) and self._buf[end] in " \t":
                    end += 1
                piece, self._buf = self._buf[:end], self._buf[end:]
                await self._sentence(piece, direction)
            if final:
                if self._buf:
                    piece, self._buf = self._buf, ""
                    await self._sentence(piece, direction)
                if not self._emitted and self._held:
                    await self._emit(self._held, direction)
                self._held = None

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
                self._src = frame
                self._buf += frame.text or ""
                await self._drain(direction, final=False)
                return
            if isinstance(frame, InterruptionFrame):
                self._reset()
            elif isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
                await self._drain(direction, final=True)
                if isinstance(frame, LLMFullResponseStartFrame):
                    self._reset()
                    self._sync_from_context()
                else:
                    self._buf, self._held, self._src = "", None, None
                    self._emitted = False
            elif direction == FrameDirection.DOWNSTREAM and self._buf:
                await self._drain(direction, final=True)
            await self.push_frame(frame, direction)

    return RepeatQuestionGuard()
