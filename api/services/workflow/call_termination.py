"""Deterministic call termination (ACTIVE -> ENDING -> ENDED).

Why: the old path queued an EndFrame as soon as the end node was entered and
then waited on a synchronous final extraction + a full pipeline drain, which
left ~5 s of dead air between the closing line and the hang-up, and the
termination itself was keyed off "is the call disposed" flags only.

Rules implemented here (all employees / workflows / languages, Guided and
Strict alike - the trigger is structural, never a phrase match):

* A call enters ENDING only from an explicit terminal action:
  - the LLM calls a transition into an ``is_end`` node (Astra: ``customer_finished``),
  - an END_CALL tool call,
  - the silence (user-idle) limit,
  - or a hard stop (max duration, voicemail, pipeline error, caller hang-up).
* While ENDING: caller audio is muted for conversation (no interruptions of the
  closing line, no new turns), user-idle prompts are suppressed, and at most one
  closing generation is allowed.
* When the closing utterance has finished playing (BotStoppedSpeaking after the
  closing LLM response ended) the hang-up runs exactly once.
* Hang-up = graceful EndFrame, then a bounded watchdog; if the transport has not
  confirmed the disconnect it escalates to a CancelFrame, then logs a failure.
* ENDED is set only when the transport/provider reports the disconnect
  (on_client_disconnected / pipeline finished).
* Every transition and attempt is recorded in ``gathered_context["call_termination"]``.

Disable at runtime with DOGRAH_DETERMINISTIC_END=false (falls back to the old path).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

ACTIVE = "ACTIVE"
ENDING = "ENDING"
ENDED = "ENDED"


def enabled() -> bool:
    return os.getenv("DOGRAH_DETERMINISTIC_END", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class CallTermination:
    def __init__(self, run_id: Any = None):
        self.run_id = run_id
        self.state = ACTIVE
        self.reason: Optional[str] = None
        self.source: Optional[str] = None
        self.closing_text = ""
        self.events: list[dict] = []
        self.hangup_attempts: list[dict] = []
        self._t0 = time.monotonic()
        self._ending_at: Optional[float] = None
        self._closing_started_at: Optional[float] = None
        self._closing_finished_at: Optional[float] = None
        self._hangup_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        # closing generation bookkeeping
        self._expect_generation = False
        self._generation_open = False
        self._generation_done = False
        self._bot_speaking = False
        self._hangup_started = False
        self._hangup_task: Optional[asyncio.Task] = None
        self._settle_task: Optional[asyncio.Task] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._pending_task: Optional[asyncio.Task] = None
        self._pending_queued = False
        self._need_new_start = False
        self._ended_evt = asyncio.Event()
        self._hangup_fn: Optional[Callable[[str, int], Awaitable[None]]] = None
        self.suppressed_generations = 0
        self.ignored_user_frames = 0

    # ------------------------------------------------------------------ utils
    def _rel(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def event(self, name: str, **data) -> None:
        ev = {"t": self._rel(), "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": name}
        ev.update({k: v for k, v in data.items() if v is not None})
        self.events.append(ev)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        logger.info(f"CALL_TERM run={self.run_id} state={self.state} {name} {data if data else ''}")

    @property
    def is_active(self) -> bool:
        return self.state == ACTIVE

    def set_hangup_fn(self, fn: Callable[[str, int], Awaitable[None]]) -> None:
        self._hangup_fn = fn

    # ------------------------------------------------------------ transitions
    def begin_ending(self, reason: str, source: str, *, expect_generation: bool, bot_speaking: bool) -> bool:
        """ACTIVE -> ENDING. Returns False (and logs) for a duplicate request."""
        if self.state != ACTIVE:
            self.event("duplicate_end_request", reason=reason, source=source)
            return False
        self.state = ENDING
        self.reason = reason
        self.source = source
        self._ending_at = time.monotonic()
        self._expect_generation = expect_generation
        self._bot_speaking = bot_speaking
        self.event("state_ENDING", reason=reason, source=source,
                   expect_generation=expect_generation, bot_speaking=bot_speaking)
        self._arm_watchdog()
        return True

    def no_closing_generation(self, closing_text: str = "", *, queued: bool = False) -> None:
        """No separate closing generation will run: the closing is speech that is
        already playing, already queued (tool message) or was just streamed by
        the same LLM turn that issued the terminal action."""
        if self.state != ENDING:
            return
        self._expect_generation = False
        self._generation_done = True
        if closing_text and not self.closing_text:
            self.closing_text = closing_text
        self.event("closing_is_current_speech", bot_speaking=self._bot_speaking, queued=queued)
        self._need_new_start = queued
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        if self._bot_speaking and not queued:
            return  # BotStoppedSpeaking will trigger the hangup
        wait = _f("DOGRAH_END_PENDING_SPEECH_START_SECS", 2.5)
        if queued:
            wait = _f("DOGRAH_END_CLOSING_START_TIMEOUT_SECS", 8.0)

        async def _pending():
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                return
            if not self._bot_speaking and not self._hangup_started:
                self.event("no_closing_audio_started", waited=wait)
                self._request_hangup("closing_not_played")

        self._pending_task = asyncio.create_task(_pending())
        self._pending_queued = queued

    def can_accept_closing_line(self) -> bool:
        """ENDING but nothing has been said as the closing yet (e.g. silence
        timeout generation that only called end_call)."""
        return (
            self.state == ENDING
            and not self._hangup_started
            and self._closing_started_at is None
            and not self._bot_speaking
            and not self.closing_text.strip()
            and not self._generation_open
        )

    def expect_generation(self) -> None:
        if self.state == ENDING:
            self._expect_generation = True
            self._generation_done = False

    # ------------------------------------------------------- frame observers
    def on_llm_frame(self, kind: str, text: str = "") -> bool:
        """kind in {start,text,end}. Returns True when the generation is allowed."""
        if self.state != ENDING:
            return True
        if kind == "start":
            if self._generation_done and not self._expect_generation:
                self.suppressed_generations += 1
                self.event("extra_generation_after_closing")
                return False
            self._generation_open = True
            self.event("closing_generation_started")
        elif kind == "text" and self._generation_open:
            self.closing_text += text
        elif kind == "end" and self._generation_open:
            self._generation_open = False
            self._generation_done = True
            self._expect_generation = False
            self.event("closing_generation_done", closing_text=self.closing_text.strip()[:500])
            if not self._bot_speaking and self._closing_started_at is None and not self.closing_text.strip():
                # no text (the model may have put the goodbye in an end_call
                # closing_line instead) - give queued speech a moment to start
                self.event("closing_generation_empty")
                self.no_closing_generation("")
        return True

    def on_bot_started(self) -> None:
        self._bot_speaking = True
        if self.state == ENDING and self._need_new_start:
            self._need_new_start = False
            self.event("queued_closing_audio_started")
        if self.state == ENDING and self._closing_started_at is None:
            self._closing_started_at = time.monotonic()
            self.event("closing_audio_started")
        if self._settle_task and not self._settle_task.done():
            self._settle_task.cancel()
            self.event("closing_audio_resumed")
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    def on_bot_stopped(self) -> None:
        self._bot_speaking = False
        if self.state != ENDING or self._hangup_started:
            return
        if self._expect_generation and not self._generation_done:
            # speech that was already in flight when ENDING began; the closing
            # generation has not finished yet - keep waiting
            self.event("bot_stopped_waiting_for_closing")
            return
        if self._generation_open:
            self.event("bot_stopped_mid_generation")
            return
        if self._need_new_start:
            # earlier speech ended; the queued closing line has not started yet
            self.event("bot_stopped_waiting_for_queued_closing")
            return
        self._closing_finished_at = time.monotonic()
        settle = _f("DOGRAH_END_SETTLE_SECS", 0.4)
        self._settle_task = asyncio.create_task(self._settle_then_hangup(settle))

    async def _settle_then_hangup(self, settle: float) -> None:
        try:
            await asyncio.sleep(settle)
        except asyncio.CancelledError:
            return
        if not self._bot_speaking:
            self.event("closing_audio_finished")
            self._request_hangup("closing_audio_finished")

    def on_user_frame_ignored(self) -> None:
        self.ignored_user_frames += 1
        if self.ignored_user_frames == 1:
            self.event("inbound_speech_ignored_while_ending")

    # ---------------------------------------------------------------- hangup
    def _arm_watchdog(self) -> None:
        start_to = _f("DOGRAH_END_CLOSING_START_TIMEOUT_SECS", 8.0)
        play_to = _f("DOGRAH_END_CLOSING_MAX_SECS", 30.0)

        async def _wd():
            try:
                await asyncio.sleep(start_to)
                if self._closing_started_at is None and not self._bot_speaking and not self._hangup_started:
                    self.event("closing_start_timeout", secs=start_to)
                    self._request_hangup("closing_start_timeout")
                    return
                await asyncio.sleep(max(0.0, play_to - start_to))
                if not self._hangup_started:
                    self.event("closing_play_timeout", secs=play_to)
                    self._request_hangup("closing_play_timeout")
            except asyncio.CancelledError:
                pass

        self._watch_task = asyncio.create_task(_wd())

    def _request_hangup(self, trigger: str) -> None:
        if self._hangup_started:
            self.event("duplicate_hangup_trigger", trigger=trigger)
            return
        self._hangup_started = True
        self._hangup_at = time.monotonic()
        self._hangup_task = asyncio.create_task(self._hangup_loop(trigger))

    async def request_hangup_now(self, trigger: str) -> None:
        self._request_hangup(trigger)

    async def _hangup_loop(self, trigger: str) -> None:
        self.event("hangup_requested", trigger=trigger)
        timeouts = [_f("DOGRAH_END_HANGUP_CONFIRM_SECS", 3.0), _f("DOGRAH_END_CANCEL_CONFIRM_SECS", 3.0)]
        for attempt, wait in enumerate(timeouts, start=1):
            if self.state == ENDED:
                return
            rec = {"attempt": attempt, "t": self._rel(), "mode": "end" if attempt == 1 else "cancel"}
            self.hangup_attempts.append(rec)
            try:
                if self._hangup_fn:
                    await self._hangup_fn(trigger, attempt)
                rec["ok"] = True
            except Exception as e:  # never let a hangup error kill the loop
                rec["ok"] = False
                rec["error"] = str(e)[:300]
                logger.error(f"CALL_TERM run={self.run_id} hangup attempt {attempt} failed: {e}")
            try:
                await asyncio.wait_for(self._ended_evt.wait(), timeout=wait)
                return
            except asyncio.TimeoutError:
                self.event("hangup_not_confirmed", attempt=attempt, waited=wait)
        logger.error(f"CALL_TERM run={self.run_id} HANGUP_FAILED: no disconnect confirmation after {len(timeouts)} attempts")
        self.event("hangup_failed")

    def mark_ended(self, source: str) -> None:
        if self.state == ENDED:
            return
        prev = self.state
        if prev == ACTIVE:
            # disconnect with no terminal action (caller hung up, network drop)
            self.reason = self.reason or "caller_or_network_disconnect"
            self.source = self.source or source
        elif prev == ENDING and not self._hangup_started:
            self.event("caller_disconnected_during_closing")
        self.state = ENDED
        self._ended_at = time.monotonic()
        self._ended_evt.set()
        for t in (self._watch_task, self._settle_task, self._pending_task):
            if t and not t.done():
                t.cancel()
        self.event("state_ENDED", confirmed_by=source)

    # ---------------------------------------------------------------- report
    def _ms(self, a: Optional[float], b: Optional[float]) -> Optional[int]:
        if a is None or b is None:
            return None
        return int(round((b - a) * 1000))

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "source": self.source,
            "closing_text": self.closing_text.strip()[:1000],
            "timings_ms": {
                "ending_to_closing_audio_start": self._ms(self._ending_at, self._closing_started_at),
                "closing_audio_end_to_hangup_request": self._ms(self._closing_finished_at, self._hangup_at),
                "closing_audio_end_to_disconnect": self._ms(self._closing_finished_at, self._ended_at),
                "hangup_request_to_disconnect": self._ms(self._hangup_at, self._ended_at),
                "ending_to_disconnect": self._ms(self._ending_at, self._ended_at),
            },
            "hangup_attempts": self.hangup_attempts,
            "ignored_user_frames": self.ignored_user_frames,
            "suppressed_generations": self.suppressed_generations,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Universal structured END_CALL action (injected at runtime on every non-end
# node of every workflow; no republish needed). The closing line is spoken via
# TTS directly, so ending costs no second LLM generation.
# ---------------------------------------------------------------------------
END_CALL_FUNCTION = "end_call"
END_CALL_DESCRIPTION = (
    "End the phone call. Call this ONLY when the conversation is truly over: the caller "
    "said goodbye / that they are done, the caller refused or asked you to stop calling, "
    "the caller is busy and the callback (or no callback) is settled, it is a wrong number, "
    "or every script step is complete. Never call it while a question is still open or "
    "the caller is still talking about something. Do not say the goodbye in your reply text; "
    "put the final goodbye in closing_line instead (one short, warm sentence in the caller's "
    "current language). On a refusal, keep it short and polite; you may add a WhatsApp "
    "follow-up offer only if your script or job explicitly offers WhatsApp follow-up. "
    "After this call nothing more is said."
)
END_CALL_PROPERTIES = {
    "reason": {
        "type": "string",
        "enum": ["completed", "caller_goodbye", "refused", "busy_callback", "wrong_number", "other"],
        "description": "Why the call is ending.",
    },
    "closing_line": {
        "type": "string",
        "description": "The exact final goodbye sentence to speak, in the caller's language.",
    },
}


def end_call_action_enabled() -> bool:
    return os.getenv("DOGRAH_END_CALL_ACTION", "true").strip().lower() not in ("0", "false", "no", "off")


def same_text(a: str, b: str) -> bool:
    n = lambda x: "".join(ch for ch in (x or "") if ch.isalnum())
    return bool(n(b)) and n(b) in n(a)


# Goodbye words in the languages our employees speak. If the reply streamed in this
# same turn already contains one, the separate spoken hang-up line (end_call
# closing_line or an end-call tool's message) is skipped so the caller never hears
# two goodbyes.
_GOODBYE_RE = re.compile(
    r"(\bbye\b|good\s*bye|take\s*care|have\s+a\s+(good|great|nice)\s+(day|evening|night)|"
    r"ధన్యవాదాలు|ధన్యవాదం|సెలవు|"
    r"धन्यवाद|शुक्रिया|अलविदा|फिर\s+मिलेंगे|"
    r"ಧನ್ಯವಾದ|நன்றி|ধন্যবাদ)",
    re.I,
)


def said_goodbye(text: str) -> bool:
    return bool(text) and bool(_GOODBYE_RE.search(str(text)))

