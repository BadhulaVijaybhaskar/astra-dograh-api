"""Runtime guard: one caller turn -> one assistant response across node transitions.

Problem (seen on run 238): the LLM streams a spoken answer AND calls a node
transition function in the same completion. The transition result then
triggers a second LLM generation in the new node, which answers the same
caller turn again, so the caller hears the answer twice.

The guard watches the frames the pipeline LLM pushes (synchronously, before
any function call runs), keyed by (session_id, generation_seq). When a
transition function completes, ``allow_follow_up`` returns False if the
generation that issued the call already spoke a complete response (a question,
or at least DOGRAH_TRANSITION_DEDUPE_MIN_WORDS words, default 6); the engine
then reports the transition result with ``run_llm=False`` so no second
generation is started for that caller turn. Generations that issued only a
tool call, or only a short filler ("సరే", "one moment"), still get their
follow-up generation, so the new node
answers normally. Non-transition tools (booking, availability, etc.) are not
affected: they keep their own run_llm semantics and acknowledgements.

Disable with env DOGRAH_TRANSITION_DEDUPE=false (no image rebuild needed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


def _env_enabled() -> bool:
    return os.getenv("DOGRAH_TRANSITION_DEDUPE", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in text or "")


def _min_words() -> int:
    try:
        return max(1, int(os.getenv("DOGRAH_TRANSITION_DEDUPE_MIN_WORDS", "6")))
    except ValueError:
        return 6


def is_complete_response(text: str, min_words: int | None = None) -> bool:
    """True when already-spoken text is a real answer, not a short filler.

    A short acknowledgement before a transition ("సరే", "Sure, one moment")
    is not an answer: the new node must still respond, so it is allowed.
    A reply that asks the caller something, or has >= min_words words, is a
    complete response: a second generation for the same turn would repeat it.
    """
    if not _speakable(text):
        return False
    t = text.strip()
    if "?" in t:
        return True
    words = [w for w in t.split() if _speakable(w)]
    return len(words) >= (min_words or _min_words())


@dataclass
class TransitionResponseGuard:
    session_id: Optional[int | str] = None
    enabled: bool = field(default_factory=_env_enabled)
    generation_seq: int = 0
    _generation_text: str = ""
    # (session_id, generation_seq) keys whose follow-up generation was suppressed
    suppressed: list = field(default_factory=list)
    # (session_id, generation_seq) keys whose follow-up generation was allowed
    allowed: list = field(default_factory=list)

    # -- frame observation -------------------------------------------------
    def on_generation_started(self) -> None:
        self.generation_seq += 1
        self._generation_text = ""

    def on_llm_text(self, text: str) -> None:
        if text:
            self._generation_text += text

    def observe_llm_frame(self, frame) -> None:
        """Feed every frame the pipeline LLM pushes (on_before_push_frame)."""
        # Local import keeps this module importable in light unit tests.
        from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame

        if isinstance(frame, LLMFullResponseStartFrame):
            self.on_generation_started()
        elif isinstance(frame, LLMTextFrame):
            # skip_tts text is never spoken, so it can't be a duplicate answer
            if not getattr(frame, "skip_tts", False):
                self.on_llm_text(frame.text)

    # -- decision -----------------------------------------------------------
    @property
    def key(self) -> tuple:
        return (self.session_id, self.generation_seq)

    def current_generation_spoke(self) -> bool:
        return is_complete_response(self._generation_text)

    def allow_follow_up(self, function_name: str, target_node: str = "") -> bool:
        """Return False when the transition's follow-up generation must be skipped."""
        if not self.enabled:
            return True
        key = self.key
        if self.current_generation_spoke():
            if key not in self.suppressed:
                self.suppressed.append(key)
            logger.info(
                "TURN_DEDUPE suppress follow-up generation: session={} gen={} "
                "fn={} -> {} (already spoke {} chars this turn)",
                key[0],
                key[1],
                function_name,
                target_node,
                len(self._generation_text.strip()),
            )
            return False
        self.allowed.append(key)
        logger.debug(
            "TURN_DEDUPE allow follow-up: session={} gen={} fn={} (no speech yet)",
            key[0],
            key[1],
            function_name,
        )
        return True
