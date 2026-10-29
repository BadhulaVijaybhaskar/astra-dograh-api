"""Config-driven spoken-term replacement for streamed LLM replies (platform-wide).

Some words should never be spoken by a given employee (e.g. sales jargon like
"qualification" in a Telugu persona). Prompts reduce them but cannot guarantee
zero, so this processor rewrites the streamed LLM text *before* TTS and before
the assistant transcript/context is built, so both what the caller hears and
what is logged use the configured spoken phrase.

Configured per workflow (absent -> processor not inserted, pipeline unchanged):
    workflow_configurations["spoken_term_replacements"] = [
        ["leads qualification", "సరైన leads ని identify చేయడం"],
        ["qualification", "సరైన leads ని identify చేయడం"],
        ...
    ]
Matching is case-insensitive, on word boundaries, longest phrase first.
Streaming-safe: only the trailing partial word (or a trailing prefix of a
multi-word phrase) is held back until the next token or the end of the reply.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

CONFIG_KEY = "spoken_term_replacements"


def parse_replacements(raw) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        raw = list(raw.items())
    if not isinstance(raw, (list, tuple)):
        return pairs
    for item in raw:
        if isinstance(item, dict):
            src, dst = item.get("term") or item.get("from"), item.get("say") or item.get("to")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            src, dst = item
        else:
            continue
        if isinstance(src, str) and src.strip() and isinstance(dst, str):
            pairs.append((" ".join(src.split()), dst))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


class SpokenTermRewriter:
    """Pure streaming rewriter (unit-testable without a pipeline)."""

    def __init__(self, pairs: Iterable[tuple[str, str]]):
        self.pairs = list(pairs)
        alts = "|".join(re.escape(src).replace(r"\ ", r"\s+") for src, _ in self.pairs)
        self._re = re.compile(rf"(?<![\w])({alts})(?![\w])", re.IGNORECASE) if alts else None
        self._map = {src.lower(): dst for src, dst in self.pairs}
        self._lower_phrases = [src.lower() for src, _ in self.pairs]
        self.pending = ""
        self.replaced = 0

    def _sub(self, text: str) -> str:
        if not self._re or not text:
            return text

        def rep(m):
            self.replaced += 1
            return self._map.get(" ".join(m.group(1).lower().split()), m.group(1))

        return self._re.sub(rep, text)

    def _hold_index(self, buf: str) -> int:
        """Index from which the buffer must be held back (not yet safe to emit)."""
        # Hold the trailing partial word.
        m = re.search(r"\S+$", buf)
        hold = m.start() if m else len(buf)
        low = buf.lower()
        ends_ws = buf[-1:].isspace()
        # Hold a trailing prefix of a multi-word phrase (e.g. "leads " before "qualification").
        for i in [0] + [k.end() for k in re.finditer(r"\s+", buf)]:
            if i >= hold:
                break
            t = " ".join(low[i:].split()) + (" " if ends_ws else "")
            if t.strip() and any(p.startswith(t) and p != t for p in self._lower_phrases):
                if i == 0 or not buf[i - 1].isalnum():
                    hold = i
                    break
        # Never split a configured phrase that is already complete in the buffer.
        if self._re is not None:
            for mm in self._re.finditer(buf):
                if mm.start() < hold < mm.end():
                    hold = mm.start()
        return hold

    def feed(self, text: str) -> str:
        self.pending += text or ""
        idx = self._hold_index(self.pending)
        out, self.pending = self.pending[:idx], self.pending[idx:]
        return self._sub(out)

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return self._sub(out)

    def reset(self) -> None:
        self.pending = ""


class SpokenTermGuardProcessor(FrameProcessor):
    def __init__(self, pairs: Iterable[tuple[str, str]], **kwargs):
        super().__init__(**kwargs)
        self._rw = SpokenTermRewriter(pairs)
        self._last_text_frame: Optional[LLMTextFrame] = None

    async def _flush(self, direction: FrameDirection) -> None:
        rest = self._rw.flush()
        if rest:
            f = LLMTextFrame(rest)
            src = self._last_text_frame
            if src is not None:
                for attr in ("skip_tts", "includes_inter_frame_spaces"):
                    if hasattr(src, attr):
                        try:
                            setattr(f, attr, getattr(src, attr))
                        except Exception:
                            pass
            await self.push_frame(f, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            before = self._rw.replaced
            out = self._rw.feed(frame.text)
            self._last_text_frame = frame
            if self._rw.replaced > before:
                logger.info(f"SPOKEN_TERM_GUARD replaced {self._rw.replaced - before} term(s)")
            if out:
                frame.text = out
                await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            self._rw.reset()
            self._last_text_frame = None
        elif isinstance(frame, (LLMFullResponseEndFrame, LLMFullResponseStartFrame)):
            before = self._rw.replaced
            await self._flush(direction)
            if self._rw.replaced > before:
                logger.info(f"SPOKEN_TERM_GUARD replaced {self._rw.replaced - before} term(s)")
            self._last_text_frame = None
        elif self._rw.pending and direction == FrameDirection.DOWNSTREAM and not isinstance(frame, LLMTextFrame):
            # any other downstream frame (e.g. function-call frames): emit held text first
            await self._flush(direction)
        await self.push_frame(frame, direction)


def build_spoken_term_guard(workflow_configurations) -> Optional[SpokenTermGuardProcessor]:
    cfg = workflow_configurations or {}
    pairs = parse_replacements(cfg.get(CONFIG_KEY)) if isinstance(cfg, dict) else []
    if not pairs:
        return None
    logger.info(f"SpokenTermGuard enabled with {len(pairs)} replacement(s)")
    return SpokenTermGuardProcessor(pairs)
