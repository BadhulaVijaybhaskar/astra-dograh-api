"""Astra: phrase-level TTS text aggregation for streaming TTS (Sarvam WS).

v2 (default, natural phrase chunking, approved 2026-10-10):
  * Emit as soon as a *meaningful* phrase is ready: a sentence end (. ? ! ।)
    or a newline, with no extra waiting beyond pipecat's 1-char lookahead.
  * Never split at an ellipsis ("…" or "..."); it is a soft pause inside a phrase.
  * Tiny pieces (< MIN_PHRASE chars, e.g. "సరే, రామ్ గారు." or "BRS టీమ్ నుండి.")
    are merged forward into the next phrase instead of being spoken alone,
    because every TTS chunk is synthesised with its own sentence-final contour.
  * Clause cuts (, ; :) only in long sentences (>= CLAUSE_MIN chars), never
    between digits ("2,000"), never right after a number/currency token, never
    when the clause would leave a tiny remainder, so a question or a name like
    "BRS పార్టీ" / "KTR గారు" is not cut midway.
  * Hard cap (MAX_PHRASE) cuts only at a space whose left word is not a number,
    currency, Latin word followed by a Latin word (English phrase) or a short
    Telugu modifier.
  It never splits inside a word and never modifies text.

v1 (legacy) is kept and selected with ASTRA_TTS_CHUNKING=v1 (rollback switch).
"""
from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

MIN_PHRASE = 25
CLAUSE_MIN = 70
MAX_PHRASE = 160
_SENT_END = (".", "?", "!", "।", "\n")
_CLAUSE = (",", ";", ":", "،")
_NUMISH = re.compile(r"^(?:[₹$]|rs\.?|inr|\d[\d,.:x%/+-]*)$", re.I)
_LATIN = re.compile(r"^[A-Za-z][A-Za-z.&'-]*$")


def _left_word(s: str) -> str:
    s = s.rstrip(" ,;:")
    return s.split(" ")[-1] if s.strip() else ""


class NaturalPhraseAggregator(SimpleTextAggregator):
    def __init__(self, min_phrase: int = MIN_PHRASE, clause_min: int = CLAUSE_MIN,
                 max_phrase: int = MAX_PHRASE, **kwargs):
        kwargs.setdefault("aggregation_type", AggregationType.SENTENCE)
        super().__init__(**kwargs)
        self._min, self._clause_min, self._max = min_phrase, clause_min, max_phrase
        self._pending_end = None  # index just after a candidate sentence end

    def _emit(self, cut: int) -> Aggregation | None:
        head, tail = self._text[:cut], self._text[cut:]
        self._pending_end = None
        if not head.strip():
            return None
        self._text = tail.lstrip(" ")
        return Aggregation(text=head.strip(" "), type=AggregationType.SENTENCE)

    def _is_ellipsis(self, i: int) -> bool:
        t = self._text
        return t[i] == "…" or (t[i] == "." and (t[i - 1:i] == "." or t[i + 1:i + 2] == "."))

    def _sentence_cut(self) -> int | None:
        """Called on every char; decides sentence boundaries with 1-char lookahead."""
        t = self._text
        c = t[-1]
        if self._pending_end is not None:
            if not c.strip():
                return None  # whitespace: still waiting for the next real char
            end = self._pending_end
            self._pending_end = None
            prev = t[end - 1]
            if prev == "." and (end >= 2 and t[end - 2].isdigit()) and c.isdigit():
                return None  # decimal "1.5"
            if prev == "." and c == ".":
                return None  # ellipsis "..."
            if len(t[:end].strip()) < self._min:
                return None  # tiny sentence: merge forward
            return end
        if c == "\n" and len(t.strip()) >= self._min:
            return len(t)
        if c in _SENT_END and not self._is_ellipsis(len(t) - 1):
            self._pending_end = len(t)
        return None

    def _clause_cut(self) -> int | None:
        t = self._text
        if len(t) < 3 or not t[-1].strip():
            return None
        i = len(t) - 2
        while i >= 0 and not t[i].strip():
            i -= 1
        if i < 0 or t[i] not in _CLAUSE or i == len(t) - 2:
            return None
        if i + 1 < self._clause_min:
            return None
        if t[i] == "," and i > 0 and t[i - 1].isdigit() and t[-1].isdigit():
            return None
        if _NUMISH.match(_left_word(t[: i + 1]) or "x"):
            return None
        return i + 1

    def _hard_cut(self) -> int | None:
        t = self._text
        if len(t) < self._max:
            return None
        for m in reversed(list(re.finditer(r" ", t))):
            left = _left_word(t[: m.start()])
            right = t[m.end():].split(" ")[0] if t[m.end():] else ""
            if not left or m.start() < self._min:
                continue
            if _NUMISH.match(left):
                continue
            if _LATIN.match(left) and _LATIN.match(right or "x"):
                continue  # keep English phrases together
            if len(left) <= 3:
                continue  # short modifier: stay with its noun
            return m.start() + 1
        return None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        if self._aggregation_type == AggregationType.TOKEN:
            if text:
                yield Aggregation(text=text, type=AggregationType.TOKEN)
            return
        for char in text:
            self._text += char
            cut = self._sentence_cut()
            if cut is None and self._pending_end is None:
                cut = self._clause_cut() or self._hard_cut()
            if cut:
                out = self._emit(cut)
                if out:
                    yield out

    async def handle_interruption(self):
        await super().handle_interruption()
        self._pending_end = None

    async def reset(self):
        await super().reset()
        self._pending_end = None


# ---- v1 legacy (rollback: ASTRA_TTS_CHUNKING=v1) ----

V1_MIN_PHRASE = 28
V1_MAX_PHRASE = 140
_V1_CLAUSE = (",", ";", ":", "،")


class LegacyPhraseTextAggregator(SimpleTextAggregator):
    def __init__(self, min_phrase: int = V1_MIN_PHRASE, max_phrase: int = V1_MAX_PHRASE, **kwargs):
        kwargs.setdefault("aggregation_type", AggregationType.SENTENCE)
        super().__init__(**kwargs)
        self._min = min_phrase
        self._max = max_phrase

    def _emit(self, cut: int) -> Aggregation | None:
        head, tail = self._text[:cut], self._text[cut:]
        if not head.strip():
            return None
        self._text = tail.lstrip(" ")
        self._needs_lookahead = False
        return Aggregation(text=head.strip(" "), type=AggregationType.SENTENCE)

    def _clause_cut(self) -> int | None:
        # buffer looks like "...X, Y" where Y is the first non-space char after the comma
        t = self._text
        if len(t) < 3 or not t[-1].strip():
            return None
        i = len(t) - 2
        while i >= 0 and not t[i].strip():
            i -= 1
        if i < 0 or t[i] not in _V1_CLAUSE or i == len(t) - 2:
            return None  # need at least one space between clause mark and next char
        if i + 1 < self._min:
            return None
        if t[i] == "," and i > 0 and t[i - 1].isdigit() and t[-1].isdigit():
            return None
        return i + 1

    def _hard_cut(self) -> int | None:
        t = self._text
        if len(t) < self._max:
            return None
        for m in reversed(list(re.finditer(r" ", t))):
            left = t[: m.start()].rstrip().split(" ")[-1] if t[: m.start()].strip() else ""
            if left and not _NUMISH.match(left) and m.start() >= self._min:
                return m.start() + 1
        return None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for char in text:
            self._text += char
            result = await self._check_sentence_with_lookahead(char)
            if result:
                yield result
                continue
            if self._needs_lookahead:
                continue
            cut = self._clause_cut() or self._hard_cut()
            if cut:
                # keep the just-typed lookahead char in the next phrase
                out = self._emit(cut)
                if out:
                    yield out


# ---- v3 (steady prosody; ASTRA_TTS_CHUNKING=v3) ----
# Longer TTS units so the voice keeps one intonation contour per sentence:
#  * first chunk of a turn may be a short clause/sentence (>= V3_FIRST_MIN chars)
#    to keep first-audio latency; later chunks are whole sentences, and short
#    sentences (< V3_MIN chars) are merged forward into the next one;
#  * clause cuts only in very long sentences (>= V3_CLAUSE_MIN chars);
#  * a sentence end is held until >= V3_FIRST_MIN chars follow it, so a short
#    last sentence ("మీరు చెప్పగలరా?") is spoken with the previous one;
#  * same word / name / number+unit protections as v2 (inherited).
V3_FIRST_MIN = 25
V3_MIN = 60
V3_CLAUSE_MIN = 120
V3_MAX = 220


class SteadyPhraseAggregator(NaturalPhraseAggregator):
    def __init__(self, **kwargs):
        super().__init__(min_phrase=V3_FIRST_MIN, clause_min=V3_CLAUSE_MIN, max_phrase=V3_MAX, **kwargs)
        self._first_done = False
        self._hold = None  # sentence end held until enough text follows it

    def _sentence_cut(self):
        t = self._text
        if self._hold is not None and len(t[self._hold:].strip()) >= V3_FIRST_MIN:
            return self._hold  # what follows is not a tiny tail: emit the held sentence(s)
        cut = super()._sentence_cut()
        if cut is None or not self._first_done:
            return cut
        if self._hold is None:
            self._hold = cut  # wait: a tiny last sentence will be merged into this one
        return None

    def _hard_cut(self):
        return None if self._hold is not None else super()._hard_cut()

    def _emit(self, cut: int):
        self._hold = None
        out = super()._emit(cut)
        if out is not None and not self._first_done:
            self._first_done = True
            self._min = V3_MIN
        return out

    def _clause_cut(self):
        if self._hold is not None:
            return None
        if not self._first_done:
            # first chunk: allow a natural clause once it is long enough
            t = self._text
            if len(t) < 3 or not t[-1].strip():
                return None
            i = len(t) - 2
            while i >= 0 and not t[i].strip():
                i -= 1
            if i < 0 or t[i] not in _CLAUSE or i == len(t) - 2 or i + 1 < V3_FIRST_MIN:
                return None
            if t[i] == "," and i > 0 and t[i - 1].isdigit() and t[-1].isdigit():
                return None
            if _NUMISH.match(_left_word(t[: i + 1]) or "x"):
                return None
            return i + 1
        return super()._clause_cut()

    def _turn_reset(self):
        self._first_done = False
        self._min = V3_FIRST_MIN
        self._pending_end = None
        self._hold = None

    async def handle_interruption(self):
        await super().handle_interruption()
        self._turn_reset()

    async def reset(self):
        await super().reset()
        self._turn_reset()


MODE_TO_VERSION = {"smooth": "v3", "balanced": "v2", "fast": "v1"}


def chunking_version(mode: str | None = None) -> str:
    """Per-employee mode (smooth/balanced/fast) wins; else global env ASTRA_TTS_CHUNKING (v1/v2/v3); default v2."""
    m = MODE_TO_VERSION.get(str(mode or "").strip().lower())
    if m:
        return m
    env = os.environ.get("ASTRA_TTS_CHUNKING", "v2").strip().lower()
    return env if env in ("v1", "v2", "v3") else "v2"


def PhraseTextAggregator(mode: str | None = None, **kwargs):  # noqa: N802 - factory kept under the old name
    """Factory used by service_factory: per-employee mode, else env, else v2."""
    v = chunking_version(mode)
    if v == "v1":
        return LegacyPhraseTextAggregator(**kwargs)
    if v == "v3":
        return SteadyPhraseAggregator(**kwargs)
    return NaturalPhraseAggregator(**kwargs)
