"""WF30 staging (round 2, fix 1): barge-in guard.

A caller interrupts the bot only with real words. Voice activity alone (line noise,
echo of the bot's own audio, an empty or punctuation-only STT result) never
interrupts:

* Bot not speaking: VAD / transcript start the user turn at once (normal fast turn).
* First ``grace_secs`` (~300 ms) of each bot utterance: VAD is ignored outright
  (no pause, no stutter); only a transcript with words interrupts.
* Later in the utterance: VAD pauses bot audio for up to ``pause_secs``. A
  transcript with words in that window interrupts (normal InterruptionFrame); no
  words = the pause is lifted and the bot continues from where it stopped.
  At most ``max_ignored_per_utterance`` such pauses per bot utterance; after that
  VAD no longer pauses, words still interrupt (no lockout, unlike the stock
  ProvisionalVADUserTurnStartStrategy).

Every decision is logged as ``STAGE_LAT {"ev":"barge_in", ...}`` with the bot
speech offset, STT text, confidence and accepted / ignored / cap.

Enabled only by the workflow config key ``astra_barge_in_guard.enabled`` (WF30
staging definition), so other workflows keep their strategies unchanged.

v2 (WF32 speakerphone, 26 Oct 2026). Extra keys, all off by default, so a config
without them behaves exactly like v1:

* ``echo_filter``: keep a rolling copy of the text the bot is speaking / just spoke
  (``BotTextTracker``, fed by ``BotTextTapProcessor`` placed before TTS). A transcript
  whose words mostly match that text (Telugu and romanised forms compared on a
  consonant skeleton, e.g. "Revanth Reddy" == "రేవంత్ రెడ్డి") is the bot's own voice
  coming back through a speakerphone: ignored while the bot speaks and for
  ``echo_tail_secs`` after it stops.
* ``min_words_while_speaking`` / ``min_word_letters``: while the bot speaks, interrupt
  only on at least N real words (a word = at least ``min_word_letters`` letters, so the
  one-syllable STT fragments "హ", "జ" from background noise do not count).
* ``interjections``: short stop words ("ఆగండి", "stop", "wait", "వద్దు", "busy" ...)
  still interrupt at once, unless they are an echo of the bot's own words.
* Bot silent (and outside the echo tail): unchanged, the first word / VAD starts the turn.
"""

import asyncio
import json
import re
import time
import unicodedata
from collections import deque

from loguru import logger
from pipecat.frames.frames import (
    BotOutputAudioPauseFrame,
    BotOutputAudioResumeFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

try:  # run id for the log line, when the runtime exposes it
    from pipecat.utils.run_context import get_current_run_id  # type: ignore
except Exception:  # pragma: no cover
    get_current_run_id = None  # type: ignore

_WORD = re.compile(r"[^\W\d_]", re.UNICODE)  # any letter in any script


def has_valid_words(text: str) -> bool:
    """A transcript counts as speech only if it contains at least one letter."""
    return bool(_WORD.search(str(text or "")))


def _confidence(frame) -> float | None:
    res = getattr(frame, "result", None)
    for src in (frame, res):
        if src is None:
            continue
        for key in ("confidence", "language_probability"):
            v = src.get(key) if isinstance(src, dict) else getattr(src, key, None)
            if isinstance(v, (int, float)):
                return float(v)
    return None


ASTRA_GUARD_VERSION = 2

# ---- v2: text normalisation (Telugu / Devanagari / Latin -> consonant skeleton) ---------
_TE_CONS = {
    "క": "k", "ఖ": "k", "గ": "g", "ఘ": "g", "ఙ": "n", "చ": "c", "ఛ": "c", "జ": "j", "ఝ": "j",
    "ఞ": "n", "ట": "t", "ఠ": "t", "డ": "d", "ఢ": "d", "ణ": "n", "త": "t", "థ": "t", "ద": "d",
    "ధ": "d", "న": "n", "ప": "p", "ఫ": "p", "బ": "b", "భ": "b", "మ": "m", "య": "y", "ర": "r",
    "ఱ": "r", "ల": "l", "ళ": "l", "వ": "v", "శ": "s", "ష": "s", "స": "s", "హ": "h",
    "ం": "n", "ఁ": "n",
}
_HI_CONS = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "च": "c", "छ": "c", "ज": "j", "झ": "j", "ट": "t",
    "ठ": "t", "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n",
    "प": "p", "फ": "p", "ब": "b", "भ": "b", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "s", "ष": "s", "स": "s", "ह": "h", "ं": "n", "ँ": "n",
}
_LAT_DIGRAPH = (("ksh", "ks"), ("chh", "c"), ("ch", "c"), ("sh", "s"), ("th", "t"), ("dh", "d"),
                ("kh", "k"), ("gh", "g"), ("ph", "p"), ("bh", "b"), ("jh", "j"), ("w", "v"),
                ("q", "k"), ("x", "ks"), ("z", "j"), ("f", "p"), ("c", "c"))
_LAT_CONS = set("bcdghjklmnprstvy")


def _skeleton(token: str) -> str:
    """Consonant skeleton of one word: script-independent, vowels dropped, doubles collapsed."""
    out = []
    t = unicodedata.normalize("NFC", token.lower())
    for a, b in _LAT_DIGRAPH:
        t = t.replace(a, b)
    t = re.sub(r"y\b", "", t)  # Latin word-final y is a vowel: "Reddy" == "రెడ్డి"
    for ch in t:
        c = _TE_CONS.get(ch) or _HI_CONS.get(ch)
        if c is None and ch in _LAT_CONS:
            c = ch
        if c and (not out or out[-1] != c):
            out.append(c)
    sk = "".join(out).replace("y", "")  # y is a glide both scripts insert/drop between vowels
    sk = re.sub(r"n(?=[pbm])", "m", sk)  # anusvara before a labial is m
    return re.sub(r"(.)\1+", r"\1", sk)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s\.,!?;:\"'()\[\]{}\-–—…/|]+", str(text or "")) if t and _WORD.search(t)]


def _letters(token: str) -> int:
    """Base letters only (Telugu vowel signs / virama are marks, not letters)."""
    return sum(1 for ch in token if unicodedata.category(ch).startswith("L"))


def _close(a: str, b: str) -> bool:
    if len(a) < 2 or len(b) < 2:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 4:
        return False
    # edit distance <= 1
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1; j += 1
    return edits + (len(a) - i) + (len(b) - j) <= 1


def echo_overlap(text: str, bot_text: str) -> float:
    """Share of the transcript's words that also occur (fuzzily) in the bot's recent text."""
    sk = [s for s in (_skeleton(t) for t in _tokens(text)) if len(s) >= 2]
    if not sk:
        return 0.0
    bot = {s for s in (_skeleton(t) for t in _tokens(bot_text)) if len(s) >= 2}
    if not bot:
        return 0.0
    hit = sum(1 for s in sk if s in bot or any(_close(s, b) for b in bot))
    return hit / len(sk)


DEFAULT_INTERJECTIONS = (
    "ఆగండి", "ఆగు", "ఆపండి", "ఆపు", "వద్దు", "వద్దండి", "బిజీ", "లేదు", "లేదండి", "ఒక్క నిమిషం",
    "stop", "wait", "busy", "no", "hold on", "hello", "హలో",
    "aagandi", "agandi", "aapandi", "vaddu", "vaddandi",
    "रुकिए", "रुको", "बस", "नहीं",
)


class BotTextTracker:
    """Rolling window of what the bot said (per call). Fed by BotTextTapProcessor."""

    def __init__(self, window_secs: float = 12.0, clock=time.time):
        self._window = float(window_secs)
        self._clock = clock
        self._items: deque = deque()  # (t, text)
        self._last_t = 0.0

    def add(self, text: str, *, sep: bool = False) -> None:
        if not text:
            return
        now = self._clock()
        if sep and self._items:
            self._items.append((now, " "))
        self._items.append((now, str(text)))
        self._last_t = now
        cutoff = now - self._window
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def break_word(self) -> None:
        if self._items:
            self._items.append((self._clock(), " "))

    def recent_text(self) -> str:
        cutoff = self._clock() - self._window
        return "".join(t for ts, t in self._items if ts >= cutoff)


class BotTextTapProcessor(FrameProcessor):
    """Pass-through, placed before TTS: copies the bot's text into a BotTextTracker."""

    def __init__(self, tracker: BotTextTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            try:
                if isinstance(frame, TTSSpeakFrame):
                    self._tracker.add(frame.text, sep=True)
                elif isinstance(frame, LLMTextFrame):
                    self._tracker.add(frame.text)
                elif isinstance(frame, LLMFullResponseEndFrame):
                    self._tracker.break_word()
            except Exception:  # never break the call on the tap
                pass
        await self.push_frame(frame, direction)


class AstraBargeInGuardStrategy(BaseUserTurnStartStrategy):
    def __init__(
        self,
        *,
        pause_secs: float = 1.2,
        grace_secs: float = 0.3,
        max_ignored_per_utterance: int = 1,
        use_interim: bool = True,
        run_id=None,
        echo_filter: bool = False,
        echo_overlap: float = 0.6,
        echo_tail_secs: float = 0.0,
        bot_text: "BotTextTracker | None" = None,
        min_words_while_speaking: int = 1,
        min_word_letters: int = 1,
        interjections=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._echo_filter = bool(echo_filter)
        self._echo_overlap = float(echo_overlap)
        self._echo_tail = max(0.0, float(echo_tail_secs))
        self.bot_text = bot_text if bot_text is not None else (BotTextTracker() if echo_filter else None)
        self._min_words = max(1, int(min_words_while_speaking))
        self._min_letters = max(1, int(min_word_letters))
        ij = DEFAULT_INTERJECTIONS if interjections is None else interjections
        self._interjections = {_skeleton(w) for w in ij if len(_skeleton(w)) >= 2}
        self._interjection_phrases = [w.lower() for w in ij]
        self._bot_stopped_at = 0.0
        self._pause_secs = float(pause_secs)
        self._grace_secs = float(grace_secs)
        self._max_ignored = max(0, int(max_ignored_per_utterance))
        self._use_interim = use_interim
        self._run_id = run_id
        self._bot_speaking = False
        self._bot_started_at = 0.0
        self._ignored = 0
        self._pause_active = False
        self._pause_started_at = 0.0
        self._resume_task: asyncio.Task | None = None

    # ---- logging -------------------------------------------------------------------------
    def _log(self, decision: str, **extra) -> None:
        run = self._run_id
        if run is None and get_current_run_id is not None:
            try:
                run = get_current_run_id()
            except Exception:
                run = None
        now = time.time()
        rec = {
            "run": run,
            "ev": "barge_in",
            "t": round(now, 4),
            "decision": decision,
            "bot_speaking": self._bot_speaking,
            "bot_ms": round((now - self._bot_started_at) * 1000) if self._bot_speaking else None,
            "ignored_in_utterance": self._ignored,
        }
        rec.update(extra)
        logger.info("STAGE_LAT " + json.dumps(rec, ensure_ascii=False))

    # ---- lifecycle -----------------------------------------------------------------------
    async def cleanup(self):
        await super().cleanup()
        await self._clear_pause()

    async def handle_user_turn_started(self):
        await self._clear_pause()
        self._bot_speaking = False

    async def handle_user_turn_stopped(self):
        await self._clear_pause()

    async def _clear_pause(self):
        await self._cancel_resume_task()
        if self._pause_active:
            await self._resume_output_audio()
        self._pause_active = False

    # ---- frames --------------------------------------------------------------------------
    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_started_at = time.time()
            self._ignored = 0
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._clear_pause()
            self._bot_speaking = False
            self._bot_stopped_at = time.time()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            return await self._on_vad_started()
        elif isinstance(frame, InterimTranscriptionFrame):
            if self._use_interim:
                return await self._on_transcript(frame, final=False)
        elif isinstance(frame, TranscriptionFrame):
            return await self._on_transcript(frame, final=True)
        return ProcessFrameResult.CONTINUE

    def _in_echo_tail(self) -> bool:
        return (
            self._echo_filter
            and self._echo_tail > 0
            and not self._bot_speaking
            and self._bot_stopped_at > 0
            and (time.time() - self._bot_stopped_at) < self._echo_tail
        )

    def _is_echo(self, text: str):
        if not self._echo_filter or self.bot_text is None:
            return False, 0.0
        ov = echo_overlap(text, self.bot_text.recent_text())
        return ov >= self._echo_overlap, round(ov, 2)

    def _real_words(self, text: str) -> int:
        return sum(1 for t in _tokens(text) if _letters(t) >= self._min_letters)

    def _has_interjection(self, text: str) -> bool:
        low = str(text or "").lower()
        if any(" " in p and p in low for p in self._interjection_phrases):
            return True
        return any(_skeleton(t) in self._interjections for t in _tokens(text))

    async def _on_vad_started(self) -> ProcessFrameResult:
        if not self._bot_speaking:
            if self._in_echo_tail():
                self._log("deferred", reason="vad_in_echo_tail", stage="vad")
                return ProcessFrameResult.CONTINUE
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        if self._pause_active:
            return ProcessFrameResult.CONTINUE
        in_grace = (time.time() - self._bot_started_at) < self._grace_secs
        if in_grace:
            self._log("ignored", reason="vad_in_grace_window", stage="vad")
            return ProcessFrameResult.CONTINUE
        if self._ignored >= self._max_ignored:
            self._log("cap", reason="vad_no_pause_cap_reached", stage="vad")
            return ProcessFrameResult.CONTINUE
        self._pause_active = True
        self._pause_started_at = time.time()
        await self.push_frame(BotOutputAudioPauseFrame(), FrameDirection.DOWNSTREAM)
        self._resume_task = self.create_task(self._resume_after_timeout(), name="astra_guard_resume")
        self._log("paused", reason="vad_wait_for_words", stage="vad", pause_secs=self._pause_secs)
        return ProcessFrameResult.CONTINUE

    async def _on_transcript(self, frame, *, final: bool) -> ProcessFrameResult:
        text = getattr(frame, "text", "") or ""
        conf = _confidence(frame)
        stage = "final" if final else "interim"
        if not self._bot_speaking:
            if not has_valid_words(text):
                return ProcessFrameResult.CONTINUE
            if self._in_echo_tail():
                echo, ov = self._is_echo(text)
                if echo:
                    self._log("ignored", reason="echo_tail", stage=stage, text=text[:80], overlap=ov)
                    await self.trigger_reset_aggregation()
                    return ProcessFrameResult.CONTINUE
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        if not has_valid_words(text):
            # noise / echo / empty STT while the bot speaks: not a barge-in, nothing to aggregate
            self._log("ignored", reason="no_words", stage="final" if final else "interim", text=text[:80], confidence=conf)
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE
        if self._echo_filter or self._min_words > 1 or self._min_letters > 1:
            echo, ov = self._is_echo(text)
            if echo:
                self._log("ignored", reason="echo", stage=stage, text=text[:80], overlap=ov, confidence=conf)
                await self.trigger_reset_aggregation()
                return ProcessFrameResult.CONTINUE
            words = self._real_words(text)
            if words < self._min_words and not self._has_interjection(text):
                self._log("ignored", reason="too_few_words", stage=stage, text=text[:80], words=words, overlap=ov, confidence=conf)
                await self.trigger_reset_aggregation()
                return ProcessFrameResult.CONTINUE
        waited = round((time.time() - self._pause_started_at) * 1000) if self._pause_active else None
        await self._cancel_resume_task()
        if self._pause_active:
            self._pause_active = False
            await self._resume_output_audio()
        self._log("accepted", reason="words", stage="final" if final else "interim", text=text[:80], confidence=conf, waited_ms=waited)
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    async def _resume_after_timeout(self):
        try:
            await asyncio.sleep(self._pause_secs)
            if not self._pause_active:
                return
            self._pause_active = False
            self._ignored += 1
            await self._resume_output_audio()
            self._log("ignored", reason="no_words_in_window", stage="timeout", resumed=True)
        except asyncio.CancelledError:
            return
        finally:
            if self._resume_task is asyncio.current_task():
                self._resume_task = None

    async def _resume_output_audio(self):
        await self.push_frame(BotOutputAudioResumeFrame(), FrameDirection.DOWNSTREAM)

    async def _cancel_resume_task(self):
        if self._resume_task and not self._resume_task.done():
            await self.cancel_task(self._resume_task)
        self._resume_task = None


def guard_from_config(run_configs: dict):
    """Return the guard strategy list when the workflow enables it, else None."""
    cfg = (run_configs or {}).get("astra_barge_in_guard")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    kw = {}
    if cfg.get("echo_filter"):
        kw["echo_filter"] = True
        kw["echo_overlap"] = float(cfg.get("echo_overlap", 0.6))
        kw["echo_tail_secs"] = float(cfg.get("echo_tail_secs", 0.0))
        kw["bot_text"] = BotTextTracker(window_secs=float(cfg.get("echo_window_secs", 12.0)))
    if "min_words_while_speaking" in cfg:
        kw["min_words_while_speaking"] = int(cfg["min_words_while_speaking"])
    if "min_word_letters" in cfg:
        kw["min_word_letters"] = int(cfg["min_word_letters"])
    if isinstance(cfg.get("interjections"), list):
        kw["interjections"] = [str(w) for w in cfg["interjections"]]
    return [
        AstraBargeInGuardStrategy(
            pause_secs=float(cfg.get("pause_secs", 1.2)),
            grace_secs=float(cfg.get("grace_secs", 0.3)),
            max_ignored_per_utterance=int(cfg.get("max_ignored_per_utterance", 1)),
            **kw,
        )
    ]


def bot_text_tap_for(strategies):
    """BotTextTapProcessor for a guard that needs the bot's text (echo filter), else None."""
    for s in strategies or []:
        if isinstance(s, AstraBargeInGuardStrategy) and s._echo_filter and s.bot_text is not None:
            return BotTextTapProcessor(s.bot_text)
    return None
