"""Platform-wide per-turn Sarvam TTS language + short-utterance STT finalizer.

Primary / WF static tts.language is greeting/default only until caller language
is known. After every caller turn, resolve language with:

  1. Reliable STT detected-language metadata/confidence
  2. Strong unique script evidence (Telugu/Tamil/… — not blind Devanagari→hi)
  3. Current conversational / previous confirmed caller language
  4. Employee primary_language fallback

Devanagari alone must NOT force hi-IN when Deepgram multilingual confidence is
weak/ambiguous. Ambiguous Devanagari retains current session/primary language (never
blind hi-IN). Clear Hindi still switches to hi-IN; English to en-IN.

Also promotes last stable InterimTranscriptionFrame once after end-of-speech
when Deepgram never emits a final (short valid utterances). No fake words.
Speaker / voice_id stay locked. Not Maya-specific.
Chain: primary→greeting only; resolved caller lang→LLM instruction + TTS.
Reply-path LLM script detection must NOT override a confident resolved
caller-turn language; it is fallback only when caller language is unresolved.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMMessagesTransformFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# Detected / reply BCP-47 (or Deepgram short codes) → Pipecat Language.
_LANG_TO_PIPECAT: dict[str, Language] = {
    "te": Language.TE_IN,
    "te-in": Language.TE_IN,
    "telugu": Language.TE_IN,
    "en": Language.EN_IN,
    "en-in": Language.EN_IN,
    "en-us": Language.EN_IN,
    "english": Language.EN_IN,
    "hi": Language.HI_IN,
    "hi-in": Language.HI_IN,
    "hindi": Language.HI_IN,
    "hinglish": Language.HI_IN,
    "ta": Language.TA_IN,
    "ta-in": Language.TA_IN,
    "kn": Language.KN_IN,
    "kn-in": Language.KN_IN,
    "ml": Language.ML_IN,
    "ml-in": Language.ML_IN,
    "mr": Language.MR_IN,
    "mr-in": Language.MR_IN,
    "bn": Language.BN_IN,
    "bn-in": Language.BN_IN,
    "gu": Language.GU_IN,
    "gu-in": Language.GU_IN,
    "pa": Language.PA_IN,
    "pa-in": Language.PA_IN,
    "or": Language.OR_IN,
    "or-in": Language.OR_IN,
    "od": Language.OR_IN,
    "od-in": Language.OR_IN,
}

# Unique Indic scripts (NOT Devanagari — that is ambiguous hi/mr + ASR confusion).
_UNIQUE_SCRIPT_RANGES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\u0C00-\u0C7F]"), "te-IN"),  # Telugu
    (re.compile(r"[\u0B80-\u0BFF]"), "ta-IN"),  # Tamil
    (re.compile(r"[\u0C80-\u0CFF]"), "kn-IN"),  # Kannada
    (re.compile(r"[\u0D00-\u0D7F]"), "ml-IN"),  # Malayalam
    (re.compile(r"[\u0980-\u09FF]"), "bn-IN"),  # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu-IN"),  # Gujarati
    (re.compile(r"[\u0A00-\u0A7F]"), "pa-IN"),  # Gurmukhi
]

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# Reply-path script list keeps Devanagari for genuine Hindi LLM replies, but
# resolve_turn_language applies continuity before trusting it for STT.
_SCRIPT_RANGES: list[tuple[re.Pattern[str], str]] = [
    *_UNIQUE_SCRIPT_RANGES,
    (_DEVANAGARI_RE, "hi-IN"),
]

# Clear Hindi cues (Devanagari function words / common particles).
_HINDI_CLEAR_RE = re.compile(
    r"(आप|हमारी|हमारा|हमारे|क्या|हाँ|नहीं|नही|हैं|है|में|के लिए|सकता|सकती|सकते|"
    r"मदद|कृपया|नमस्ते|जी|अच्छा|ठीक|कैसे|कहाँ|कब|क्यों)"
)

# Clear Latin English (enough letters, no Indic script).
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")

# Minimum STT alternative confidence to trust language metadata alone.
_STT_LANG_CONFIDENCE_MIN = 0.55

# Delay after VAD/user stop before promoting last interim (lets Deepgram finalize).
_INTERIM_PROMOTE_DELAY_S = 0.45

_DEFAULT_ALLOWED = ("te-IN", "en-IN", "hi-IN")


def _normalize_key(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, Language):
        raw = raw.value if hasattr(raw, "value") else str(raw)
    s = str(raw).strip()
    if not s or s.lower() in ("multi", "auto", "unknown", "none"):
        return ""
    return s.lower().replace("_", "-")


def map_to_pipecat_language(raw: object) -> Optional[Language]:
    """Map STT/LLM/config language tag to Pipecat Language for Sarvam TTS."""
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _LANG_TO_PIPECAT:
        return _LANG_TO_PIPECAT[key]
    base = key.split("-", 1)[0]
    return _LANG_TO_PIPECAT.get(base)


def _lang_tag(lang: Optional[Language]) -> str:
    if lang is None:
        return ""
    return lang.value if hasattr(lang, "value") else str(lang)


def _has_unique_indic_script(text: str) -> Optional[str]:
    raw = str(text or "")
    for pattern, tag in _UNIQUE_SCRIPT_RANGES:
        if pattern.search(raw):
            return tag
    return None


def _has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI_RE.search(str(text or "")))


def _has_telugu_script(text: str) -> bool:
    return bool(_TELUGU_RE.search(str(text or "")))


def _clear_hindi(text: str) -> bool:
    return bool(_HINDI_CLEAR_RE.search(str(text or "")))


def _latin_english(text: str) -> bool:
    raw = str(text or "")
    if any(p.search(raw) for p, _ in _SCRIPT_RANGES):
        return False
    return bool(_LATIN_WORD_RE.search(raw))


def _extract_stt_confidence(result: Any) -> Optional[float]:
    """Best-effort confidence from Deepgram ListenV1Results (or similar)."""
    if result is None:
        return None
    try:
        channel = getattr(result, "channel", None)
        alts = getattr(channel, "alternatives", None) if channel is not None else None
        if not alts:
            # dict-shaped
            if isinstance(result, dict):
                ch = result.get("channel") or {}
                alts = ch.get("alternatives") or []
            else:
                return None
        if not alts:
            return None
        alt0 = alts[0]
        conf = getattr(alt0, "confidence", None)
        if conf is None and isinstance(alt0, dict):
            conf = alt0.get("confidence")
        if conf is None:
            return None
        return float(conf)
    except Exception:
        return None


def detect_language_from_text(text: str) -> Optional[Language]:
    """Script-based detection (reply path). Prefer unique Indic, then Devanagari, then Latin."""
    raw = str(text or "")
    if not raw.strip():
        return None
    unique = _has_unique_indic_script(raw)
    if unique:
        return map_to_pipecat_language(unique)
    if _has_devanagari(raw):
        return map_to_pipecat_language("hi-IN")
    if _latin_english(raw):
        return Language.EN_IN
    return None


@dataclass
class SessionLanguageState:
    """Shared per-call conversational language (STT + reply processors)."""

    primary_language: Optional[Language] = None
    allowed_languages: set[str] = field(default_factory=lambda: set(_DEFAULT_ALLOWED))
    confirmed_caller_language: Optional[Language] = None  # sticky mid-call
    # Config lock (workflow auto_language_switch=false / language_mode=locked):
    # every caller turn resolves to primary_language; no mid-call switching.
    language_locked: bool = False
    # Astra WF30 staging (round 2 fix 4): follow the caller's latest clear language.
    # Only set when workflow config astra_language_policy.follow_latest_caller is true.
    follow_latest_caller: bool = False

    @property
    def current_language(self) -> Optional[Language]:
        """Active conversational language: confirmed caller, else primary default."""
        return self.confirmed_caller_language or self.primary_language

    def allow(self, lang: Optional[Language]) -> Optional[Language]:
        if lang is None:
            return None
        tag = _lang_tag(lang)
        if not self.allowed_languages or tag in self.allowed_languages:
            return lang
        # Base match (te vs te-IN)
        base = tag.split("-", 1)[0].lower()
        for allowed in self.allowed_languages:
            if allowed.lower().startswith(base):
                return map_to_pipecat_language(allowed)
        return None


def resolve_turn_language(
    *,
    stt_language: object = None,
    text: str = "",
    stt_confidence: Optional[float] = None,
    session: Optional[SessionLanguageState] = None,
    preserve_original_transcript: bool = True,
) -> tuple[Optional[Language], str]:
    """Resolve turn language with platform precedence.

    Returns (language, reason). Never mutates/translates ``text``; caller keeps
    the original transcript for logs/LLM.
    """
    _ = preserve_original_transcript  # documented contract
    raw = str(text or "")
    session = session or SessionLanguageState()
    primary = session.primary_language
    confirmed = session.confirmed_caller_language
    primary_tag = _lang_tag(primary)
    confirmed_tag = _lang_tag(confirmed)

    # 0) Config language lock: honor auto_language_switch=false / language_mode=locked
    # generically — STT metadata / script never switch the conversation language.
    if getattr(session, "language_locked", False) and primary is not None:
        return primary, "language_locked"

    stt_mapped = map_to_pipecat_language(stt_language)
    unique_tag = _has_unique_indic_script(raw)
    has_deva = _has_devanagari(raw)
    clear_hi = has_deva and _clear_hindi(raw)
    latin = _latin_english(raw)

    # 1) Reliable STT metadata
    if stt_mapped is not None:
        stt_tag = _lang_tag(stt_mapped)
        conf_ok = stt_confidence is None or stt_confidence >= _STT_LANG_CONFIDENCE_MIN
        # Unique script agreeing with STT → trust
        if unique_tag and map_to_pipecat_language(unique_tag) == stt_mapped:
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_script_agree"
        # Clear Hindi text + STT hi → trust
        if stt_tag.startswith("hi") and clear_hi and conf_ok:
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_clear_hindi"
        # STT te / unique Indic → trust
        if stt_tag.startswith("te") or (
            unique_tag and stt_tag.startswith(unique_tag.split("-")[0].lower())
        ):
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_indic"
        # STT en + Latin text → trust
        if stt_tag.startswith("en") and latin and conf_ok:
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_english"
        # STT hi + Devanagari but NOT clear Hindi, while session/primary is Telugu
        # → do NOT trust blind hi (Deepgram Telugu→Devanagari failure mode)
        if stt_tag.startswith("hi") and has_deva and not clear_hi:
            follow = getattr(session, "follow_latest_caller", False)
            if follow and confirmed_tag.startswith("hi"):
                # caller already speaking Hindi: Hindi stays Hindi
                return session.allow(stt_mapped) or stt_mapped, "stt_hi_session_follow"
            if confirmed_tag.startswith("te") or (
                primary_tag.startswith("te") and not (follow and confirmed is not None)
            ):
                prefer = confirmed if confirmed_tag.startswith("te") else primary
                allowed = session.allow(prefer)
                if allowed is not None:
                    return allowed, "stt_hi_rejected_te_continuity"
            if not conf_ok:
                # fall through to script / continuity
                pass
            elif confirmed_tag.startswith("hi"):
                return session.allow(stt_mapped) or stt_mapped, "stt_meta_hi_session"
            else:
                # Ambiguous: don't lock hi yet — fall through
                pass
        elif conf_ok and not has_deva:
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_confident"
        elif conf_ok and clear_hi:
            return session.allow(stt_mapped) or stt_mapped, "stt_meta_confident_hi"

    # 2) Strong unique script (never Devanagari alone)
    if unique_tag:
        mapped = map_to_pipecat_language(unique_tag)
        allowed = session.allow(mapped)
        if allowed is not None:
            return allowed, "unique_script"

    # Clear Hindi Devanagari
    if clear_hi:
        mapped = map_to_pipecat_language("hi-IN")
        allowed = session.allow(mapped)
        if allowed is not None:
            return allowed, "clear_hindi_script"

    # Latin English
    if latin:
        mapped = Language.EN_IN
        allowed = session.allow(mapped)
        if allowed is not None:
            return allowed, "latin_english"

    # 3) Ambiguous Devanagari → RETAIN current confirmed / primary.
    # Devanagari script alone must NEVER force hi-IN. Only clear_hindi above
    # (or strong STT meta) may switch to Hindi.
    if has_deva:
        if confirmed is not None:
            allowed = session.allow(confirmed)
            if allowed is not None:
                return allowed, "ambiguous_deva_session_continuity"
        if primary is not None:
            allowed = session.allow(primary)
            if allowed is not None:
                return allowed, "ambiguous_deva_primary_continuity"
        return None, "ambiguous_deva_unresolved"

    # 3b) Previous confirmed caller language
    if confirmed is not None:
        allowed = session.allow(confirmed)
        if allowed is not None:
            return allowed, "confirmed_caller"

    # 4) Primary fallback
    if primary is not None:
        allowed = session.allow(primary)
        if allowed is not None:
            return allowed, "primary_fallback"

    # Legacy script/Latin fallback
    legacy = detect_language_from_text(raw)
    if legacy is not None:
        allowed = session.allow(legacy)
        return (allowed or legacy), "legacy_script"

    return None, "unresolved"


def _primary_from_tts(tts: Any) -> Optional[Language]:
    """Read greeting/default language from live TTS service settings."""
    if tts is None:
        return None
    settings = getattr(tts, "_settings", None)
    if settings is None:
        return None
    raw = getattr(settings, "language", None)
    return map_to_pipecat_language(raw)



_LANG_ALIGN_MARKER = "#LANGUAGE_ALIGNMENT#"
_LANG_REPLY_NAMES = {
    "te-IN": "Telugu",
    "te": "Telugu",
    "en-IN": "English",
    "en": "English",
    "hi-IN": "Hindi",
    "hi": "Hindi",
}


def _language_align_instruction(lang: Language) -> str:
    """Minimal per-turn runtime instruction; does not rewrite employee prompts."""
    tag = _lang_tag(lang) or "en-IN"
    base = tag.split("-", 1)[0]
    name = _LANG_REPLY_NAMES.get(tag) or _LANG_REPLY_NAMES.get(base, tag)
    return (
        f"{_LANG_ALIGN_MARKER} Resolved caller language for this turn: {tag} ({name}). "
        f"You MUST reply entirely in {name}. "
        f"Keep the original caller transcript unchanged in context; answer in {name} only."
    )


def _upsert_language_align_messages(messages, lang: Language):
    """Replace sticky language-alignment system message; leave all other messages intact."""
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, str) and _LANG_ALIGN_MARKER in content:
                continue
            if isinstance(content, list):
                joined = " ".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                )
                if _LANG_ALIGN_MARKER in joined:
                    continue
        out.append(m)
    out.append({"role": "system", "content": _language_align_instruction(lang)})
    return out


class MultilingualTtsLanguageProcessor(FrameProcessor):
    """Push TTSUpdateSettingsFrame(language=…) per turn; never change voice."""

    def __init__(
        self,
        *,
        source: str = "stt",
        session: Optional[SessionLanguageState] = None,
    ):
        super().__init__()
        self._source = source
        self._session = session or SessionLanguageState()
        # Seed current language from primary so greeting/default matches config
        # before any caller turn (current_language == primary_language at init).
        primary = self._session.primary_language
        self._current_lang: Optional[str] = _lang_tag(primary) if primary is not None else None
        self._reply_buf = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and self._source in ("stt", "both"):
            conf = _extract_stt_confidence(getattr(frame, "result", None))
            text = getattr(frame, "text", "") or ""
            lang, reason = resolve_turn_language(
                stt_language=getattr(frame, "language", None),
                text=text,
                stt_confidence=conf,
                session=self._session,
            )
            # Preserve original transcript on the frame; optionally stamp resolved lang
            # for downstream logging without rewriting text.
            if lang is not None:
                try:
                    frame.language = lang
                except Exception:
                    pass
                # Confirm sticky caller language on STT turns (not greeting).
                self._session.confirmed_caller_language = lang
                # B) Pass resolved caller lang into LLM turn (runtime instruction only).
                await self.push_frame(
                    LLMMessagesTransformFrame(
                        transform=lambda msgs, _lang=lang: _upsert_language_align_messages(
                            msgs, _lang
                        ),
                        run_llm=False,
                    ),
                    direction,
                )
            await self._maybe_update(lang, reason=f"stt_transcription:{reason}")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame) and self._source in ("reply", "both"):
            self._reply_buf = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and self._source in ("reply", "both"):
            text = getattr(frame, "text", "") or ""
            self._reply_buf += text
            resolved = self._session.confirmed_caller_language
            unique = _has_unique_indic_script(self._reply_buf)
            clear_hi = _has_devanagari(self._reply_buf) and _clear_hindi(self._reply_buf)
            llm_lang: Optional[Language] = None
            if unique:
                llm_lang = map_to_pipecat_language(unique)
            elif clear_hi:
                llm_lang = map_to_pipecat_language("hi-IN")
            elif _has_devanagari(self._reply_buf):
                llm_lang = map_to_pipecat_language("hi-IN")
            elif len(self._reply_buf) >= 24 and _latin_english(self._reply_buf):
                llm_lang = Language.EN_IN

            # Resolved caller-turn language is authoritative for TTS this turn.
            # Reply-text detection must NOT override a confident resolved lang
            # (run 224: te-IN overwritten by reply_clear_hindi).
            if (
                getattr(self._session, "follow_latest_caller", False)
                and llm_lang is not None
                and self._session.allow(llm_lang) is not None
                and (resolved is None or _lang_tag(llm_lang) != _lang_tag(resolved))
            ):
                # WF30: the reply language is decided per turn from the caller's latest clear
                # language (explicit request, script, full English sentence); speak it in that
                # language instead of the STT-resolved one (run 477 turn 7: Hindi -> Telugu voice)
                await self._maybe_update(
                    self._session.allow(llm_lang), reason="reply_follow_latest_caller"
                )
                await self.push_frame(frame, direction)
                return
            if resolved is not None:
                await self._maybe_update(
                    resolved, reason="reply_keep_resolved_turn"
                )
                if (
                    llm_lang is not None
                    and _lang_tag(llm_lang) != _lang_tag(resolved)
                ):
                    logger.info(
                        "LANGUAGE_ALIGNMENT_MISMATCH "
                        f"resolved_turn_language={_lang_tag(resolved)!r} "
                        f"llm_response_language={_lang_tag(llm_lang)!r} "
                        f"expected_llm_language={_lang_tag(resolved)!r} "
                        f"final_tts_language={_lang_tag(resolved)!r}"
                    )
                await self.push_frame(frame, direction)
                return

            # Caller-turn language unresolved — reply detection is fallback only.
            if unique:
                await self._maybe_update(
                    map_to_pipecat_language(unique), reason="reply_unique_script"
                )
            elif clear_hi:
                await self._maybe_update(
                    map_to_pipecat_language("hi-IN"), reason="reply_clear_hindi"
                )
            elif _has_devanagari(self._reply_buf):
                prefer = self._session.primary_language
                if prefer is not None:
                    await self._maybe_update(
                        prefer, reason="reply_deva_primary_fallback"
                    )
            elif len(self._reply_buf) >= 24 and _latin_english(self._reply_buf):
                await self._maybe_update(Language.EN_IN, reason="reply_latin")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSSpeakFrame) and self._source in ("reply", "both"):
            # Resolved caller-turn language wins; else primary greeting/default.
            # Do not re-detect speak-text language when a turn is already resolved.
            if self._session.confirmed_caller_language is not None:
                await self._maybe_update(
                    self._session.confirmed_caller_language,
                    reason="tts_speak:keep_resolved_turn",
                )
            elif self._session.primary_language is not None:
                await self._maybe_update(
                    self._session.primary_language,
                    reason="tts_speak:primary_greeting",
                )
            else:
                lang, reason = resolve_turn_language(
                    text=getattr(frame, "text", "") or "",
                    session=self._session,
                )
                await self._maybe_update(lang, reason=f"tts_speak:{reason}")
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _maybe_update(self, lang: Optional[Language], *, reason: str) -> None:
        if lang is None:
            return
        allowed = self._session.allow(lang)
        if allowed is None:
            return
        lang = allowed
        key = _lang_tag(lang)
        if key == self._current_lang:
            return
        prev = self._current_lang
        self._current_lang = key
        await self.push_frame(
            TTSUpdateSettingsFrame(delta=TTSSettings(language=lang)),
            FrameDirection.DOWNSTREAM,
        )
        logger.info(
            f"multilingual_tts_language[{self._source}]: "
            f"TTS language {prev!r} → {key!r} ({reason})"
        )


class InterimTranscriptFinalizer(FrameProcessor):
    """Promote last stable interim → final once when Deepgram never finalizes.

    Place after STT, before MultilingualTtsLanguageProcessor(source=stt) and
    before the user aggregator. Does not invent text. Suppresses duplicate
    finals that match a just-promoted interim. Cancels on new speech / hangup.
    """

    def __init__(self, *, promote_delay_s: float = _INTERIM_PROMOTE_DELAY_S):
        super().__init__()
        self._promote_delay_s = promote_delay_s
        self._last_interim: Optional[InterimTranscriptionFrame] = None
        self._promote_task: Optional[asyncio.Task] = None
        self._final_seen_this_utterance = False
        self._last_promoted_norm: Optional[str] = None
        self._hanging_up = False

    def _norm(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).lower()

    def _cancel_promote(self) -> None:
        task = self._promote_task
        self._promote_task = None
        if task is not None and not task.done():
            task.cancel()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._hanging_up = True
            self._cancel_promote()
            self._last_interim = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._cancel_promote()
            self._final_seen_this_utterance = False
            # Keep last interim until replaced — user may still be mid-phrase.
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterimTranscriptionFrame):
            text = (getattr(frame, "text", "") or "").strip()
            if text:
                self._last_interim = frame
                self._final_seen_this_utterance = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            text = (getattr(frame, "text", "") or "").strip()
            norm = self._norm(text)
            # Duplicate of just-promoted interim → drop to avoid double LLM/TTS.
            if (
                text
                and self._last_promoted_norm
                and norm == self._last_promoted_norm
            ):
                logger.info(
                    "interim_finalizer: suppressing duplicate final matching "
                    f"promoted interim ({norm[:80]!r})"
                )
                self._cancel_promote()
                self._final_seen_this_utterance = True
                self._last_interim = None
                self._last_promoted_norm = None
                return
            self._cancel_promote()
            self._final_seen_this_utterance = True
            self._last_interim = None
            self._last_promoted_norm = None
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            await self.push_frame(frame, direction)
            if (
                not self._final_seen_this_utterance
                and self._last_interim is not None
                and not self._hanging_up
            ):
                self._schedule_promote()
            return

        await self.push_frame(frame, direction)

    def _schedule_promote(self) -> None:
        self._cancel_promote()
        try:
            self._promote_task = asyncio.create_task(self._promote_after_delay())
        except Exception as exc:
            logger.warning(f"interim_finalizer: could not schedule promote: {exc}")

    async def _promote_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._promote_delay_s)
        except asyncio.CancelledError:
            return
        if self._hanging_up or self._final_seen_this_utterance:
            return
        interim = self._last_interim
        if interim is None:
            return
        text = (getattr(interim, "text", "") or "").strip()
        if not text:
            return
        # Promote once.
        self._final_seen_this_utterance = True
        self._last_promoted_norm = self._norm(text)
        self._last_interim = None
        promoted = TranscriptionFrame(
            text,
            getattr(interim, "user_id", "") or "",
            time_now_iso8601(),
            getattr(interim, "language", None),
            result={"promoted_from_interim": True, "source": "interim_finalizer"},
            finalized=True,
        )
        logger.info(
            f"interim_finalizer: promoted interim → final ({text[:80]!r})"
        )
        await self.push_frame(promoted, FrameDirection.DOWNSTREAM)
