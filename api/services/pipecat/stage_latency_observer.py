"""Instrumentation only: per-turn stage timestamps for latency analysis.

Logs one line per pipeline stage event (wall clock, epoch seconds) as
``STAGE_LAT {"run":..,"ev":..,"t":..}``. Events:
  vad_stop / user_stopped  - caller speech end (VAD) / turn end decision
  stt_final                - final TranscriptionFrame from the STT service
  llm_request              - LLMContextFrame delivered to the LLM service
  llm_start / llm_first_token - LLMFullResponseStartFrame / first LLMTextFrame
  tts_request / tts_first_audio - TTSStartedFrame / first TTSAudioRawFrame
  bot_started              - BotStartedSpeakingFrame (audio leaving transport)
No settings are changed. Disable with env STAGE_LATENCY_LOG=false.
"""

from __future__ import annotations

import json
import os
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService


def stage_latency_enabled() -> bool:
    return os.getenv("STAGE_LATENCY_LOG", "true").strip().lower() not in ("0", "false", "no", "off")


class StageLatencyObserver(BaseObserver):
    def __init__(self, run_id, **kwargs):
        super().__init__(**kwargs)
        self._run = run_id
        self._seen: dict[int, None] = {}
        self._llm_first_pending = False
        self._tts_first_pending = False

    def _emit(self, ev: str, **extra) -> None:
        rec = {"run": self._run, "ev": ev, "t": round(time.time(), 4)}
        rec.update(extra)
        logger.info("STAGE_LAT " + json.dumps(rec, ensure_ascii=False))

    def _first_sight(self, frame) -> bool:
        fid = frame.id
        if fid in self._seen:
            return False
        self._seen[fid] = None
        if len(self._seen) > 4096:
            for k in list(self._seen)[:2048]:
                self._seen.pop(k, None)
        return True

    async def on_push_frame(self, data: FramePushed):
        try:
            f = data.frame
            src = data.source
            if isinstance(f, (VADUserStoppedSpeakingFrame, VADUserStartedSpeakingFrame,
                              UserStoppedSpeakingFrame, UserStartedSpeakingFrame,
                              BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InterruptionFrame)):
                if self._first_sight(f):
                    name = {VADUserStoppedSpeakingFrame: "vad_stop", VADUserStartedSpeakingFrame: "vad_start",
                            UserStoppedSpeakingFrame: "user_stopped", UserStartedSpeakingFrame: "user_started",
                            BotStartedSpeakingFrame: "bot_started", BotStoppedSpeakingFrame: "bot_stopped",
                            InterruptionFrame: "interruption"}[type(f)]
                    self._emit(name)
            elif isinstance(f, TranscriptionFrame) and isinstance(src, STTService):
                if self._first_sight(f):
                    extra = {"chars": len(f.text or "")}
                    res = getattr(f, "result", None)
                    if isinstance(res, dict):
                        lc = res.get("language_code") or res.get("language")
                        if lc:
                            extra["lang"] = str(lc)
                        conf = res.get("language_confidence")
                        if conf is None:
                            conf = res.get("language_probability")
                        if isinstance(conf, (int, float)):
                            extra["lang_conf"] = round(float(conf), 3)
                    if "lang" not in extra and getattr(f, "language", None) is not None:
                        extra["lang"] = str(getattr(f.language, "value", f.language))
                    self._emit("stt_final", **extra)
            elif isinstance(f, LLMContextFrame) and isinstance(data.destination, LLMService):
                if self._first_sight(f):
                    self._emit("llm_request")
            elif isinstance(f, LLMFullResponseStartFrame) and isinstance(src, LLMService):
                if self._first_sight(f):
                    self._llm_first_pending = True
                    self._emit("llm_start")
            elif isinstance(f, LLMTextFrame) and isinstance(src, LLMService):
                if self._llm_first_pending and (f.text or "").strip() and self._first_sight(f):
                    self._llm_first_pending = False
                    self._emit("llm_first_token")
            elif isinstance(f, TTSStartedFrame) and isinstance(src, TTSService):
                if self._first_sight(f):
                    self._tts_first_pending = True
                    self._emit("tts_request")
            elif isinstance(f, TTSAudioRawFrame) and isinstance(src, TTSService):
                if self._tts_first_pending and self._first_sight(f):
                    self._tts_first_pending = False
                    self._emit("tts_first_audio")
        except Exception as e:  # instrumentation must never break a call
            logger.debug(f"StageLatencyObserver error: {e}")
