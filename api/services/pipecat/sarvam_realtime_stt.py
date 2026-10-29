"""Generic Sarvam realtime streaming STT (``/speech-to-text-realtime/ws``).

Supports ``saaras:v4`` and ``saaras:v3-realtime``: raw PCM in, ``transcript.partial``
-> InterimTranscriptionFrame, ``transcript.final`` -> finalized TranscriptionFrame.
End-of-utterance is decided by Sarvam's server VAD (``silence_duration_ms``);
local VAD / turn analyzer still own turn-taking and interruptions.
No workflow-specific logic lives here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse
from collections.abc import AsyncGenerator

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODELS = ("saaras:v4", "saaras:v3-realtime")
DEFAULT_SILENCE_MS = 500

_LANG = {
    "bn-IN": Language.BN_IN, "gu-IN": Language.GU_IN, "hi-IN": Language.HI_IN,
    "kn-IN": Language.KN_IN, "ml-IN": Language.ML_IN, "mr-IN": Language.MR_IN,
    "ta-IN": Language.TA_IN, "te-IN": Language.TE_IN, "pa-IN": Language.PA_IN,
    "od-IN": Language.OR_IN, "en-IN": Language.EN_IN, "as-IN": Language.AS_IN,
}


def is_realtime_model(model: str | None) -> bool:
    return bool(model) and str(model) in REALTIME_MODELS


def build_realtime_query(
    *,
    model: str,
    language_code: str | None,
    sample_rate: int,
    mode: str | None = None,
    silence_duration_ms: int | None = None,
    stream_type: str | None = None,
    keyterms: list[str] | None = None,
) -> list[tuple[str, str]]:
    q: list[tuple[str, str]] = [
        ("model", model),
        ("encoding", "linear16"),
        ("sample_rate", str(sample_rate)),
        ("endpointing", "vad"),
    ]
    if language_code and language_code != "unknown":
        q.append(("language_code", language_code))
    if mode:
        q.append(("mode", mode))
    if stream_type:
        q.append(("stream_type", stream_type))
    if silence_duration_ms:
        q.append(("silence_duration_ms", str(int(silence_duration_ms))))
    if keyterms:
        q.append(("keyterms", json.dumps(list(keyterms), ensure_ascii=False)))
    return q


class SarvamRealtimeSTTService(WebsocketSTTService):
    """Sarvam realtime streaming STT over a raw websocket."""

    # Every utterance ends with transcript.final, so interims must not be promoted.
    guarantees_final_transcripts = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saaras:v4",
        language_code: str | None = "unknown",
        mode: str | None = None,
        silence_duration_ms: int | None = None,
        stream_type: str | None = None,
        keyterms: list[str] | None = None,
        url: str = REALTIME_URL,
        sample_rate: int | None = None,
        **kwargs,
    ):
        silence_s = (silence_duration_ms or DEFAULT_SILENCE_MS) / 1000.0
        super().__init__(
            sample_rate=sample_rate,
            # Safety net for the turn stop strategy: server silence + network/model.
            ttfs_p99_latency=round(silence_s + 0.9, 3),
            keepalive_timeout=5,
            keepalive_interval=5,
            settings=STTSettings(model=model, language=language_code),
            **kwargs,
        )
        self._api_key = api_key
        self._url = url
        self._model = model
        self._language_code = language_code
        self._mode = mode
        self._silence_ms = silence_duration_ms
        self._stream_type = stream_type
        self._keyterms = keyterms
        self._receive_task = None
        # Finalization re-assert: a local-VAD segment that produced no STT output
        # (noise/echo blip after the final) must not leave the user turn waiting
        # for a final that Sarvam will never send.
        self._final_in_turn = False
        self._stt_activity_since_vad_start = False
        self._reassert_task: asyncio.Task | None = None
        self.reassert_grace_s = 0.35

    def can_generate_metrics(self) -> bool:
        return True

    def _language_enum(self, code: str | None) -> Language:
        return _LANG.get(code or self._language_code or "", Language.TE_IN if (self._language_code or "").startswith("te") else Language.EN_IN)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._send_json({"event": "end"})
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.on_vad_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.on_vad_stopped()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.on_user_turn_stopped()

    def on_vad_started(self):
        self._stt_activity_since_vad_start = False
        self._cancel_reassert()

    def on_vad_stopped(self):
        if self._final_in_turn and not self._stt_activity_since_vad_start:
            self._cancel_reassert()
            self._reassert_task = asyncio.create_task(self._reassert_after_grace())

    def on_user_turn_stopped(self):
        self._final_in_turn = False
        self._cancel_reassert()

    def _cancel_reassert(self):
        t = self._reassert_task
        self._reassert_task = None
        if t is not None and not t.done():
            t.cancel()

    async def _reassert_after_grace(self):
        try:
            await asyncio.sleep(self.reassert_grace_s)
        except asyncio.CancelledError:
            return
        if self._final_in_turn and not self._stt_activity_since_vad_start:
            logger.info(f"{self}: no speech in last VAD segment; re-asserting finalized transcript")
            # Whitespace-only text: aggregators ignore it (no duplicate words),
            # turn-stop strategies see a finalized transcript and can release.
            await self.push_frame(
                TranscriptionFrame(text=" ", user_id=self._user_id, timestamp=time_now_iso8601(),
                                   language=self._language_enum(None), finalized=True)
            )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        await self._send_json(
            {"event": "audio_input", "audio": base64.b64encode(audio).decode("ascii")}
        )
        yield None

    async def _send_json(self, payload: dict):
        ws = self._websocket
        if ws is not None and ws.state is State.OPEN:
            try:
                await ws.send(json.dumps(payload))
            except Exception as e:  # pragma: no cover - network
                logger.warning(f"{self}: send failed: {e}")

    async def _send_keepalive(self, silence: bytes):
        await self._send_json(
            {"event": "audio_input", "audio": base64.b64encode(silence).decode("ascii")}
        )

    def _build_url(self) -> str:
        q = build_realtime_query(
            model=self._model,
            language_code=self._language_code,
            sample_rate=self.sample_rate,
            mode=self._mode,
            silence_duration_ms=self._silence_ms,
            stream_type=self._stream_type,
            keyterms=self._keyterms,
        )
        return self._url + "?" + urllib.parse.urlencode(q)

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._websocket = await self._websocket_connect(
                self._build_url(),
                additional_headers={"api-subscription-key": self._api_key},
                max_size=None,
            )
            logger.info(
                f"Connected to Sarvam realtime STT model={self._model} lang={self._language_code} "
                f"mode={self._mode} silence_ms={self._silence_ms or DEFAULT_SILENCE_MS}"
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Sarvam realtime STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.debug(f"{self}: close error {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                await self.handle_message(json.loads(message))
            except json.JSONDecodeError:
                logger.warning(f"{self}: non-JSON message")
            except Exception as e:
                logger.warning(f"{self}: error processing message: {e}")

    async def handle_message(self, d: dict):
        """Map one realtime server event to pipecat frames (public for tests)."""
        ev = d.get("event") or d.get("type")
        text = (d.get("text") or d.get("transcript") or "").strip()
        lang = self._language_enum(d.get("language_code") or d.get("language"))
        if ev in ("transcript.partial", "transcript.final") and text:
            self._stt_activity_since_vad_start = True
            self._cancel_reassert()
        if ev == "transcript.partial":
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(text=text, user_id=self._user_id,
                                              timestamp=time_now_iso8601(), language=lang, result=d)
                )
        elif ev == "transcript.final":
            if text:
                self._final_in_turn = True
                await self.emit_stt_usage_metrics()
                await self.push_frame(
                    TranscriptionFrame(text=text, user_id=self._user_id, timestamp=time_now_iso8601(),
                                       language=lang, result=d, finalized=True)
                )
                await self._handle_transcription(text, True, lang)
            await self.stop_processing_metrics()
        elif ev == "error":
            msg = {k: d.get(k) for k in ("code", "message", "is_fatal")}
            logger.warning(f"{self}: Sarvam realtime error {msg}")
            if d.get("is_fatal"):
                await self.push_error(error_msg=f"Sarvam realtime STT error: {msg}")

    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        pass
