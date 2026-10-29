"""Astra: Sarvam cloned-voice TTS (separate runtime path from curated Bulbul voices).

Curated voices keep the existing SarvamTTSService on /text-to-speech/ws.
Cloned voices (model "voice-clone", voice "svc-<uuid>") use the documented
voice-cloning WebSocket wss://api.sarvam.ai/voices/clone/ws, whose protocol
matches Bulbul (config -> text/flush -> audio/final). Docs:
https://docs.sarvam.ai/api-reference/voice-cloning/clone-ws

Rules: never cross ids (a curated speaker can't go to the clone path and a
svc- id can't go to Bulbul); no fallback voice; pace = employee speaking rate;
output = raw 16-bit PCM at the transport sample rate (8 kHz on PSTN; the
telephony serializer encodes mu-law), identical to the curated path.
"""
import json
import re
from dataclasses import replace as _dc_replace

from loguru import logger
from pipecat.services.sarvam import tts as _sarvam_tts
from pipecat.services.sarvam.tts import SarvamTTSService

CLONE_MODEL = "voice-clone"
CLONE_WS_URL = "wss://api.sarvam.ai/voices/clone/ws?send_completion_event=true"
UNAVAILABLE = "Selected voice is unavailable. Choose another voice."
_SVC_RE = re.compile(r"^svc-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_clone_request(model, voice) -> bool:
    v = str(voice or "").strip().lower()
    return str(model or "") == CLONE_MODEL or v.startswith("svc-") or v.startswith("cloned:")


def validate_clone_selection(model, voice) -> str:
    """Return the provider voice id or raise (never substitute another voice)."""
    v = str(voice or "").strip()
    if str(model or "") != CLONE_MODEL or not _SVC_RE.match(v):
        raise ValueError(UNAVAILABLE)
    return v


def _ensure_profile():
    cfgs = _sarvam_tts.TTS_MODEL_CONFIGS
    if CLONE_MODEL not in cfgs:
        base = cfgs.get("bulbul:v3")
        cfgs[CLONE_MODEL] = _dc_replace(base, supports_temperature=False, speakers=())


class SarvamClonedTTSService(SarvamTTSService):
    """Bulbul-compatible WS client pointed at the voice-cloning endpoint."""

    async def _send_config(self):
        if not self._websocket:
            raise Exception("WebSocket not connected")
        data = {
            "target_language_code": self._settings.language,
            "voice_id": self._settings.voice,
            "speech_sample_rate": self._speech_sample_rate,
            "output_audio_codec": self._output_audio_codec,
            "min_buffer_size": self._settings.min_buffer_size,
            "max_chunk_length": self._settings.max_chunk_length,
            "pace": self._settings.pace,
        }
        logger.debug(
            f"Cloned TTS config lang={data['target_language_code']} rate={data['speech_sample_rate']} "
            f"codec={data['output_audio_codec']} pace={data['pace']}"
        )
        await self._websocket.send(json.dumps({"type": "config", "data": data}))


def build_cloned_tts(*, api_key, model, voice, pipecat_language, speed, min_buffer_size, settings_cls, **tts_kwargs):
    voice_id = validate_clone_selection(model, voice)
    _ensure_profile()
    kw = {"model": CLONE_MODEL, "voice": voice_id, "language": pipecat_language}
    if speed and float(speed) != 1.0:
        kw["pace"] = float(speed)
    try:
        kw["min_buffer_size"] = max(50, int(min_buffer_size)) if min_buffer_size is not None else 50
        settings = settings_cls(**kw)
    except (TypeError, ValueError):
        kw.pop("min_buffer_size", None)
        settings = settings_cls(**kw)
    svc = SarvamClonedTTSService(api_key=api_key, settings=settings, **tts_kwargs)
    # pipecat appends ?model=...; the clone WS takes only send_completion_event.
    svc._websocket_url = CLONE_WS_URL
    logger.info(f"Sarvam CLONED TTS live config: voice={voice_id[:12]}… pace={kw.get('pace', 1.0)}")
    return svc
