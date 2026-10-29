import os

from loguru import logger

from api.services.pipecat.audio_config import AudioConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.utils.run_context import turn_var

from api.services.pipecat.multilingual_tts_language import (
    InterimTranscriptFinalizer,
    MultilingualTtsLanguageProcessor,
    SessionLanguageState,
    _DEFAULT_ALLOWED,
    map_to_pipecat_language,
)


def create_pipeline_components(audio_config: AudioConfig):
    """Create and return the main pipeline components with proper audio configuration"""
    logger.info(f"Creating pipeline components with audio config: {audio_config}")

    # Use native AudioBufferProcessor for merged audio recording
    audio_buffer = AudioBufferProcessor(
        sample_rate=audio_config.pipeline_sample_rate,
        buffer_size=audio_config.buffer_size_bytes,
    )

    context = LLMContext()

    return audio_buffer, context


def _resolve_employee_primary_language(
    *,
    workflow_configurations=None,
    call_context_vars=None,
):
    """Canonical employee primary_language only.

    Precedence:
      1. call context astra_primary_language / primary_language (from employee UI)
      2. workflow_configurations.primary_language
    Never derive from TTS static language, org defaults, or provider defaults.
    """
    ctx = call_context_vars or {}
    cfg = workflow_configurations or {}
    candidates = [
        ctx.get("astra_primary_language"),
        ctx.get("primary_language"),
        cfg.get("primary_language"),
    ]
    voice = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else None
    if voice:
        candidates.extend([voice.get("primary_language"), voice.get("language")])
    for raw in candidates:
        if raw:
            mapped = map_to_pipecat_language(raw)
            if mapped is not None:
                return mapped
    return None


def _build_session_language_state(
    tts=None,
    *,
    primary_language=None,
    workflow_configurations=None,
    call_context_vars=None,
) -> SessionLanguageState:
    """Primary from employee.primary_language; allow te/en/hi (+ primary)."""
    primary = primary_language or _resolve_employee_primary_language(
        workflow_configurations=workflow_configurations,
        call_context_vars=call_context_vars,
    )
    # Intentionally do NOT fall back to TTS static language.
    cfg = workflow_configurations or {}
    allowed = set(_DEFAULT_ALLOWED)
    # Honor explicit workflow allowed_languages when configured (generic).
    raw_allowed = cfg.get("allowed_languages")
    if isinstance(raw_allowed, (list, tuple)) and raw_allowed:
        explicit = set()
        for raw in raw_allowed:
            mapped = map_to_pipecat_language(raw)
            if mapped is not None:
                explicit.add(mapped.value if hasattr(mapped, "value") else str(mapped))
        if explicit:
            allowed = explicit
    tag = None
    if primary is not None:
        tag = primary.value if hasattr(primary, "value") else str(primary)
        allowed.add(tag)
    # Language lock: auto_language_switch=false or language_mode=locked.
    auto_switch = cfg.get("auto_language_switch")
    locked = (
        auto_switch is False
        or str(auto_switch).strip().lower() in ("false", "0", "no", "off")
        or str(cfg.get("language_mode") or "").strip().lower() == "locked"
    ) and tag is not None
    if locked:
        allowed = {tag}
    _policy = cfg.get("astra_language_policy")
    follow = bool(isinstance(_policy, dict) and _policy.get("follow_latest_caller")) and not locked
    return SessionLanguageState(
        primary_language=primary,
        allowed_languages=allowed,
        language_locked=locked,
        follow_latest_caller=follow,
    )


def build_pipeline(
    transport,
    stt,
    audio_buffer,
    llm,
    tts,
    user_context_aggregator,
    assistant_context_aggregator,
    pipeline_engine_callback_processor,
    pipeline_metrics_aggregator,
    voicemail_detector=None,
    recording_router=None,
    *,
    primary_language=None,
    workflow_configurations=None,
    call_context_vars=None,
    pre_llm_processors=None,
    post_llm_processors=None,
    pre_tts_processors=None,
):
    """Build the main pipeline with all components.

    Args:
        audio_buffer: AudioBufferProcessor that handles both input and output audio recording.
        voicemail_detector: Optional native pipecat VoicemailDetector. When provided,
            inserts voicemail detection after STT. Note: We don't use the TTS gate
            to avoid blocking TTS frames during classification.
        recording_router: Optional RecordingRouterProcessor. When provided,
            inserts between callback processor and TTS to route between
            pre-recorded audio playback and dynamic TTS.
    """
    session = _build_session_language_state(
        tts,
        primary_language=primary_language,
        workflow_configurations=workflow_configurations,
        call_context_vars=call_context_vars,
    )
    _primary_tag = (
        session.primary_language.value
        if session.primary_language is not None
        and hasattr(session.primary_language, "value")
        else session.primary_language
    )
    _current = getattr(session, "current_language", None)
    _current_tag = (
        _current.value
        if _current is not None and hasattr(_current, "value")
        else _current
    )
    logger.info(
        "multilingual session: primary={!r} current={!r} allowed={} locked={}",
        _primary_tag,
        _current_tag if _current_tag is not None else _primary_tag,
        sorted(session.allowed_languages),
        getattr(session, "language_locked", False),
    )

    # Build processors list with optional voicemail detection
    processors = [
        transport.input(),  # Transport user input
        stt,
    ]
    # Promote short-utterance interims only for STT services that may never
    # finalize (e.g. Deepgram). Services that guarantee a final per utterance
    # (e.g. Sarvam realtime) opt out; promoting their interims would duplicate
    # the caller's words in the LLM context.
    if interim_finalizer_needed(stt):
        processors.append(InterimTranscriptFinalizer())
    processors += [
        # Platform-wide: STT detected language → Sarvam TTS language_code
        # (employee.primary_language remains greeting/default only).
        MultilingualTtsLanguageProcessor(source="stt", session=session),
    ]

    # Insert voicemail detector after STT if enabled
    # Note: We intentionally do NOT use voicemail_detector.gate() to allow TTS
    # frames to continue flowing during classification (non-blocking detection)

    # Note: We must keep user_context_aggregator after voicemail_detector
    # or else, LLMContextFrames generated from user_context_aggregator will
    # start generating LLM Completion from Voicemail Classifier
    if voicemail_detector:
        logger.info("Adding native voicemail detector to pipeline")
        processors.append(voicemail_detector.detector())

    # Continue with the rest of the pipeline
    post_llm = list(post_llm_processors or [])
    # Per-workflow spoken-term replacements (config-driven; absent -> not inserted).
    try:
        from api.services.pipecat.spoken_term_guard import build_spoken_term_guard

        _stg = build_spoken_term_guard(workflow_configurations)
        if _stg is not None:
            post_llm.append(_stg)
    except Exception as e:
        logger.warning(f"SpokenTermGuard not inserted: {e}")
    post_llm += [
        pipeline_engine_callback_processor,
        # Platform-wide: reply script/language → Sarvam TTS language_code;
        # speaker stays locked. Must sit before TTS.
        MultilingualTtsLanguageProcessor(source="reply", session=session),
    ]
    if recording_router:
        post_llm.append(recording_router)
    # Optional pass-through observers right before TTS (e.g. Astra barge-in guard
    # bot-text tap). None -> pipeline unchanged.
    post_llm.extend(list(pre_tts_processors or []))

    processors.append(user_context_aggregator)

    # Insert LLM gate before the main LLM when voicemail detection is enabled.
    # This prevents the main LLM from being triggered until classification
    # determines whether a human or voicemail answered the call.
    if voicemail_detector:
        processors.append(voicemail_detector.llm_gate())

    # Optional processors that observe each LLM request (e.g. caller facts).
    processors.extend(list(pre_llm_processors or []))

    processors.extend(
        [
            llm,  # LLM
            *post_llm,
            tts,  # TTS
            transport.output(),  # Transport bot output
            audio_buffer,  # AudioBufferProcessor - records both input and output audio
            assistant_context_aggregator,  # Assistant spoken responses
            pipeline_metrics_aggregator,
        ]
    )

    return Pipeline(processors)


def build_realtime_pipeline(
    transport,
    realtime_llm,
    audio_buffer,
    user_context_aggregator,
    assistant_context_aggregator,
    pipeline_engine_callback_processor,
    pipeline_metrics_aggregator,
    voicemail_detector=None,
):
    """Build a pipeline for realtime (speech-to-speech) LLM services.

    Realtime services (e.g. OpenAI Realtime, Gemini Live) handle STT+LLM+TTS
    internally, so no separate STT or TTS processors are needed.

    Args:
        voicemail_detector: Optional VoicemailDetector. Placed *below* the
            realtime LLM. This is asymmetric with the non-realtime layout
            (where the detector sits between STT and the main user aggregator)
            because the realtime LLM is both the source of TranscriptionFrame
            (broadcast downstream) and the sink of LLMContextFrame (consumed
            by _handle_context without forwarding). Placing the detector below
            the realtime LLM means: downstream TranscriptionFrames reach the
            classifier branch, UserStartedSpeakingFrame /
            UserStoppedSpeakingFrame are forwarded through by the LLM, and the
            main aggregator's LLMContextFrame is absorbed by the realtime LLM
            and never leaks into the classifier (which would otherwise run a
            voicemail completion on the workflow's main context).

            The TTS gate and LLM gate are intentionally not used: the realtime
            LLM reacts to audio directly, not to LLMContextFrames. On voicemail
            detection we drop the call via end_call_with_reason; the detector's
            ConversationGate also blocks downstream audio output until the call
            ends.
    """
    processors = [
        transport.input(),
        user_context_aggregator,
        realtime_llm,
    ]

    if voicemail_detector:
        logger.info("Adding native voicemail detector to realtime pipeline")
        processors.append(voicemail_detector.detector())

    processors.extend(
        [
            pipeline_engine_callback_processor,
            transport.output(),
            audio_buffer,
            assistant_context_aggregator,
            pipeline_metrics_aggregator,
        ]
    )

    return Pipeline(processors)


def create_pipeline_task(
    pipeline,
    workflow_run_id,
    audio_config: AudioConfig = None,
    *,
    conversation_parent_context=None,
    conversation_type: str = "voice",
    additional_span_attributes: dict | None = None,
):
    """Create a pipeline task with appropriate parameters.

    Args:
        pipeline: The pipeline to run.
        workflow_run_id: Run id, used as the conversation id.
        audio_config: Optional audio configuration.
        conversation_parent_context: Optional OTEL context carrying a fixed
            trace id. When provided, the conversation span attaches to that
            trace instead of starting a new root trace (used by text chat to
            stitch every per-turn pipeline into one trace).
        conversation_type: ``conversation.type`` span attribute value.
        additional_span_attributes: Extra attributes set on the conversation
            span (e.g. ``langfuse.trace.name`` to name a stitched trace that
            has no real root span).
    """
    # Set up pipeline params with audio configuration if provided
    pipeline_params = PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
        send_initial_empty_metrics=False,
        enable_heartbeats=True,
        start_metadata={"workflow_run_id": workflow_run_id},
    )

    # If audio_config is provided, set the audio sample rates
    if audio_config:
        pipeline_params.audio_in_sample_rate = audio_config.transport_in_sample_rate
        pipeline_params.audio_out_sample_rate = audio_config.transport_out_sample_rate
        logger.debug(
            f"Setting pipeline audio params - in: {audio_config.transport_in_sample_rate}Hz, "
            f"out: {audio_config.transport_out_sample_rate}Hz"
        )

    task = PipelineWorker(
        pipeline,
        params=pipeline_params,
        enable_tracing=True,
        enable_rtvi=False,
        conversation_id=f"{workflow_run_id}",
        conversation_parent_context=conversation_parent_context,
        conversation_type=conversation_type,
        additional_span_attributes=additional_span_attributes,
    )

    # Check if turn logging is enabled
    enable_turn_logging = os.getenv("ENABLE_TURN_LOGGING", "false").lower() == "true"

    if enable_turn_logging:
        # Attach event handlers to propagate turn information into the logging context
        turn_observer = task.turn_tracking_observer

        if turn_observer is not None:
            # Import turn context manager only if needed
            from api.services.pipecat.turn_context import get_turn_context_manager

            async def _on_turn_started(observer, turn_number: int):
                """Set the current turn number into the context variable."""
                # Set in both contextvar and turn context manager
                turn_var.set(turn_number)
                turn_manager = get_turn_context_manager()
                turn_manager.set_turn(turn_number)

            # Register the handlers with the observer
            turn_observer.add_event_handler("on_turn_started", _on_turn_started)

    return task


def interim_finalizer_needed(stt) -> bool:
    """False when the STT service declares it always emits a final transcript."""
    return not bool(getattr(stt, "guarantees_final_transcripts", False))
