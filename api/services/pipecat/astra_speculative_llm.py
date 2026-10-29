"""Start the LLM reply on the caller's partial transcript (Astra, 2026-10-29).

While the caller is still finishing a turn, the turn-end wait (STT silence + turn
analyzer) costs several hundred ms before the user aggregator pushes the real
``LLMContextFrame``. This module starts the LLM early on the text heard so far and
holds the reply back until the turn really ends:

* final text matches the speculated text (and nothing else in the context moved)
  -> the held reply is released at once and the real request is dropped;
* final text differs, or the speculated reply wanted a tool
  -> the speculation is cancelled in the LLM and the real request runs as usual.

Nothing from a speculation is ever spoken, added to the context or allowed to run
a function unless it is adopted. Opt-in per workflow with
``workflow_configurations["astra_speculative_llm"] = {"enabled": true}``; absent or
disabled -> the pipeline is byte-for-byte the old one.

Placement (see ``pipeline_builder.build_pipeline``)::

    ... -> SpeculativeTranscriptTap -> user_context_aggregator -> [pre-LLM]
        -> SpeculativeInjector -> llm -> SpeculativeGate -> [post-LLM] -> tts ...

Ordering inside the LLM is guaranteed by in-band markers: ``SpecBeginFrame`` /
``SpecEndFrame`` bracket the speculative request and ``SpecEndFrame`` /
``SpecReleaseFrame`` are uninterruptible, so they survive the LLM queue flush that
cancels a speculation. The cancel itself is ``SpecCancelFrame`` (an
``InterruptionFrame`` subclass, so the LLM aborts the in-flight completion) and the
gate swallows it, so TTS / transport never see it.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ControlFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    SystemFrame,
    TranscriptionFrame,
    UninterruptibleFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

CONFIG_KEY = "astra_speculative_llm"


@dataclass
class SpeculativeConfig:
    enabled: bool = False
    # Text must be unchanged this long before a speculation starts (caps LLM spend
    # while words are still streaming in).
    stable_ms: int = 150
    # Ignore tiny fragments ("ha", "ok") as a speculation trigger.
    min_chars: int = 2
    # Speculations per caller turn (each restart after a text change counts).
    max_attempts: int = 3


def speculative_config(workflow_configurations: Optional[dict]) -> SpeculativeConfig:
    raw = (workflow_configurations or {}).get(CONFIG_KEY)
    if raw is True:
        return SpeculativeConfig(enabled=True)
    if not isinstance(raw, dict):
        return SpeculativeConfig()
    cfg = SpeculativeConfig(enabled=bool(raw.get("enabled", False)))
    for name, lo, hi in (("stable_ms", 0, 2000), ("min_chars", 1, 50), ("max_attempts", 1, 10)):
        v = raw.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
            setattr(cfg, name, int(v))
    return cfg


def normalize_text(text: Any) -> str:
    """Compare-form of a transcript: NFKC, casefold, punctuation/symbols dropped, spaces collapsed."""
    if isinstance(text, list):  # content parts
        text = " ".join(
            p.get("text", "") for p in text if isinstance(p, dict) and p.get("type") == "text"
        )
    if not isinstance(text, str):
        return ""
    out = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        elif cat[0] == "Z" or ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


# ----------------------------------------------------------------------------- frames
@dataclass
class SpecBeginFrame(ControlFrame):
    spec_id: int = 0


@dataclass
class SpecEndFrame(ControlFrame, UninterruptibleFrame):
    spec_id: int = 0


@dataclass
class SpecReleaseFrame(ControlFrame, UninterruptibleFrame):
    spec_id: int = 0


@dataclass
class SpecDiscardFrame(ControlFrame, UninterruptibleFrame):
    spec_id: int = 0


@dataclass
class SpecCancelFrame(InterruptionFrame):
    """Aborts the in-flight speculative completion in the LLM; swallowed by the gate."""

    spec_id: int = 0


# ----------------------------------------------------------------------------- state
@dataclass
class Speculation:
    spec_id: int
    text: str
    norm: str
    snapshot: list
    context: LLMContext
    started: float = field(default_factory=time.monotonic)
    state: str = "running"  # running | done | adopted | cancelled
    needs_tools: bool = False


def _message_text(msg: Any) -> Any:
    return msg.get("content") if isinstance(msg, dict) else None


class SpeculativeReplyController:
    """Shared state for the tap, injector and gate of one call."""

    def __init__(
        self,
        config: SpeculativeConfig,
        *,
        context_provider: Callable[[], Optional[LLMContext]],
        llm: Any = None,
    ):
        self.config = config
        self._context_provider = context_provider
        self.tap = SpeculativeTranscriptTap(self)
        self.injector = SpeculativeInjector(self)
        self.gate = SpeculativeGate(self)
        self.spec: Optional[Speculation] = None
        self.adopted_id: Optional[int] = None
        self.cancelled_ids: set[int] = set()
        self.stats = {"started": 0, "adopted": 0, "cancelled": 0, "tool_fallbacks": 0}
        self._next_id = 0
        self._finals: list[TextPartForConcatenation] = []
        self._interim = ""
        self._attempts = 0
        self._bot_speaking = False
        self._timer = None
        if llm is not None:
            self._guard_function_calls(llm)

    # ---- function-call guard: a speculation never executes a tool
    def _guard_function_calls(self, llm: Any) -> None:
        orig = getattr(llm, "run_function_calls", None)
        if orig is None:
            return
        controller = self

        async def guarded_run_function_calls(function_calls):
            spec = controller.spec
            if spec is not None and any(
                getattr(fc, "context", None) is spec.context for fc in (function_calls or [])
            ):
                spec.needs_tools = True
                controller.stats["tool_fallbacks"] += 1
                logger.info(
                    "speculative LLM #{}: reply wants a tool -> not executed, real turn will run it",
                    spec.spec_id,
                )
                return None
            return await orig(function_calls)

        llm.run_function_calls = guarded_run_function_calls

    # ---- turn text (fed by the tap)
    def candidate_text(self) -> str:
        parts = list(self._finals)
        if self._interim.strip():
            parts.append(TextPartForConcatenation(self._interim, includes_inter_part_spaces=False))
        return concatenate_aggregated_text(parts).strip() if parts else ""

    def on_final(self, frame: TranscriptionFrame) -> None:
        if not (frame.text or "").strip():
            return
        self._finals.append(
            TextPartForConcatenation(
                frame.text,
                includes_inter_part_spaces=getattr(frame, "includes_inter_frame_spaces", False),
            )
        )
        self._interim = ""
        self._schedule()

    def on_interim(self, frame: InterimTranscriptionFrame) -> None:
        self._interim = frame.text or ""
        self._schedule()

    def on_bot_speaking(self, speaking: bool) -> None:
        self._bot_speaking = speaking
        if speaking:
            self._cancel_timer()

    def reset_turn(self) -> None:
        self._cancel_timer()
        self._finals = []
        self._interim = ""
        self._attempts = 0

    def _schedule(self) -> None:
        text = self.candidate_text()
        norm = normalize_text(text)
        if len(norm) < self.config.min_chars or self._bot_speaking:
            return
        spec = self.spec
        if spec is not None and spec.state in ("running", "done") and spec.norm == norm:
            return
        if self._attempts >= self.config.max_attempts:
            return
        self._cancel_timer()
        self._timer = self.tap.create_task(self._fire(text), name="astra-spec-llm-timer")

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None and not timer.done():
            timer.cancel()

    async def _fire(self, text: str) -> None:
        import asyncio

        await asyncio.sleep(self.config.stable_ms / 1000.0)
        self._timer = None
        if self.candidate_text() != text:
            return
        await self.start(text)

    # ---- speculation lifecycle (pushes go through the injector, right before the LLM)
    async def start(self, text: str) -> Optional[Speculation]:
        if self._bot_speaking or self._attempts >= self.config.max_attempts:
            return None
        ctx = self._context_provider()
        if ctx is None:
            return None
        await self.abandon("text changed")
        snapshot = list(ctx.messages)
        spec_ctx = LLMContext(
            messages=[dict(m) if isinstance(m, dict) else m for m in snapshot]
            + [{"role": "user", "content": text}],
            tools=ctx.tools,
            tool_choice=ctx.tool_choice,
        )
        self._next_id += 1
        spec = Speculation(self._next_id, text, normalize_text(text), snapshot, spec_ctx)
        self.spec = spec
        self._attempts += 1
        self.stats["started"] += 1
        logger.debug("speculative LLM #{}: start on partial ({} chars)", spec.spec_id, len(text))
        await self.injector.push_frame(SpecBeginFrame(spec_id=spec.spec_id))
        await self.injector.push_frame(LLMContextFrame(context=spec_ctx))
        await self.injector.push_frame(SpecEndFrame(spec_id=spec.spec_id))
        return spec

    async def abandon(self, why: str) -> None:
        spec = self.spec
        if spec is None or spec.state not in ("running", "done"):
            return
        was_running = spec.state == "running"
        spec.state = "cancelled"
        self.cancelled_ids.add(spec.spec_id)
        self.stats["cancelled"] += 1
        logger.debug("speculative LLM #{}: cancelled ({})", spec.spec_id, why)
        if was_running:
            await self.injector.push_frame(SpecCancelFrame(spec_id=spec.spec_id))
        else:
            await self.injector.push_frame(SpecDiscardFrame(spec_id=spec.spec_id))

    def matches(self, spec: Speculation, context: LLMContext) -> bool:
        msgs = list(context.messages)
        if len(msgs) != len(spec.snapshot) + 1:
            return False
        last = msgs[-1]
        if not isinstance(last, dict) or last.get("role") != "user":
            return False
        if normalize_text(_message_text(last)) != spec.norm:
            return False
        return all(a is b or a == b for a, b in zip(msgs[:-1], spec.snapshot))

    async def on_real_context(self, frame: LLMContextFrame) -> bool:
        """Real user-turn request reached the LLM's door. True -> drop it (speculation adopted)."""
        spec = self.spec
        self.reset_turn()
        if spec is None or spec.state not in ("running", "done"):
            return False
        if not spec.needs_tools and self.matches(spec, frame.context):
            spec.state = "adopted"
            self.adopted_id = spec.spec_id
            self.stats["adopted"] += 1
            logger.info(
                "speculative LLM #{}: adopted, reply started {:.0f} ms before the turn ended",
                spec.spec_id,
                (time.monotonic() - spec.started) * 1000,
            )
            # The speculation still owns the LLM's tools/handlers context; the reply goes to
            # the real shared context through the assistant aggregator downstream.
            await self.injector.push_frame(SpecReleaseFrame(spec_id=spec.spec_id))
            return True
        await self.abandon("final text differs" if not spec.needs_tools else "needs tools")
        return False

    def on_interruption(self) -> None:
        """A real interruption flows to the LLM: it also aborts any running speculation."""
        spec = self.spec
        if spec is not None and spec.state in ("running", "done"):
            spec.state = "cancelled"
            self.cancelled_ids.add(spec.spec_id)
            self.stats["cancelled"] += 1


# ----------------------------------------------------------------------------- processors
class SpeculativeTranscriptTap(FrameProcessor):
    """Before the user aggregator: watches what the caller has said so far. Pass-through."""

    def __init__(self, controller: SpeculativeReplyController, **kwargs):
        super().__init__(name="SpeculativeTranscriptTap", **kwargs)
        self._c = controller

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._c.on_final(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            self._c.on_interim(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._c.on_bot_speaking(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._c.on_bot_speaking(False)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._c.reset_turn()
        await self.push_frame(frame, direction)


class SpeculativeInjector(FrameProcessor):
    """Right before the LLM: injects speculative requests, decides adopt vs. restart."""

    def __init__(self, controller: SpeculativeReplyController, **kwargs):
        super().__init__(name="SpeculativeInjector", **kwargs)
        self._c = controller

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMContextFrame):
                spec = self._c.spec
                if spec is None or frame.context is not spec.context:
                    if await self._c.on_real_context(frame):
                        return
            elif isinstance(frame, InterruptionFrame) and not isinstance(frame, SpecCancelFrame):
                self._c.on_interruption()
        await self.push_frame(frame, direction)


class SpeculativeGate(FrameProcessor):
    """Right after the LLM: holds a speculative reply until it is adopted, drops it otherwise."""

    def __init__(self, controller: SpeculativeReplyController, **kwargs):
        super().__init__(name="SpeculativeGate", **kwargs)
        self._c = controller
        self._cur: Optional[int] = None
        self._mode = "idle"  # idle | capture | held | live | discard
        self._buf: list = []

    @property
    def mode(self) -> str:
        return self._mode

    async def _flush(self) -> None:
        buf, self._buf = self._buf, []
        for f, d in buf:
            await self.push_frame(f, d)

    def _drop(self) -> None:
        self._buf = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, SpecCancelFrame):
            # Not passed to super(): no queue flush here (anything still queued from the
            # speculation is dropped below by mode="discard"). Never forwarded.
            if frame.spec_id == self._cur:
                self._drop()
                self._mode = "discard" if self._mode == "capture" else "idle"
            return
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, SpecBeginFrame):
            self._cur, self._buf = frame.spec_id, []
            self._mode = "live" if self._c.adopted_id == frame.spec_id else "capture"
            return
        if isinstance(frame, SpecEndFrame):
            if frame.spec_id == self._cur:
                spec = self._c.spec
                if spec is not None and spec.spec_id == frame.spec_id and spec.state == "running":
                    spec.state = "done"
                if frame.spec_id in self._c.cancelled_ids:
                    self._drop()
                    self._mode = "idle"
                elif self._mode == "capture" and self._c.adopted_id == frame.spec_id:
                    await self._flush()
                    self._mode = "idle"
                elif self._mode == "capture":
                    self._mode = "held"
                else:
                    self._drop()
                    self._mode = "idle"
            return
        if isinstance(frame, SpecReleaseFrame):
            if frame.spec_id == self._cur:
                if self._mode in ("capture", "held"):
                    await self._flush()
                self._mode = "live" if self._mode == "capture" else "idle"
            return
        if isinstance(frame, SpecDiscardFrame):
            if frame.spec_id == self._cur:
                self._drop()
                self._mode = "discard" if self._mode == "capture" else "idle"
            return
        if isinstance(frame, InterruptionFrame):
            # Real barge-in: an unadopted speculation is dead (the LLM aborted it too).
            if self._mode in ("capture", "held"):
                self._drop()
                self._mode = "discard" if self._mode == "capture" else "idle"
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if self._mode == "capture" and self._cur in self._c.cancelled_ids:
            self._drop()
            self._mode = "discard"
        if self._mode == "capture":
            if self._c.adopted_id == self._cur:
                await self._flush()
                self._mode = "live"
                await self.push_frame(frame, direction)
            else:
                self._buf.append((frame, direction))
            return
        if self._mode == "discard":
            return
        # idle / live / held (anything after SpecEnd is not the speculation's) -> pass
        await self.push_frame(frame, direction)


def build_speculative_controller(
    workflow_configurations: Optional[dict],
    *,
    user_context_aggregator: Any,
    llm: Any,
    is_realtime: bool = False,
    voicemail_enabled: bool = False,
) -> Optional[SpeculativeReplyController]:
    """None unless the workflow opts in (and the pipeline is a plain STT->LLM->TTS cascade)."""
    cfg = speculative_config(workflow_configurations)
    if not cfg.enabled:
        return None
    if is_realtime or voicemail_enabled:
        logger.info("speculative LLM requested but skipped (realtime={} voicemail={})", is_realtime, voicemail_enabled)
        return None
    agg = user_context_aggregator
    return SpeculativeReplyController(
        cfg, context_provider=lambda: getattr(agg, "context", None), llm=llm
    )
