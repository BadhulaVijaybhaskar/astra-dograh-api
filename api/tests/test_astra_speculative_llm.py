"""Start-LLM-on-partial-transcript (astra_speculative_llm): adopt when the final text
matches, cancel + run the real turn when it differs, never speak/act on an unadopted
speculation. Fake LLM, no network; frames run through a real Pipecat pipeline."""

import asyncio

import pytest
from pipecat.frames.frames import (
    ControlFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from api.services.pipecat.astra_speculative_llm import (
    CONFIG_KEY,
    SpeculativeReplyController,
    build_speculative_controller,
    normalize_text,
    speculative_config,
)
from api.services.pipecat.pipeline_builder import build_pipeline
from api.services.pipecat.worker_runner import (
    run_pipeline_worker,
    wait_for_pipeline_worker_started,
)


class TurnEnd(ControlFrame):
    """Test stimulus: the caller's turn ended (what the turn analyzer decides)."""


class FakeUserAggregator(FrameProcessor):
    """Mimics LLMUserAggregator: eats transcripts, on turn end appends the user message
    to the shared context and pushes the real LLMContextFrame."""

    def __init__(self, context):
        super().__init__(name="FakeUserAggregator")
        self.context = context
        self._parts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._parts.append(frame.text)
        elif isinstance(frame, InterimTranscriptionFrame):
            pass
        elif isinstance(frame, TurnEnd):
            text = " ".join(self._parts).strip()
            self._parts = []
            self.context.add_message({"role": "user", "content": text})
            await self.push_frame(LLMContextFrame(context=self.context))
        else:
            await self.push_frame(frame, direction)


class FakeLLM(FrameProcessor):
    """Streams "re: <user text>" word by word. 'book' in the text -> a tool call."""

    def __init__(self, delay=0.02):
        super().__init__(name="FakeLLM")
        self.delay = delay
        self.requests = []  # user texts it started on
        self.completed = []
        self.cancelled = []
        self.executed_tools = []

    async def run_function_calls(self, function_calls):
        self.executed_tools.extend(fc.function_name for fc in function_calls)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            text = frame.context.messages[-1]["content"]
            self.requests.append(text)
            await self.push_frame(LLMFullResponseStartFrame())
            try:
                for w in ("re:", *text.split()):
                    await asyncio.sleep(self.delay)
                    await self.push_frame(LLMTextFrame(w + " "))
                if "book" in text:
                    await self.run_function_calls(
                        [FunctionCallFromLLM(function_name="book_demo", tool_call_id="t1", arguments={}, context=frame.context)]
                    )
                self.completed.append(text)
            except asyncio.CancelledError:
                self.cancelled.append(text)
                raise
            finally:
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__(name="Sink")
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)

    def spoken(self):
        return "".join(f.text for f in self.frames if isinstance(f, LLMTextFrame)).strip()

    def count(self, cls):
        return sum(1 for f in self.frames if isinstance(f, cls))


def make(cfg=None, delay=0.02):
    ctx = LLMContext(messages=[{"role": "system", "content": "You are Sanjana."}])
    agg = FakeUserAggregator(ctx)
    llm = FakeLLM(delay=delay)
    c = SpeculativeReplyController(
        speculative_config({CONFIG_KEY: {"enabled": True, "stable_ms": 30, **(cfg or {})}}),
        context_provider=lambda: agg.context,
        llm=llm,
    )
    sink = Sink()
    pipeline = Pipeline([c.tap, agg, c.injector, llm, c.gate, sink])
    return c, agg, llm, sink, pipeline


async def run(pipeline, script):
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)
    task = asyncio.create_task(run_pipeline_worker(worker))
    await wait_for_pipeline_worker_started(worker, run_task=task)
    for step in script:
        if isinstance(step, (int, float)):
            await asyncio.sleep(step)
        elif callable(step):
            step()
        else:
            await worker.queue_frame(step)
    await asyncio.sleep(0.05)
    await worker.queue_frame(EndFrame())
    await asyncio.wait_for(task, timeout=5)


def final(text):
    return TranscriptionFrame(text=text, user_id="u", timestamp="t")


def interim(text):
    return InterimTranscriptionFrame(text=text, user_id="u", timestamp="t")


# ------------------------------------------------------------------ pure helpers
def test_config_is_opt_in_and_bounded():
    assert speculative_config(None).enabled is False
    assert speculative_config({}).enabled is False
    assert speculative_config({CONFIG_KEY: {"enabled": False}}).enabled is False
    assert speculative_config({CONFIG_KEY: True}).enabled is True
    c = speculative_config({CONFIG_KEY: {"enabled": True, "stable_ms": 99999, "max_attempts": 2}})
    assert (c.enabled, c.stable_ms, c.max_attempts) == (True, 150, 2)
    assert build_speculative_controller({}, user_context_aggregator=None, llm=None) is None
    on = {CONFIG_KEY: {"enabled": True}}
    assert build_speculative_controller(on, user_context_aggregator=None, llm=None, is_realtime=True) is None
    assert build_speculative_controller(on, user_context_aggregator=None, llm=None, voicemail_enabled=True) is None
    assert build_speculative_controller(on, user_context_aggregator=None, llm=None) is not None


def test_normalize_text_ignores_case_punctuation_spacing_keeps_telugu():
    assert normalize_text("Hello,  how are you?") == normalize_text("hello how are you")
    assert normalize_text("నమస్కారం, మీరు ఎలా ఉన్నారు?") == "నమస్కారం మీరు ఎలా ఉన్నారు"
    assert normalize_text("demo book") != normalize_text("demo booked")
    assert normalize_text([{"type": "text", "text": "Hi!"}]) == "hi"


# ------------------------------------------------------------------ pipeline behaviour
@pytest.mark.asyncio
async def test_final_matches_partial_reply_is_adopted_and_not_repeated():
    c, agg, llm, sink, p = make()
    # caller says it, speculation starts after 30 ms stable; turn ends 120 ms later
    await run(p, [interim("what is"), 0.01, final("What is the price?"), 0.15, TurnEnd(), 0.3])
    assert c.stats["adopted"] == 1 and c.stats["cancelled"] == 0
    assert llm.requests == ["What is the price?"], "LLM ran once, on the partial; the real request was dropped"
    assert sink.spoken() == "re: What is the price?"
    assert sink.count(LLMFullResponseStartFrame) == 1 and sink.count(LLMFullResponseEndFrame) == 1
    # the shared context got the user turn exactly once
    assert [m["role"] for m in agg.context.messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_adopted_while_still_streaming_flushes_and_goes_live():
    c, agg, llm, sink, p = make(delay=0.05)
    await run(p, [final("tell me about your plans please"), 0.06, TurnEnd(), 0.6])
    assert c.stats["adopted"] == 1
    assert llm.requests == ["tell me about your plans please"]
    assert sink.spoken() == "re: tell me about your plans please"


@pytest.mark.asyncio
async def test_final_differs_speculation_cancelled_real_reply_runs():
    c, agg, llm, sink, p = make(delay=0.05)
    # partial "I want" speculated; caller keeps talking -> final is longer
    await run(p, [final("I want"), 0.06, final("a callback tomorrow"), 0.0, TurnEnd(), 0.8])
    assert c.stats["adopted"] == 0
    assert llm.requests[0] == "I want"
    assert llm.requests[-1] == "I want a callback tomorrow"
    assert sink.spoken() == "re: I want a callback tomorrow", "nothing from the stale speculation is spoken"
    assert sink.count(LLMFullResponseStartFrame) == 1 and sink.count(LLMFullResponseEndFrame) == 1
    assert not any(isinstance(f, InterruptionFrame) for f in sink.frames), "the cancel never reaches TTS/transport"


@pytest.mark.asyncio
async def test_text_change_mid_turn_restarts_speculation_then_adopts_latest():
    c, agg, llm, sink, p = make(delay=0.03)
    await run(p, [final("can you"), 0.05, final("call me"), 0.2, TurnEnd(), 0.4])
    assert c.stats["started"] == 2 and c.stats["adopted"] == 1 and c.stats["cancelled"] == 1
    assert llm.requests == ["can you", "can you call me"]
    assert sink.spoken() == "re: can you call me"


@pytest.mark.asyncio
async def test_speculation_never_runs_a_tool_real_turn_does():
    c, agg, llm, sink, p = make(delay=0.01)
    await run(p, [final("please book a demo"), 0.15, TurnEnd(), 0.4])
    assert c.stats["tool_fallbacks"] == 1 and c.stats["adopted"] == 0
    assert llm.executed_tools == ["book_demo"], "executed once, by the real turn only"
    assert llm.requests == ["please book a demo", "please book a demo"]
    assert sink.spoken() == "re: please book a demo"


@pytest.mark.asyncio
async def test_no_speculation_while_bot_speaking_and_below_min_chars():
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame

    c, agg, llm, sink, p = make({"min_chars": 3})
    await run(p, [final("ok"), 0.08, TurnEnd(), 0.15])
    assert c.stats["started"] == 0 and llm.requests == ["ok"] and sink.spoken() == "re: ok"
    c2, agg2, llm2, sink2, p2 = make()
    await run(p2, [BotStartedSpeakingFrame(), final("hello there"), 0.08, BotStoppedSpeakingFrame(), TurnEnd(), 0.2])
    assert c2.stats["started"] == 0 and llm2.requests == ["hello there"]


@pytest.mark.asyncio
async def test_context_moved_between_partial_and_final_is_a_mismatch():
    c, agg, llm, sink, p = make()

    def node_switch():  # engine replaced the system prompt after the speculation started
        agg.context.messages[0] = {"role": "system", "content": "You are Sanjana (node 2)."}

    await run(p, [final("what time is it"), 0.15, node_switch, TurnEnd(), 0.3])
    assert c.stats["adopted"] == 0
    assert llm.requests == ["what time is it", "what time is it"]
    assert sink.spoken() == "re: what time is it"


@pytest.mark.asyncio
async def test_real_barge_in_kills_unadopted_speculation():
    c, agg, llm, sink, p = make(delay=0.05)
    await run(p, [final("hmm let me think"), 0.08, InterruptionFrame(), 0.05, final("no wait"), 0.0, TurnEnd(), 0.8])
    assert c.stats["adopted"] == 0
    assert "re: hmm let me think" != sink.spoken()
    assert sink.spoken().endswith("hmm let me think no wait")


def test_build_pipeline_inserts_tap_injector_gate_only_when_enabled():
    class P(FrameProcessor):
        def __init__(self, n):
            super().__init__(name=n)

    class T:
        def input(self):
            return P("in")

        def output(self):
            return P("out")

    def names(spec):
        pipe = build_pipeline(
            T(), P("stt"), P("buf"), P("llm"), P("tts"), P("uagg"), P("aagg"), P("cb"), P("metrics"),
            speculative_llm=spec,
        )
        return [x.name.split("#")[0] for x in pipe.processors if not x.name.startswith("Pipeline")]

    base = names(None)
    assert not any(n.startswith("Speculative") for n in base)
    ctl = SpeculativeReplyController(speculative_config({CONFIG_KEY: True}), context_provider=lambda: None)
    n = names(ctl)
    assert n.index("SpeculativeTranscriptTap") + 1 == n.index("uagg")
    assert n.index("SpeculativeInjector") + 1 == n.index("llm")
    assert n.index("llm") + 1 == n.index("SpeculativeGate")
    assert [x for x in n if not x.startswith("Speculative")] == base
