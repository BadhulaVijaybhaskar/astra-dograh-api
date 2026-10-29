"""One caller turn -> one assistant response across node transitions (run 238 bug)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor

from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.transition_response_guard import (
    TransitionResponseGuard,
    is_complete_response,
)

RUN238_TURN4 = (
    "అర్థమందండి ఈ voice setup లో WhatsApp message integration ప్రస్తుతం "
    "configured లేదు మీకు ఇంకేమైనా help కావాలా"
)


# ---------------------------------------------------------------- guard unit
def test_complete_response_vs_filler():
    assert not is_complete_response("")
    assert not is_complete_response(" . ")
    assert not is_complete_response("సరే")
    assert not is_complete_response("సరేనండి, ఒక్క నిమిషం")
    assert not is_complete_response("Sure, one moment.")
    assert is_complete_response("మీ పేరు చెప్తారా?")
    assert is_complete_response(RUN238_TURN4)
    assert is_complete_response("Our voice agent answers calls and books demos for you.")


def _feed(guard, *texts, start=True):
    if start:
        guard.observe_llm_frame(LLMFullResponseStartFrame())
    for t in texts:
        guard.observe_llm_frame(LLMTextFrame(t))
    guard.observe_llm_frame(LLMFullResponseEndFrame())


def test_guard_suppresses_after_spoken_answer_keyed_on_session_and_generation():
    g = TransitionResponseGuard(session_id=238, enabled=True)
    _feed(g, RUN238_TURN4)
    assert g.allow_follow_up("understand_enquiry", "enquiry") is False
    assert g.suppressed == [(238, 1)]


def test_guard_allows_tool_only_generation():
    g = TransitionResponseGuard(session_id=1, enabled=True)
    _feed(g)  # tool call only, no text
    assert g.allow_follow_up("understand_enquiry", "enquiry") is True
    assert g.allowed == [(1, 1)]


def test_guard_allows_short_filler_before_transition():
    g = TransitionResponseGuard(session_id=1, enabled=True)
    _feed(g, "సరేనండి,", " ఒక్క నిమిషం")
    assert g.allow_follow_up("demo_callback", "callback") is True


def test_guard_resets_on_each_generation_and_barge_in():
    g = TransitionResponseGuard(session_id=5, enabled=True)
    _feed(g, RUN238_TURN4)
    # caller barges in; the next caller turn starts a fresh generation
    g.observe_llm_frame(InterruptionFrame())
    _feed(g)  # new generation: tool call only
    assert g.allow_follow_up("understand_enquiry", "enquiry") is True
    assert g.generation_seq == 2


def test_skip_tts_text_is_not_counted_as_spoken():
    g = TransitionResponseGuard(session_id=1, enabled=True)
    g.observe_llm_frame(LLMFullResponseStartFrame())
    f = LLMTextFrame(RUN238_TURN4)
    f.skip_tts = True
    g.observe_llm_frame(f)
    assert g.allow_follow_up("x", "y") is True


def test_guard_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DOGRAH_TRANSITION_DEDUPE", "false")
    g = TransitionResponseGuard(session_id=1)
    _feed(g, RUN238_TURN4)
    assert g.allow_follow_up("x", "y") is True


# ---------------------------------------------------------------- engine wiring
class _FakeLLM(FrameProcessor):
    """A real pipecat FrameProcessor standing in for the pipeline LLM."""


async def _emit(llm, frame):
    await llm._call_event_handler("on_before_push_frame", frame)


def _engine(three_node_workflow):
    llm = _FakeLLM()
    engine = PipecatEngine(
        llm=llm,
        context=LLMContext(),
        workflow=three_node_workflow,
        call_context_vars={},
        workflow_run_id=4242,
    )
    engine.set_node = AsyncMock()
    engine._perform_variable_extraction_if_needed = AsyncMock()
    engine._current_node = SimpleNamespace(is_end=False)
    return engine, llm


async def _run_transition(engine):
    fn = await engine._create_transition_func("collect_info", "agent")
    params = SimpleNamespace(arguments={}, result_callback=AsyncMock())
    await fn(params)
    engine.set_node.assert_awaited_once_with("agent")
    args, kwargs = params.result_callback.await_args
    assert args[0] == {"status": "done"}
    return kwargs["properties"]


@pytest.mark.asyncio
async def test_engine_transition_after_spoken_answer_does_not_rerun_llm(three_node_workflow):
    engine, llm = _engine(three_node_workflow)
    assert engine._turn_guard_attached
    await _emit(llm, LLMFullResponseStartFrame())
    for chunk in RUN238_TURN4.split(" "):
        await _emit(llm, LLMTextFrame(chunk + " "))
    props = await _run_transition(engine)
    assert props.run_llm is False  # no second generation for this caller turn
    assert props.on_context_updated is not None  # end-node handling still runs
    assert engine._turn_guard.suppressed == [(4242, 1)]


@pytest.mark.asyncio
async def test_engine_transition_without_speech_keeps_default_generation(three_node_workflow):
    engine, llm = _engine(three_node_workflow)
    await _emit(llm, LLMFullResponseStartFrame())
    props = await _run_transition(engine)
    assert props.run_llm is None  # pipecat default -> new node answers the turn


@pytest.mark.asyncio
async def test_engine_transition_after_filler_keeps_default_generation(three_node_workflow):
    engine, llm = _engine(three_node_workflow)
    await _emit(llm, LLMFullResponseStartFrame())
    await _emit(llm, LLMTextFrame("సరేనండి"))
    props = await _run_transition(engine)
    assert props.run_llm is None


@pytest.mark.asyncio
async def test_engine_each_turn_is_independent(three_node_workflow):
    engine, llm = _engine(three_node_workflow)
    await _emit(llm, LLMFullResponseStartFrame())
    await _emit(llm, LLMTextFrame(RUN238_TURN4))
    assert (await _run_transition(engine)).run_llm is False
    engine.set_node.reset_mock()
    # next caller turn: generation issues only the tool call
    await _emit(llm, LLMFullResponseStartFrame())
    assert (await _run_transition(engine)).run_llm is None


def test_custom_tools_do_not_consult_the_transition_guard():
    """Tool-call acknowledgements (booking etc.) keep their own run_llm semantics."""
    import inspect

    from api.services.workflow import pipecat_engine_custom_tools as ct

    assert "_turn_guard" not in inspect.getsource(ct)
