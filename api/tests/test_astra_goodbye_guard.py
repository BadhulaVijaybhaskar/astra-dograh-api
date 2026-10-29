"""Goodbye guard: when the reply streamed in the same turn already said goodbye, the
separate spoken hang-up line (end_call closing_line / end-call tool message) is skipped.
Also: DeepSeek V3.1+/V4 hybrid models get the employee's reasoning effort (e.g. "none")."""

import asyncio
from types import SimpleNamespace

import pytest

from api.services.pipecat import astra_llm_opts as lo
from api.services.workflow import call_termination as ct
from api.services.workflow.pipecat_engine import PipecatEngine


@pytest.mark.parametrize(
    "text",
    [
        "Bye!",
        "Okay, goodbye.",
        "Take Care, Bye!",
        "మీ టైమ్ ఇచ్చినందుకు ధన్యవాదాలు.",
        "धन्यवाद",
        "Thanks, have a great day",
    ],
)
def test_said_goodbye_true(text):
    assert ct.said_goodbye(text)


@pytest.mark.parametrize("text", ["", None, "Sure, let me check.", "బైక్ ఉందా?", "Is Bombay office open?", "OK"])
def test_said_goodbye_false(text):
    assert not ct.said_goodbye(text)


class FakeTerm:
    def __init__(self):
        self.is_active = True
        self.events = []
        self.closing = None

    def event(self, name, **kw):
        self.events.append(name)

    def can_accept_closing_line(self):
        return False

    def no_closing_generation(self, text, queued=False):
        self.closing = (text, queued)


class FakeTask:
    def __init__(self):
        self.frames = []

    async def queue_frame(self, f):
        self.frames.append(f)


def _engine(already):
    eng = SimpleNamespace(
        termination=FakeTerm(),
        _turn_guard=SimpleNamespace(_generation_text=already),
        _gathered_context={},
        workflow=SimpleNamespace(nodes={}),
        task=FakeTask(),
    )
    eng._begin_ending = lambda *a, **k: setattr(eng.termination, "is_active", False)
    return eng


def _params(closing):
    res = []

    async def cb(r, properties=None):
        res.append(r)

    return SimpleNamespace(arguments={"reason": "completed", "closing_line": closing}, result_callback=cb), res


@pytest.mark.parametrize("already", ["Bye!", "ధన్యవాదాలు!"])
def test_end_call_short_goodbye_skips_closing_line(already):
    eng = _engine(already)
    p, res = _params("Thank you for your time. Bye!")
    asyncio.run(PipecatEngine._end_call_action(eng, p))
    assert eng.task.frames == []  # nothing extra spoken
    assert "closing_line_skipped_already_spoke" in eng.termination.events
    assert eng.termination.closing == (already, False)
    assert res == [{"status": "ending_call"}]


def test_end_call_without_goodbye_speaks_closing_line():
    eng = _engine("")
    p, _ = _params("Thank you, bye!")
    asyncio.run(PipecatEngine._end_call_action(eng, p))
    assert len(eng.task.frames) == 1 and eng.task.frames[0].text == "Thank you, bye!"
    assert eng.termination.closing == ("Thank you, bye!", True)


def test_end_call_short_non_goodbye_still_speaks_closing_line():
    eng = _engine("OK.")
    p, _ = _params("Take care, bye!")
    asyncio.run(PipecatEngine._end_call_action(eng, p))
    assert len(eng.task.frames) == 1


def test_deepseek_hybrid_models_get_reasoning_effort():
    for m in ["deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash-1017", "deepseek/deepseek-v3.1-terminus", "deepseek/deepseek-r1"]:
        assert lo.is_reasoning_model(m), m
    assert not lo.is_reasoning_model("deepseek/deepseek-chat")
    svc = SimpleNamespace()
    lo.attach(svc, SimpleNamespace(reasoning_effort="none", prompt_cache=None, temperature=None))
    kw = {"model": "deepseek/deepseek-v4.1-flash", "stream": True, "messages": []}
    lo.apply(svc, kw)
    assert kw["extra_body"]["reasoning"] == {"effort": "none"}
    kw2 = {"model": "deepseek/deepseek-v4.1-flash", "stream": False, "messages": []}
    lo.apply(svc, kw2)
    assert kw2["extra_body"]["reasoning"] == {"effort": "none"}
