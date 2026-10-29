"""Unit tests: per-call caller facts store + employee identity (no network)."""
import asyncio

from api.services.pipecat import caller_facts as cf
from api.services.pipecat.regional_language_style import build_regional_style_prompt


CFG = {
    "primary_language": "te-IN",
    "caller_facts": {"enabled": True, "fields": [{"key": "property_type", "label": "Property type", "options": ["Flat", "Villa"]}]},
    "astra_employee": {"id": "emp_a", "name": "Anjali"},
}


def test_disabled_without_config():
    assert cf.is_enabled({}) is False
    assert cf.is_enabled({"caller_facts": {"enabled": False}}) is False
    assert cf.is_enabled(CFG) is True


def test_fields_default_plus_data_collection():
    keys = [f["key"] for f in cf.resolve_fields(CFG)]
    for k in ("caller_name", "company_name", "lead_volume", "sales_team_size", "business_type", "requirement",
              "languages_required", "location", "budget", "crm", "preferred_demo_time", "email", "phone"):
        assert k in keys
    assert keys[-1] == "property_type"


def test_identity_from_config_then_context():
    assert cf.identity_context_vars(CFG, {"employee_name": "Old"}) == {"employee_name": "Anjali", "employee_id": "emp_a"}
    assert cf.identity_context_vars({}, {"employee_name": "Kavya"})["employee_name"] == "Kavya"
    assert cf.identity_context_vars({}, {}) == {}


def test_store_never_erases_known_fact():
    s = cf.CallerFactsStore(cf.resolve_fields(CFG))
    s.update({"caller_name": "Ravi", "lead_volume": "120 per day"})
    s.update({"caller_name": None, "lead_volume": "", "sales_team_size": "8 members"})
    assert s.snapshot() == {"caller_name": "Ravi", "lead_volume": "120 per day", "sales_team_size": "8 members"}


def test_block_separates_employee_and_caller():
    s = cf.CallerFactsStore(cf.resolve_fields(CFG))
    s.update({"caller_name": "Ravi"})
    block = cf.build_facts_block(s, CFG, "Anjali")
    assert cf.MARKER in block
    assert "Your own name (the AI employee) is Anjali" in block
    assert "- Caller name: Ravi" in block
    assert "Ravi గారు" in block
    assert "Never ask again" in block


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    async def run_inference(self, ctx, system_instruction=None):
        return self.reply


class _FakeEngine:
    def __init__(self, reply):
        self._call_context_vars = {}
        self._gathered_context = {"company_name": "Acme"}
        self.variable_extraction_llm = _FakeLLM(reply)
        self.refreshed = 0

    async def refresh_caller_facts_prompt(self):
        self.refreshed += 1


def test_tracker_extracts_and_survives_node_changes():
    eng = _FakeEngine('{"caller_name": "Asha", "lead_volume": "80 per day", "sales_team_size": "6 members"}')
    t = cf.CallerFactsTracker(eng, CFG)

    async def run():
        await t.before_generation([{"role": "assistant", "content": "చెప్పండి. మీ పేరు ఏంటి?"}, {"role": "user", "content": "Hi, నా పేరు Asha."}])
        await t.wait_idle()
    asyncio.run(run())
    assert t.store.facts["caller_name"] == "Asha"
    assert eng.refreshed >= 1
    assert t.store.asked == ["మీ పేరు ఏంటి?"]
    block = t.prompt_block()  # merges variable-extraction values too
    assert "- Company: Acme" in block and "- Lead volume: 80 per day" in block
    assert eng._gathered_context["caller_facts"]["sales_team_size"] == "6 members"
    # same user message again does not re-extract
    t.on_context([{"role": "user", "content": "Hi, నా పేరు Asha."}])
    assert not t._pending


def test_regional_profile_is_modern_and_has_no_mechanical_opener_rule():
    for style in ("telangana", "andhra", "neutral"):
        p = build_regional_style_prompt("te-IN", style)
        assert "MODERN EVERYDAY TELUGU" in p
        assert "Join the short opener" not in p
        assert "ఎన్ని calls" not in p
        assert "You: అవునండి" not in p and "You: సరేనండి" not in p


def test_repeat_question_detection_and_earlier_window():
    from api.services.pipecat import caller_facts as cf

    st = cf.CallerFactsStore(cf.resolve_fields({"caller_facts": {"enabled": True}}))
    msgs = [
        {"role": "assistant", "content": "Okay. Which industry are your leads from?"},
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "Sure. Which language do you need?"},
        {"role": "user", "content": "hmm?"},
    ]
    st.note_assistant_questions(msgs)
    assert "Which industry are your leads from?" in st.earlier_asked
    assert "Which language do you need?" not in st.earlier_asked  # unclear reply: may re-ask once
    st.note_assistant_questions(msgs[:-1] + [{"role": "user", "content": "how will you handle them?"}])
    assert "Which language do you need?" in st.earlier_asked  # caller moved on: do not repeat
    st.note_assistant_questions(msgs)
    assert cf.is_repeat_question("Can you tell me which industry your leads are from?", st.earlier_asked)
    assert not cf.is_repeat_question("What time is convenient for the demo?", st.earlier_asked)
    assert not cf.is_repeat_question("Your leads are from retail.", st.earlier_asked)
    # a reply without questions in between: the older question becomes "earlier"
    st.note_assistant_questions(msgs + [{"role": "user", "content": "y"}, {"role": "assistant", "content": "You have 8 people."}])
    assert "Which language do you need?" in st.earlier_asked
    assert cf.is_repeat_question("Anything else?", ["anything else?"])
    # inflected Telugu ending + caller name ignored
    prev = ["Asha గారు, leads mostly ఏ sector నుండి వస్తున్నాయి?"]
    assert cf.is_repeat_question("ఇంకా మీ leads mostly ఏ sector నుండి వస్తున్నాయో చెప్పగలరా?", prev, ignore={"asha", "గారు"})
    assert not cf.is_repeat_question("Asha గారు, మీకు ఏ time convenient?", prev, ignore={"asha", "గారు"})
    # questions already spoken in the current turn (before a tool call) count as asked
    st.note_assistant_questions([{"role": "user", "content": "book"}, {"role": "assistant", "content": "Sure. What time suits you?"}])
    assert "What time suits you?" in st.earlier_asked


def test_repeat_question_guard_drops_only_when_reply_has_other_text():
    import asyncio
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
    from api.services.pipecat import caller_facts as cf

    class T:
        pass

    tr = T()
    tr.store = cf.CallerFactsStore(cf.resolve_fields({"caller_facts": {"enabled": True}}))
    tr.store.earlier_asked = ["Which industry are your leads from?"]
    g = cf.build_repeat_question_guard(tr)
    out = []

    async def push(frame, direction=None):
        if isinstance(frame, LLMTextFrame):
            out.append(frame.text)

    g.push_frame = push

    async def run(chunks):
        out.clear()
        await g.process_frame(LLMFullResponseStartFrame(), None)
        for c in chunks:
            f = LLMTextFrame(c)
            g._src = f
            g._buf += c
            await g._drain(None, final=False)
        await g._drain(None, final=True)
        return "".join(out)

    said = asyncio.run(run(["You get 120 leads a day. ", "Which industry are ", "your leads from?"]))
    assert "120 leads" in said and "industry" not in said
    said = asyncio.run(run(["Which industry are your leads from?"]))
    assert "industry" in said  # never leave a reply empty
