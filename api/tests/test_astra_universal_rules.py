"""Platform rules shared by every AI Employee (workflow_configurations.astra_universal_rules)."""

from types import SimpleNamespace

from api.services.pipecat import universal_rules as ur

TEXT = "# PLATFORM RULES (same for every AI Employee)\n- Never give any price."
HINT = "- Booking tool for this step: {tools}. Call it right away."


def _edge(fn):
    return SimpleNamespace(get_function_name=lambda: fn)


def _node(nid, prompt, *, add_global=True, end=False, edges=(), tools=(), kind=None):
    kind = kind or ("endCall" if end else "agentNode")
    return SimpleNamespace(id=nid, prompt=prompt, add_global_prompt=add_global, is_end=end, node_type=kind,
                           out_edges=[_edge(e) for e in edges], tool_uuids=list(tools))


def _graph():
    nodes = {
        "global": _node("global", "Persona."),
        "start": _node("start", "Greet.", add_global=False, kind="startCall", edges=["demo___callback", "pricing_question", "end_call"]),
        "enquiry": _node("enquiry", "Answer.", edges=["arrange_follow_up", "goodbye"]),
        "booking": _node("booking", "Book.", tools=["u-book", "u-check"]),
        "end": _node("end", "Bye.", add_global=False, end=True, edges=["book_again"]),
        "save": _node("save", None, add_global=False, kind="webhook"),
        "qa": _node("qa", "Score the call.", add_global=False, kind="qa"),
    }
    return SimpleNamespace(nodes=nodes, global_node_id="global")


def _cfg(**kw):
    return {ur.UNIVERSAL_RULES_KEY: {"version": 1, "text": TEXT, "booking_hint": HINT, **kw}}


def test_noop_without_key_or_when_disabled_or_malformed():
    for cfg in ({}, None, {ur.UNIVERSAL_RULES_KEY: "x"}, _cfg(enabled=False), _cfg(text=""), _cfg(text="x" * 7000)):
        g = _graph()
        assert ur.apply_universal_rules_to_graph(g, cfg) is None
        assert g.nodes["start"].prompt == "Greet."


def test_text_reaches_every_node_once_and_skips_end_nodes():
    g = _graph()
    out = ur.apply_universal_rules_to_graph(g, _cfg(), {"u-book": "book_appointment", "u-check": "check_availability"})
    assert g.nodes["global"].prompt.count("# PLATFORM RULES") == 1
    assert g.nodes["start"].prompt.count("# PLATFORM RULES") == 1  # no global prompt -> own copy
    assert "# PLATFORM RULES" not in g.nodes["enquiry"].prompt  # gets it through the global prompt
    assert g.nodes["end"].prompt == "Bye."  # end node untouched
    assert g.nodes["save"].prompt is None and g.nodes["qa"].prompt == "Score the call."  # webhook / qa untouched
    assert set(out["nodes"]) == {"global", "start"}


def test_booking_hint_follows_each_nodes_own_tools():
    g = _graph()
    out = ur.apply_universal_rules_to_graph(g, _cfg(), {"u-book": "book_appointment", "u-check": "check_availability"})
    assert out["booking_hint"] == {"start": ["demo___callback"], "enquiry": ["arrange_follow_up"], "booking": ["book_appointment"]}
    assert "Booking tool for this step: demo___callback." in g.nodes["start"].prompt
    assert "check_availability" not in g.nodes["booking"].prompt


def test_idempotent():
    g = _graph()
    ur.apply_universal_rules_to_graph(g, _cfg(), {})
    snap = {k: n.prompt for k, n in g.nodes.items()}
    ur.apply_universal_rules_to_graph(g, _cfg(), {})
    assert {k: n.prompt for k, n in g.nodes.items()} == snap


def test_no_global_node_every_non_end_node_gets_text():
    g = _graph()
    del g.nodes["global"]
    g.global_node_id = None
    ur.apply_universal_rules_to_graph(g, _cfg(booking_hint=""), {})
    for nid in ("start", "enquiry", "booking"):
        assert g.nodes[nid].prompt.count("# PLATFORM RULES") == 1
    assert "Booking tool" not in g.nodes["start"].prompt


def test_booking_name_filter():
    for yes in ("demo___callback", "book_appointment", "schedule_meeting", "arrange_follow_up", "request_callback"):
        assert ur.is_booking_function(yes), yes
    for no in ("check_availability", "cancel_booking", "reschedule_demo", "pricing_question", "end_call", "get_slots"):
        assert not ur.is_booking_function(no), no
