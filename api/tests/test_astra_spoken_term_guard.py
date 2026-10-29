import pytest

from api.services.pipecat.spoken_term_guard import (
    SpokenTermRewriter,
    build_spoken_term_guard,
    parse_replacements,
)

WF9 = [
    ["leads qualification", "సరైన leads ని identify చేయడం"],
    ["lead qualification", "సరైన leads ని identify చేయడం"],
    ["qualified leads", "interest ఉన్న leads"],
    ["qualifications", "సరైన leads ని identify చేయడం"],
    ["qualification", "సరైన leads ని identify చేయడం"],
    ["qualifying", "identify చేయడం"],
    ["qualify", "identify"],
    ["qualified", "interest ఉన్న"],
]


def _stream(rw, text, sizes=(1, 2, 3, 5, 7)):
    out, i, k = [], 0, 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out.append(rw.feed(text[i : i + n]))
        i += n
        k += 1
    out.append(rw.flush())
    return "".join(out)


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            "మీకు leads qualification కోసం ప్రత్యేక criteria ఉన్నాయా?",
            "మీకు సరైన leads ని identify చేయడం కోసం ప్రత్యేక criteria ఉన్నాయా?",
        ),
        ("Astra Voice leads ని qualify చేస్తుంది.", "Astra Voice leads ని identify చేస్తుంది."),
        ("Qualified leads మాత్రమే పంపిస్తుంది.", "interest ఉన్న leads మాత్రమే పంపిస్తుంది."),
        ("మీ leads తో మాట్లాడుతుంది.", "మీ leads తో మాట్లాడుతుంది."),
        ("quality చాలా బాగుంటుంది", "quality చాలా బాగుంటుంది"),
    ],
)
def test_streamed_rewrite_any_chunking(src, expected):
    for sizes in [(1,), (2, 3), (4, 1, 6), (50,)]:
        rw = SpokenTermRewriter(parse_replacements(WF9))
        assert _stream(rw, src, sizes) == expected
        assert "qualif" not in _stream(SpokenTermRewriter(parse_replacements(WF9)), src, sizes).lower()


def test_emits_text_promptly_and_only_holds_tail():
    rw = SpokenTermRewriter(parse_replacements(WF9))
    assert rw.feed("సరేనండి, Astra Voice ") == "సరేనండి, Astra Voice "
    assert rw.feed("మీ leads") == "మీ "  # 'leads' may start a configured phrase
    assert rw.feed(" తో") == "leads "
    assert rw.flush() == "తో"


def test_absent_config_inserts_nothing():
    assert build_spoken_term_guard({}) is None
    assert build_spoken_term_guard(None) is None
    assert build_spoken_term_guard({"spoken_term_replacements": []}) is None
    assert build_spoken_term_guard({"spoken_term_replacements": WF9}) is not None
