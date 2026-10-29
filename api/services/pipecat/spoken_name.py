"""Astra: runtime spoken name (canonical contact name stays untouched).

canonical_name : the contact name as stored (language-neutral, e.g. "Ram").
spoken_language: the language the employee is speaking on this call.
spoken_name    : how that name is written for TTS in that language's script,
                 e.g. "राम" / "రామ" -> "రామ్" for Telugu. Runtime only; never
                 written back to the contact.

Deterministic, no network: Indic -> Indic via the aligned Unicode blocks, with
word-final schwa deletion (Hindi "राम" is pronounced Ram, so Telugu gets a
virama: రామ్). Latin names are passed through unchanged (the TTS reads Latin
names in Indic sentences as English words, which is how "Ram"/"Vijay" sound).
"""
from __future__ import annotations

import re
import unicodedata

_BLOCKS = {  # script -> (block start, virama)
    "deva": (0x0900, "\u094D"), "beng": (0x0980, "\u09CD"), "guru": (0x0A00, "\u0A4D"),
    "gujr": (0x0A80, "\u0ACD"), "orya": (0x0B00, "\u0B4D"), "taml": (0x0B80, "\u0BCD"),
    "telu": (0x0C00, "\u0C4D"), "knda": (0x0C80, "\u0CCD"), "mlym": (0x0D00, "\u0D4D"),
}
_LANG_SCRIPT = {"hi": "deva", "mr": "deva", "bn": "beng", "pa": "guru", "gu": "gujr", "od": "orya",
                "or": "orya", "ta": "taml", "te": "telu", "kn": "knda", "ml": "mlym"}
_SCHWA_DELETING = {"deva", "guru", "gujr", "beng"}  # word-final inherent 'a' is silent


def _script_of(ch: str) -> str | None:
    cp = ord(ch)
    for s, (base, _) in _BLOCKS.items():
        if base <= cp < base + 0x80:
            return s
    return None


def name_script(name: str) -> str | None:
    for ch in name or "":
        s = _script_of(ch)
        if s:
            return s
        if ch.isalpha() and ch.isascii():
            return "latn"
    return None


def _is_consonant(ch: str) -> bool:
    try:
        return unicodedata.name(ch).split(" ")[1] == "LETTER" and "VOWEL" not in unicodedata.name(ch) \
            and unicodedata.category(ch) == "Lo"
    except (ValueError, IndexError):
        return False


def to_spoken(name: str, language: str | None) -> str:
    name = (name or "").strip()
    lang = (language or "").split("-")[0].lower()
    target = _LANG_SCRIPT.get(lang)
    src = name_script(name)
    if not name or not target or src in (None, "latn", target):
        return name
    sbase, svir = _BLOCKS[src]
    tbase, tvir = _BLOCKS[target]
    out = []
    for word in re.split(r"(\s+)", name):
        if not word.strip():
            out.append(word); continue
        mapped = []
        for ch in word:
            cp = ord(ch)
            if sbase <= cp < sbase + 0x80:
                t = chr(tbase + (cp - sbase))
                mapped.append(t if unicodedata.category(t) != "Cn" else ch)
            else:
                mapped.append(ch)
        w = "".join(mapped)
        if src in _SCHWA_DELETING and target not in _SCHWA_DELETING and w and _is_consonant(w[-1]) \
                and len(word) > 1:
            w += tvir
        out.append(w)
    return "".join(out)


def spoken_vars(canonical: str, language: str | None) -> dict:
    return {"canonical_name": canonical or "", "spoken_language": language or "",
            "spoken_name": to_spoken(canonical, language)}
