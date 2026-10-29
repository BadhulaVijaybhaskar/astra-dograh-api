"""Generic Product Knowledge + Configured Actions/Tools injection (platform-wide).

A workflow opts in with workflow_configurations.product_knowledge:

    {"version": 1,
     "sets": [{"name": str,
               "confirmed": [str, ...],          # safe to say
               "do_not_claim": [str, ...],       # never claim unless a tool provides it
               "fallback_examples": [{"when": str, "reply": str}, ...]}],
     "employee_knowledge": [{"title": str, "text": str}, ...]}   # optional

Only Active sets are synced here by astra-ai. The configured actions/tools are
NOT taken from config text: they are detected from the workflow graph's own
tool_uuids at call time, so the model is told exactly what this workflow can
do. No-op when product_knowledge is absent (every existing workflow).

Assembly order per node: employee instructions (node prompt) -> regional
language style -> product knowledge -> employee knowledge -> configured
actions/tools. Workflow context and data-collection fields are composed by the
engine as before.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

PRODUCT_KNOWLEDGE_KEY = "product_knowledge"
MARKER = "# PRODUCT KNOWLEDGE (PLATFORM"

_DISABLED_RE = re.compile(r"\b(disabled|not enabled|not configured|inactive)\b", re.I)
_BOOK_RE = re.compile(r"(book|appointment|booking|schedule)", re.I)
_NOT_BOOK_RE = re.compile(r"(cancel|reschedul|check|availability|list|lookup|get_)", re.I)

_MAX_ITEMS = 40
_MAX_ITEM_CHARS = 400


def _clean_list(values: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return out
    for v in values:
        s = " ".join(str(v or "").split())[:_MAX_ITEM_CHARS]
        if s and s not in out:
            out.append(s)
        if len(out) >= _MAX_ITEMS:
            break
    return out


def normalize_product_knowledge(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    sets = []
    for s in raw.get("sets") or []:
        if not isinstance(s, dict):
            continue
        if str(s.get("status") or "active").lower() != "active":
            continue  # only Active facts reach runtime
        confirmed = _clean_list(s.get("confirmed"))
        do_not = _clean_list(s.get("do_not_claim"))
        examples = []
        for ex in s.get("fallback_examples") or []:
            if isinstance(ex, dict) and str(ex.get("reply") or "").strip():
                examples.append({
                    "when": " ".join(str(ex.get("when") or "").split())[:160],
                    "reply": " ".join(str(ex.get("reply") or "").split())[:_MAX_ITEM_CHARS],
                })
        if confirmed or do_not or examples:
            sets.append({"name": str(s.get("name") or "Product facts")[:120],
                         "confirmed": confirmed, "do_not_claim": do_not,
                         "fallback_examples": examples[:12]})
    knowledge = []
    for k in raw.get("employee_knowledge") or []:
        if isinstance(k, dict) and str(k.get("text") or "").strip():
            knowledge.append({"title": str(k.get("title") or "")[:120],
                              "text": str(k.get("text") or "").strip()[:2000]})
    if not sets and not knowledge:
        return None
    return {"sets": sets, "employee_knowledge": knowledge[:10]}


def collect_tool_uuids(workflow_graph: Any) -> list[str]:
    seen: list[str] = []
    for node in (getattr(workflow_graph, "nodes", {}) or {}).values():
        for u in getattr(node, "tool_uuids", None) or []:
            if u and u not in seen:
                seen.append(u)
    return seen


def describe_tools(tools: Iterable[Any]) -> list[dict]:
    """ToolModel rows -> [{name, description, disabled, booking}] (no secrets, no URLs)."""
    out = []
    for t in tools or []:
        name = str(getattr(t, "name", "") or "").strip()
        if not name:
            continue
        desc = " ".join(str(getattr(t, "description", "") or "").split())[:220]
        out.append({
            "name": name,
            "description": desc,
            "disabled": bool(_DISABLED_RE.search(desc)),
            "booking": bool(_BOOK_RE.search(name)) and not _NOT_BOOK_RE.search(name),
        })
    out.sort(key=lambda x: x["name"])
    return out


async def load_configured_tools(workflow_graph: Any, organization_id: Any) -> list[dict]:
    uuids = collect_tool_uuids(workflow_graph)
    if not uuids or organization_id is None:
        return []
    from api.db import db_client

    tools = await db_client.get_tools_by_uuids(uuids, organization_id)
    return describe_tools(tools)


def build_product_knowledge_prompt(pk: dict, tools: list[dict], language: str = "") -> str:
    te = str(language or "").lower().startswith("te")
    lines: list[str] = [
        "# PRODUCT KNOWLEDGE (PLATFORM, HIGHEST PRIORITY FOR FACTS)",
        "Answer questions about what the product can do ONLY from the confirmed facts and the configured "
        "actions/tools below. This section overrides every other instruction, example or word list about "
        "capabilities. Word lists elsewhere (for example which English words to keep in Latin letters) are "
        "spelling rules only and never mean a feature or integration exists.",
    ]
    confirmed: list[str] = []
    do_not: list[str] = []
    examples: list[dict] = []
    for s in pk.get("sets") or []:
        confirmed += [c for c in s["confirmed"] if c not in confirmed]
        do_not += [c for c in s["do_not_claim"] if c not in do_not]
        examples += s["fallback_examples"]
    if confirmed:
        lines += ["", "CONFIRMED CAPABILITIES (safe to say):"] + [f"- {c}" for c in confirmed]

    lines += ["", "CONFIGURED ACTIONS/TOOLS ON THIS WORKFLOW (detected from the workflow configuration):"]
    if tools:
        for t in tools:
            if t["disabled"]:
                lines.append(f"- {t['name']}: present but DISABLED. Do not offer it or describe it as working.")
            else:
                lines.append(f"- {t['name']}: {t['description']}" if t["description"] else f"- {t['name']}")
    else:
        lines.append("- None.")
    lines.append("Only these actions exist. Never say an action happened unless its tool call returned success "
                 "in this conversation.")

    if do_not:
        lines += ["", "NOT CONFIRMED (never claim, promise or imply these unless a configured, non-disabled "
                      "tool above clearly provides it):"] + [f"- {c}" for c in do_not]
    lines += [
        "",
        "WHEN ASKED ABOUT ANYTHING NOT CONFIRMED (an integration, channel, CRM, price, number or feature):",
        "- Say plainly, in one short sentence, that it is not confirmed or not configured in this setup, then "
        "continue the conversation. Do not say it can be connected, do not say it works, do not guess, and do "
        "not say it is impossible.",
        "- Never invent prices, plans, timelines, customer names, numbers or integrations.",
    ]
    if examples:
        lines.append("Examples:")
        for ex in examples:
            lines.append(f"- {ex['when']}: \"{ex['reply']}\"" if ex["when"] else f"- \"{ex['reply']}\"")

    booking = [t for t in tools if t["booking"] and not t["disabled"]]
    lines += ["", "DEMO / APPOINTMENT BOOKING:"]
    if booking:
        names = ", ".join(t["name"] for t in booking)
        start = ("\"We can start the demo booking process.\" (in Telugu: \"Demo booking process start చేద్దాం.\")"
                 if te else "\"We can start the demo booking process.\"")
        lines += [
            f"- You may say {start} and then follow the booking steps.",
            f"- Say the demo is booked or confirmed ONLY after {names} returned a confirmed success in this "
            "conversation. If it failed, returned an error, or has not been called yet, never say booked or "
            "confirmed; say it is not confirmed yet and offer another slot or a team follow-up.",
        ]
    else:
        lines.append("- No booking tool is configured. Never say a demo or appointment is booked; offer that the "
                     "team will follow up.")

    knowledge = pk.get("employee_knowledge") or []
    if knowledge:
        lines += ["", "# EMPLOYEE KNOWLEDGE (business-specific facts, also confirmed)"]
        for k in knowledge:
            lines.append(f"- {k['title']}: {k['text']}" if k["title"] else f"- {k['text']}")
    return "\n".join(lines).strip()


def apply_product_knowledge_to_graph(
    workflow_graph: Any, run_configs: Optional[dict], tools: Optional[list[dict]] = None
) -> Optional[dict]:
    cfg = run_configs if isinstance(run_configs, dict) else {}
    pk = normalize_product_knowledge(cfg.get(PRODUCT_KNOWLEDGE_KEY))
    if not pk:
        return None
    block = build_product_knowledge_prompt(pk, tools or [], str(cfg.get("primary_language") or ""))
    nodes = getattr(workflow_graph, "nodes", {}) or {}
    global_id = getattr(workflow_graph, "global_node_id", None)
    touched: list[str] = []

    def _append(node: Any) -> None:
        prompt = getattr(node, "prompt", None)
        if not prompt or MARKER in prompt:
            return
        node.prompt = prompt.rstrip() + "\n\n" + block
        touched.append(str(getattr(node, "id", "?")))

    if global_id and global_id in nodes:
        _append(nodes[global_id])
    for nid, node in nodes.items():
        if nid == global_id:
            continue
        if not (bool(global_id) and bool(getattr(node, "add_global_prompt", True))):
            _append(node)
    return {"sets": len(pk["sets"]), "tools": [t["name"] for t in tools or []], "nodes": touched,
            "chars": len(block)}
