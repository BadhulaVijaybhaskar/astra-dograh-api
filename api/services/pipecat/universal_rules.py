"""Platform rules shared by every AI Employee (workflow_configurations.astra_universal_rules).

The console owns the text (console/lib/universal-rules.js) and writes the same value to every
employee's workflow, hand-built graphs included, so one block reaches all employees through one
code path. Nothing here knows any workflow id.

Shape: {"version": int, "text": str, "booking_hint": str containing "{tools}"}.
- text: appended once to the global node prompt (reaches nodes with add_global_prompt=True) and to
  the own prompt of every start/agent node that does not include the global prompt.
- booking_hint: appended to the own prompt of each start/agent node that has a booking-type tool or
  stage (transition or custom tool whose function name looks like demo/book/callback/schedule/
  appointment/follow-up), with {tools} replaced by those function names. Keyed only on the node's
  own tools, so it follows whatever booking tool an employee has.
No-op when the key is missing, disabled or malformed. Idempotent.
"""

import re
from typing import Any, Optional

UNIVERSAL_RULES_KEY = "astra_universal_rules"
_MAX_TEXT = 6000
_BOOKING_RE = re.compile(r"(demo|book|callback|call_back|schedul|appoint|follow_up)", re.I)
_LLM_NODE_TYPES = {"startCall", "agentNode"}
_NOT_BOOKING_RE = re.compile(r"(cancel|reschedul|check|availab|list|lookup|^get_)", re.I)


def parse_universal_rules(run_configs: Optional[dict]) -> Optional[dict]:
    cfg = run_configs if isinstance(run_configs, dict) else {}
    raw = cfg.get(UNIVERSAL_RULES_KEY)
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None
    text = str(raw.get("text") or "").strip()
    if not text or len(text) > _MAX_TEXT:
        return None
    hint = str(raw.get("booking_hint") or "").strip()
    if hint and ("{tools}" not in hint or len(hint) > 500):
        hint = ""
    return {"version": raw.get("version"), "text": text, "booking_hint": hint}


def is_booking_function(name: str) -> bool:
    n = str(name or "")
    return bool(_BOOKING_RE.search(n)) and not _NOT_BOOKING_RE.search(n)


def node_booking_functions(node: Any, tool_names: Optional[dict] = None) -> list[str]:
    names: list[str] = []
    for edge in getattr(node, "out_edges", None) or []:
        try:
            fn = edge.get_function_name()
        except Exception:
            continue
        if is_booking_function(fn) and fn not in names:
            names.append(fn)
    for uuid in getattr(node, "tool_uuids", None) or []:
        fn = (tool_names or {}).get(uuid)
        if fn and is_booking_function(fn) and fn not in names:
            names.append(fn)
    return names


def apply_universal_rules_to_graph(
    workflow_graph: Any, run_configs: Optional[dict], tool_names: Optional[dict] = None
) -> Optional[dict]:
    rules = parse_universal_rules(run_configs)
    if not rules:
        return None
    text, hint = rules["text"], rules["booking_hint"]
    marker = text.splitlines()[0].strip()
    nodes = getattr(workflow_graph, "nodes", {}) or {}
    global_id = getattr(workflow_graph, "global_node_id", None)
    touched: list[str] = []
    hinted: dict[str, list[str]] = {}

    def _append(node: Any, block: str) -> bool:
        prompt = getattr(node, "prompt", None) or ""
        if block in prompt:
            return False
        node.prompt = (prompt.rstrip() + "\n\n" + block) if prompt.strip() else block
        return True

    if global_id and global_id in nodes and marker not in (nodes[global_id].prompt or ""):
        if _append(nodes[global_id], text):
            touched.append(str(global_id))
    for nid, node in nodes.items():
        # Only conversation nodes the LLM speaks from (start/agent) with a prompt of their own;
        # end, webhook, trigger and qa nodes are left untouched.
        if nid == global_id or getattr(node, "node_type", None) not in _LLM_NODE_TYPES:
            continue
        if not str(getattr(node, "prompt", None) or "").strip():
            continue
        gets_global = bool(global_id) and bool(getattr(node, "add_global_prompt", True))
        if not gets_global and marker not in (node.prompt or ""):
            if _append(node, text):
                touched.append(str(nid))
        if hint:
            fns = node_booking_functions(node, tool_names)
            if fns and _append(node, hint.replace("{tools}", ", ".join(fns))):
                hinted[str(nid)] = fns
    return {"version": rules["version"], "nodes": touched, "booking_hint": hinted}


async def load_tool_function_names(workflow_graph: Any, organization_id: Any) -> dict:
    """tool_uuid -> LLM function name for the custom tools used by this graph's nodes."""
    uuids: list[str] = []
    for node in (getattr(workflow_graph, "nodes", {}) or {}).values():
        for u in getattr(node, "tool_uuids", None) or []:
            if u and u not in uuids:
                uuids.append(u)
    if not uuids or organization_id is None:
        return {}
    from api.db import db_client
    from api.services.workflow.tools.custom_tool import custom_tool_function_name

    tools = await db_client.get_tools_by_uuids(uuids, organization_id)
    return {t.tool_uuid: custom_tool_function_name(t.name) for t in tools if getattr(t, "name", None)}
