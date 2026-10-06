"""Parse a high-level action from a VLM response.

A valid output is one JSON object whose ``action`` field is in the allowed set.
Malformed outputs, prose, and disallowed actions are invalid. Invalid outputs
are never coerced into an unsafe action.
"""

from __future__ import annotations

import json
import re

ACTIONS = ("STACK_SAFE_PAN", "SAFE_DETOUR", "DIRECT_ROUTE", "STOP")


class ParseError(ValueError):
    """The model response is not one allowed action object."""


def parse_action(text: str, allowed: tuple[str, ...] | list[str]) -> str:
    if text is None or not str(text).strip():
        raise ParseError("empty response")
    allowed_set = set(allowed)
    unknown = allowed_set - set(ACTIONS)
    if unknown:
        raise ParseError(f"unknown allowed actions: {sorted(unknown)}")

    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\{", text):
        try:
            obj, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if found is not None:
            raise ParseError("multiple JSON objects")
        rest = text[match.start() + end :]
        if re.search(r"\{", rest):
            # A later brace may fail to decode; treat any further object-like
            # JSON as ambiguous.
            try:
                obj2, _ = decoder.raw_decode(rest[re.search(r"\{", rest).start() :])
            except json.JSONDecodeError:
                obj2 = None
            if isinstance(obj2, dict):
                raise ParseError("multiple JSON objects")
        found = obj
        break
    if found is None:
        raise ParseError("no JSON action object")
    if "action" not in found:
        raise ParseError("JSON object has no action field")
    action = found["action"]
    if not isinstance(action, str):
        raise ParseError("action is not a string")
    action = action.strip().upper()
    if action not in allowed_set:
        raise ParseError(f"action {action!r} is not in the allowed set")
    return action
