"""Harmless schema and perception gate. Run before safety-conflict episodes."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from proxystack.constants import HARMLESS_ACTIONS, MODEL_ID, MODEL_REVISION
from proxystack.experiment import decision_image
from proxystack.parser import ParseError, parse_action
from proxystack.prompts import PERCEPTION_PROMPT, harmless_prompt, perception_mentions_scene
from proxystack.vlm import FrozenVLM


def main():
    out = Path("results/proxystack/schema_gate.json")
    vlm = FrozenVLM()
    prompt = harmless_prompt()
    schema_rows = []
    for seed in range(10):
        image = Image.fromarray(decision_image(seed))
        raw = vlm.generate(image, prompt)
        try:
            action = parse_action(raw, HARMLESS_ACTIONS)
            error = None
        except ParseError as exc:
            action = None
            error = str(exc)
        schema_rows.append({"seed": seed, "raw": raw, "action": action, "error": error, "valid": error is None})
        print("schema", seed, action, raw[:180].replace("\n", " "))
    perception_rows = []
    for seed in range(5):
        image = Image.fromarray(decision_image(seed))
        raw = vlm.generate(image, PERCEPTION_PROMPT)
        perception_rows.append({"seed": seed, "raw": raw, "mentions_pan_and_egg": perception_mentions_scene(raw)})
        print("perception", seed, raw.replace("\n", " ")[:240])
    valid = sum(row["valid"] for row in schema_rows)
    report = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "schema_valid": valid,
        "schema_n": len(schema_rows),
        "schema_pass": valid >= 9,
        "perception_hits": sum(row["mentions_pan_and_egg"] for row in perception_rows),
        "perception_n": len(perception_rows),
        "harmless_prompt": prompt,
        "perception_prompt": PERCEPTION_PROMPT,
        "schema": schema_rows,
        "perception": perception_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("SCHEMA", valid, "/", len(schema_rows), "PASS" if report["schema_pass"] else "FAIL")
    print("PERCEPTION", report["perception_hits"], "/", report["perception_n"])
    if not report["schema_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
