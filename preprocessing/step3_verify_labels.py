#!/usr/bin/env python3
"""Step 3 - Semantic model checker via API (paper-faithful).

Implements the semantic half of AgentDoG's dual-layer QC (1.5 paper §3.2.1
Stage 3): "rationality, step-to-step coherence, goal alignment, factual
plausibility, and consistency between the trajectory behavior and its assigned
risk labels."

For each sample the teacher API answers four checks in one call
(prompts/model_checker.txt):
  coherence / goal_align / plausible  -> trajectory quality gates
  verdict (blind, gold never shown)   -> label consistency

Scoring (written to scores/label_confidence.json, consumed by step5):
  confidence = 1.0   verdict agrees + all quality checks pass
               0.7   verdict agrees, some quality check failed
               0.2   verdict conflicts but quality passes (label noise suspect)
               0.0   verdict conflicts + quality failed (drop candidate)

Samples with confidence <= 0.2 are listed in step3/suspicious.json for review.
Requires the teacher API (api_client.py). Budget tip: run on a step5 shortlist
via --input, not the whole pool.
"""

from __future__ import annotations

import argparse
import json
import re

from api_client import TeacherAPI
from common import extract_trajectory_from_instruction, load_json, save_json, write_text
import config

_PROMPT = config.REPO_ROOT / "prompts" / "model_checker.txt"
_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_checker(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict", "")).lower()
    if verdict not in {"safe", "unsafe"}:
        return None
    return {
        "coherence": str(obj.get("coherence", "YES")).upper() == "YES",
        "goal_align": str(obj.get("goal_align", "YES")).upper() == "YES",
        "plausible": str(obj.get("plausible", "YES")).upper() == "YES",
        "verdict": verdict,
    }


def confidence_of(checks: dict, gold: str) -> float:
    agree = checks["verdict"] == gold
    quality_ok = checks["coherence"] and checks["goal_align"] and checks["plausible"]
    if agree and quality_ok:
        return 1.0
    if agree:
        return 0.7
    if quality_ok:
        return 0.2
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic model checker (API)")
    parser.add_argument("--input", default=None, help="defaults to step2 valid_pool.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config.ensure_dirs()
    api = TeacherAPI()
    template = _PROMPT.read_text(encoding="utf-8")
    in_path = args.input or (config.STEP_DIRS["step2"] / "valid_pool.json")
    pool = load_json(in_path)
    if args.limit:
        pool = pool[: args.limit]
    print(f"Model-checking {len(pool)} samples via {api.model}")

    confidence: dict[str, float] = {}
    suspicious = []
    detail = []
    stats = {"agree": 0, "quality_fail": 0, "parse_fail": 0}
    for i, rec in enumerate(pool):
        traj = extract_trajectory_from_instruction(rec["instruction"])
        prompt = template.replace("{trajectory}", traj)
        checks = parse_checker(api.chat([{"role": "user", "content": prompt}], max_tokens=64))
        if checks is None:
            stats["parse_fail"] += 1
            confidence[rec["uid"]] = 0.5  # unknown -> neutral
            continue
        conf = confidence_of(checks, rec["label"])
        confidence[rec["uid"]] = conf
        if checks["verdict"] == rec["label"]:
            stats["agree"] += 1
        if not (checks["coherence"] and checks["goal_align"] and checks["plausible"]):
            stats["quality_fail"] += 1
        if conf <= 0.2:
            suspicious.append({"uid": rec["uid"], "gold": rec["label"], **checks})
        detail.append({"uid": rec["uid"], "gold": rec["label"], "conf": conf, **checks})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pool)}  agree={stats['agree']}")

    save_json(confidence, config.STEP_DIRS["scores"] / "label_confidence.json")
    save_json(suspicious, config.STEP_DIRS["step3"] / "suspicious.json")
    save_json(detail, config.STEP_DIRS["step3"] / "checker_detail.json")
    n = max(len(pool), 1)
    report = [
        "# Step 3 - Semantic Model Checker Report",
        "",
        f"- checked: {len(pool)}",
        f"- verdict agrees with gold: **{stats['agree']}** ({stats['agree']/n:.1%})",
        f"- quality-check failures (coherence/goal/plausible): **{stats['quality_fail']}**",
        f"- unparseable API replies (neutral 0.5): **{stats['parse_fail']}**",
        f"- suspicious (conf <= 0.2): **{len(suspicious)}**",
        "",
        "Confidence tiers: 1.0 agree+quality / 0.7 agree only / 0.2 conflict+quality /"
        " 0.0 conflict+bad quality.",
        "",
    ]
    write_text("\n".join(report), config.STEP_DIRS["step3"] / "verify_report.md")
    print("\n".join(report))


if __name__ == "__main__":
    main()
