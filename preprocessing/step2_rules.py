#!/usr/bin/env python3
"""Step 2 - Rule-based quality control (paper-faithful rule checker).

Mirrors the structural half of AgentDoG's dual-layer QC (1.5 paper §3.2.1
Stage 3): "tool-call format, schema and type constraints, value constraints,
and referential integrity".

Checks per trajectory:
  structure   - trajectory present, has user + agent turns, not truncated
  tool-call   - [AGENT] action JSON blocks parse; have name + arguments
  referential - called tool names exist in the trajectory's Available-tools list
                (when that list is present)
  label       - usable safe/unsafe label

Hard failures are dropped; soft issues are recorded as flags (kept, reported).
The semantic half (coherence, label-behavior consistency) is step3's job.

Local-safe: pure text rules, no model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from common import extract_trajectory_from_instruction, load_json, save_json, write_text
import config

MIN_TRAJ_CHARS = 80
MAX_TRAJ_CHARS = 60_000  # flagged, not dropped

# [AGENT] action payloads that look like JSON tool calls
_ACTION_JSON_RE = re.compile(r"\[ACTION\]:\s*(\{.*?\})\s*(?=\n\[|$)", re.DOTALL)
_AVAIL_TOOLS_RE = re.compile(r"Available tools:\s*(\[.*?\])\s*(?:\n===|\n\n)", re.DOTALL)
# Complete{...} / ToolName{...} style actions used by some sources
_NAMED_ACTION_RE = re.compile(r"^([A-Za-z_][\w]*)\s*[\{(]", re.MULTILINE)


def parse_available_tools(traj: str) -> set[str] | None:
    """Tool names declared in the trajectory header, or None if not declared."""
    m = _AVAIL_TOOLS_RE.search(traj)
    if not m:
        return None
    try:
        tools = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    names = {t.get("name") for t in tools if isinstance(t, dict) and t.get("name")}
    return names or None


def check(rec: dict) -> tuple[list[str], list[str]]:
    """Return (hard_fails, soft_flags)."""
    fails: list[str] = []
    flags: list[str] = []
    traj = extract_trajectory_from_instruction(rec["instruction"])
    low = traj.lower()

    # --- structure ----------------------------------------------------------
    if len(traj) < MIN_TRAJ_CHARS:
        fails.append("trajectory_too_short")
    if "[user]" not in low:
        fails.append("no_user_turn")
    if "[agent]" not in low:
        fails.append("no_agent_turn")
    stripped = traj.rstrip()
    if stripped.endswith(("[AGENT]:", "[USER]:", "[ENV]:", "[ENVIRONMENT]:", "[ACTION]:")):
        fails.append("truncated_dangling_turn")
    if len(traj) > MAX_TRAJ_CHARS:
        flags.append("long_context")

    # --- tool-call format (schema-ish) --------------------------------------
    actions = _ACTION_JSON_RE.findall(traj)
    bad_json = 0
    missing_fields = 0
    called_names: set[str] = set()
    for raw in actions:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            bad_json += 1
            continue
        name = obj.get("name")
        if not name or "arguments" not in obj:
            missing_fields += 1
        if isinstance(name, str):
            called_names.add(name)
        args = obj.get("arguments")
        if args is not None and not isinstance(args, dict):
            missing_fields += 1
    if actions:
        if bad_json == len(actions):
            fails.append("all_tool_calls_unparseable")
        elif bad_json:
            flags.append(f"some_tool_calls_unparseable:{bad_json}")
        if missing_fields:
            flags.append(f"tool_calls_missing_fields:{missing_fields}")

    # --- referential integrity ----------------------------------------------
    declared = parse_available_tools(traj)
    if declared and called_names:
        unknown = called_names - declared
        # tolerate the pseudo-action Complete / finish wrappers
        unknown = {u for u in unknown if u.lower() not in {"complete", "finish", "final_answer"}}
        if unknown:
            flags.append(f"calls_undeclared_tools:{sorted(unknown)[:3]}")

    # --- label ----------------------------------------------------------------
    if rec.get("label") not in {"safe", "unsafe"}:
        fails.append("bad_label")
    return fails, flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based QC (paper-faithful)")
    parser.add_argument("--input", default=None, help="defaults to step1 clean_pool.json")
    args = parser.parse_args()

    config.ensure_dirs()
    out_dir = config.STEP_DIRS["step2"]
    in_path = args.input or (config.STEP_DIRS["step1"] / "clean_pool.json")
    pool = load_json(in_path)
    print(f"Loaded {len(pool)} from {in_path}")

    valid: list[dict] = []
    rejects: list[dict] = []
    fail_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()

    for rec in pool:
        fails, flags = check(rec)
        for f in flags:
            flag_counter[f.split(":")[0]] += 1
        if fails:
            for f in fails:
                fail_counter[f] += 1
            rejects.append({"uid": rec["uid"], "reasons": fails, "flags": flags})
        else:
            if flags:
                rec = {**rec, "qc_flags": flags}
            valid.append(rec)

    save_json(valid, out_dir / "valid_pool.json")
    save_json(rejects, out_dir / "reject_log.json")

    label_dist = dict(Counter(r["label"] for r in valid))
    report = [
        "# Step 2 - Rule QC Report (paper-faithful)",
        "",
        f"- input: {len(pool)}",
        f"- passed: **{len(valid)}**  (labels: {label_dist})",
        f"- rejected (hard fails): **{len(rejects)}**",
        "",
        "## Hard-fail reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in fail_counter.most_common():
        report.append(f"| {reason} | {count} |")
    report += ["", "## Soft flags (kept, for downstream awareness)", "",
               "| flag | count |", "| --- | ---: |"]
    for flag, count in flag_counter.most_common():
        report.append(f"| {flag} | {count} |")
    report.append("")
    write_text("\n".join(report), out_dir / "rules_report.md")
    print("\n".join(report))
    print(f"\nWrote {out_dir / 'valid_pool.json'} ({len(valid)} rows)")


if __name__ == "__main__":
    main()
