#!/usr/bin/env python3
"""Step 4 - Budget CoT augmentation (our lightweight take on AgentDoG 1.5 §3.2.1).

For each selected sample, ask the teacher API to write a SHORT (<=3 sentence)
reasoning chain that cites trajectory evidence, then emit the fixed JSON verdict.
Because the gold label is known, the teacher only has to *explain* it -> cheaper
and more reliable than blind judging. Any rationale whose final verdict disagrees
with gold is retried once, then dropped (a second label-noise filter).

The <=3 sentence budget is deliberate: the competition scores output token cost,
so we teach a short reason + fixed JSON, not verbose chain-of-thought.

Produces training-ready records in two flavours for ablation:
  - cot   : output = short reasoning + JSON      (B version, more accurate)
  - label : output = JSON only                   (A version, cheapest tokens)

Requires the teacher API. Run on a step5 subset, e.g. selected_1000.json.
"""

from __future__ import annotations

import argparse
import json
import re

from api_client import TeacherAPI
from common import extract_trajectory_from_instruction, load_json, save_json, write_text
import config

_TEACHER_PROMPT = config.REPO_ROOT / "prompts" / "cot_teacher.txt"
_EVAL_PROMPT = config.REPO_ROOT / "prompts" / "baseline_json.txt"
_JUDGE_RE = re.compile(r'"judgment"\s*:\s*"(safe|unsafe)"', re.IGNORECASE)


def taxonomy_hint(rec: dict) -> str:
    parts = [rec.get("risk_source"), rec.get("failure_mode"), rec.get("harm")]
    parts = [p for p in parts if p]
    return "; ".join(parts) if parts else "(none)"


def final_judgment(text: str) -> str | None:
    matches = _JUDGE_RE.findall(text or "")
    return matches[-1].lower() if matches else None


def make_record(rec: dict, trajectory: str, cot_text: str, mode: str) -> dict:
    eval_prompt = _EVAL_PROMPT.read_text(encoding="utf-8")
    instruction = eval_prompt.replace("{trajectory}", trajectory).replace("{tools}", "(see trajectory)")
    if mode == "label":
        output = json.dumps({"judgment": rec["label"]}, ensure_ascii=False)
    else:
        output = cot_text.strip()
    return {"uid": rec["uid"], "instruction": instruction, "input": "", "output": output,
            "label": rec["label"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget CoT augmentation")
    parser.add_argument("--input", default=None, help="a step5 subset, e.g. selected_1000.json")
    parser.add_argument("--size", default="1000", help="which selected_<size>.json to use")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config.ensure_dirs()
    api = TeacherAPI()
    teacher_tmpl = _TEACHER_PROMPT.read_text(encoding="utf-8")
    in_path = args.input or (config.STEP_DIRS["step5"] / f"selected_{args.size}.json")
    pool = load_json(in_path)
    if args.limit:
        pool = pool[: args.limit]
    print(f"CoT-augmenting {len(pool)} samples via {api.model}")

    cot_records, label_records = [], []
    dropped = 0
    for i, rec in enumerate(pool):
        traj = extract_trajectory_from_instruction(rec["instruction"])
        prompt = (
            teacher_tmpl.replace("{gold}", rec["label"])
            .replace("{taxonomy}", taxonomy_hint(rec))
            .replace("{trajectory}", traj)
        )
        cot_text = None
        for _ in range(2):  # generate + one retry on verdict mismatch
            out = api.chat([{"role": "user", "content": prompt}], max_tokens=220)
            if final_judgment(out) == rec["label"]:
                cot_text = out
                break
        if cot_text is None:
            dropped += 1
            continue
        cot_records.append(make_record(rec, traj, cot_text, "cot"))
        label_records.append(make_record(rec, traj, cot_text, "label"))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pool)}  dropped={dropped}")

    save_json(cot_records, config.STEP_DIRS["step4"] / f"train_cot_{args.size}.json")
    save_json(label_records, config.STEP_DIRS["step4"] / f"train_label_{args.size}.json")
    report = [
        "# Step 4 - CoT Augmentation Report",
        "",
        f"- input subset: {in_path.name if hasattr(in_path, 'name') else in_path}",
        f"- augmented: **{len(cot_records)}**",
        f"- dropped (verdict never matched gold after retry): **{dropped}**",
        "",
        "Outputs (ready for LLaMA-Factory, alpaca format):",
        f"- train_cot_{args.size}.json   (reasoning + JSON, B version)",
        f"- train_label_{args.size}.json (JSON only, A version)",
        "",
    ]
    write_text("\n".join(report), config.STEP_DIRS["step4"] / "cot_report.md")
    print("\n".join(report))


if __name__ == "__main__":
    main()
