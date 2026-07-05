#!/usr/bin/env python3
"""Build A-version (label-only) training files locally — no API required.

step4 couples the label-only files to the CoT API loop, but the A version
(output = {"judgment": ...}) needs nothing from the teacher. This script
converts any step5 subset into LLaMA-Factory-ready alpaca records today:

  instruction = our evaluation prompt (prompts/baseline_json.txt) + trajectory
  output      = {"judgment": "safe"} / {"judgment": "unsafe"}

Using the SAME prompt for training and evaluation keeps the train/infer format
identical, which is what stabilizes JSON output and kills invalid_rate.

Local-safe: pure text.
"""

from __future__ import annotations

import argparse
import json

from common import extract_trajectory_from_instruction, load_json, save_json
import config

_EVAL_PROMPT = config.REPO_ROOT / "prompts" / "baseline_json.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local A-version training files")
    parser.add_argument("--sizes", default="500,1000,2000,full")
    args = parser.parse_args()

    config.ensure_dirs()
    template = _EVAL_PROMPT.read_text(encoding="utf-8")
    out_dir = config.STEP_DIRS["step4"]

    for token in args.sizes.split(","):
        token = token.strip()
        in_path = config.STEP_DIRS["step5"] / f"selected_{token}.json"
        if not in_path.exists():
            print(f"  skip {token}: {in_path.name} not found")
            continue
        subset = load_json(in_path)
        records = []
        for rec in subset:
            traj = extract_trajectory_from_instruction(rec["instruction"])
            instruction = template.replace("{trajectory}", traj).replace(
                "{tools}", "(see trajectory)"
            )
            records.append(
                {
                    "uid": rec["uid"],
                    "instruction": instruction,
                    "input": "",
                    "output": json.dumps({"judgment": rec["label"]}, ensure_ascii=False),
                    "label": rec["label"],
                }
            )
        out_path = out_dir / f"train_label_{token}.json"
        save_json(records, out_path)
        print(f"  wrote {out_path.name}: {len(records)} rows")


if __name__ == "__main__":
    main()
