#!/usr/bin/env python3
"""Summarize LoRA prediction files and adapter contents for debugging."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize_predictions(path: Path, sample_count: int) -> None:
    rows = read_jsonl(path)
    print(f"predictions_file: {path}")
    print(f"rows: {len(rows)}")
    print("labels:", dict(Counter(row.get("label") for row in rows)))
    print("predictions:", dict(Counter(row.get("prediction") for row in rows)))
    print("first_samples:")
    for row in rows[:sample_count]:
        raw = str(row.get("raw_output", "")).replace("\n", "\\n")
        print(
            {
                "id": row.get("id"),
                "label": row.get("label"),
                "prediction": row.get("prediction"),
                "raw_output": raw[:240],
            }
        )


def summarize_sft_data(path: Path, sample_count: int) -> None:
    rows = read_jsonl(path)
    labels = Counter()
    samples = []
    for row in rows:
        output = json.loads(row["output"])
        label = output.get("judgment")
        labels[label] += 1
        if len(samples) < sample_count:
            samples.append(
                {
                    "label": label,
                    "input_head": row.get("input", "").replace("\n", " ")[:240],
                }
            )
    print(f"sft_file: {path}")
    print(f"rows: {len(rows)}")
    print("labels:", dict(labels))
    print("first_samples:")
    for sample in samples:
        print(sample)


def summarize_adapter(path: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError:
        print("safetensors is not installed; skipping adapter key summary")
        return

    keys = sorted(load_file(path, device="cpu").keys())
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "out_proj",
        "in_proj_a",
        "in_proj_b",
        "in_proj_z",
    ]
    modules = Counter()
    for key in keys:
        for target in targets:
            if f".{target}." in key:
                modules[target] += 1

    print(f"adapter_file: {path}")
    print(f"adapter_key_count: {len(keys)}")
    print("module_key_counts:", dict(modules))
    print("first_adapter_keys:")
    for key in keys[:40]:
        print(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--sft-data", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--sample-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.predictions:
        summarize_predictions(args.predictions, args.sample_count)
    if args.sft_data:
        summarize_sft_data(args.sft_data, args.sample_count)
    if args.adapter:
        summarize_adapter(args.adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
