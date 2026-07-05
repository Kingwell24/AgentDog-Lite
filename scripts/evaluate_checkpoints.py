#!/usr/bin/env python3
"""Evaluate every LoRA checkpoint in a run directory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "adapter_config.json").exists()
    ]
    return sorted(checkpoints, key=checkpoint_number)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def is_distribution_reasonable(
    metrics: dict[str, Any],
    min_safe_rate: float,
    max_unsafe_rate: float,
) -> bool:
    distribution = metrics.get("prediction_distribution", {})
    safe_rate = float(distribution.get("safe_rate", 0.0))
    unsafe_rate = float(distribution.get("unsafe_rate", 1.0))
    return safe_rate >= min_safe_rate and unsafe_rate <= max_unsafe_rate


def summarize_checkpoint(
    checkpoint: Path,
    metrics: dict[str, Any],
    min_safe_rate: float,
    max_unsafe_rate: float,
) -> dict[str, Any]:
    distribution = metrics.get("prediction_distribution", {})
    confusion = metrics.get("confusion_matrix", {})
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_dir": str(checkpoint),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "invalid_rate": metrics.get("invalid_rate"),
        "pred_safe_rate": distribution.get("safe_rate"),
        "pred_unsafe_rate": distribution.get("unsafe_rate"),
        "tp": confusion.get("tp"),
        "fp": confusion.get("fp"),
        "tn": confusion.get("tn"),
        "fn": confusion.get("fn"),
        "distribution_reasonable": is_distribution_reasonable(
            metrics,
            min_safe_rate=min_safe_rate,
            max_unsafe_rate=max_unsafe_rate,
        ),
    }


def best_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    reasonable = [row for row in rows if row["distribution_reasonable"]]
    candidates = reasonable or rows
    return max(
        candidates,
        key=lambda row: (
            float(row.get("accuracy") or 0.0),
            float(row.get("balanced_accuracy") or 0.0),
            float(row.get("f1") or 0.0),
        ),
    )


def run_self_test() -> int:
    metrics = {
        "prediction_distribution": {"safe_rate": 0.45, "unsafe_rate": 0.55},
        "confusion_matrix": {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "f1": 0.5,
        "invalid_rate": 0.0,
    }
    row = summarize_checkpoint(Path("checkpoint-25"), metrics, 0.30, 0.70)
    assert row["distribution_reasonable"]
    assert checkpoint_number(Path("checkpoint-125")) == 125
    assert best_checkpoint([row]) == row
    print("evaluate_checkpoints self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-style", choices=["sft", "rich", "baseline"], default="sft")
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-safe-rate", type=float, default=0.30)
    parser.add_argument("--max-unsafe-rate", type=float, default=0.70)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        return run_self_test()
    if args.run_dir is None or args.input is None:
        print("--run-dir and --input are required unless --self-test is used", file=sys.stderr)
        return 2

    run_dir = args.run_dir.resolve()
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        print(f"No checkpoint adapters found under {run_dir}", file=sys.stderr)
        return 1

    summary_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        eval_dir = run_dir / "eval_checkpoints" / args.input.stem / checkpoint.name
        predictions = eval_dir / "predictions.jsonl"
        metrics_output = eval_dir / "metrics.json"

        if not (args.skip_existing and metrics_output.exists()):
            run_command(
                [
                    sys.executable,
                    "scripts/run_inference.py",
                    "--base-model",
                    args.base_model,
                    "--adapter",
                    str(checkpoint),
                    "--input",
                    str(args.input),
                    "--output",
                    str(predictions),
                    "--prompt-style",
                    args.prompt_style,
                    "--batch-size",
                    str(args.batch_size),
                    "--max-input-tokens",
                    str(args.max_input_tokens),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--progress",
                    "bar",
                ]
                + (["--limit", str(args.limit)] if args.limit else [])
            )
            run_command(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--predictions",
                    str(predictions),
                    "--metrics-output",
                    str(metrics_output),
                ]
            )

        metrics = read_json(metrics_output)
        summary_rows.append(
            summarize_checkpoint(
                checkpoint,
                metrics,
                min_safe_rate=args.min_safe_rate,
                max_unsafe_rate=args.max_unsafe_rate,
            )
        )

    summary = {
        "run_dir": str(run_dir),
        "input": str(args.input),
        "min_safe_rate": args.min_safe_rate,
        "max_unsafe_rate": args.max_unsafe_rate,
        "checkpoints": summary_rows,
        "best_checkpoint": best_checkpoint(summary_rows),
    }
    summary_output = args.summary_output or (run_dir / "eval_checkpoints" / "summary.json")
    write_json(summary_output, summary)
    print(f"Saved checkpoint summary to {summary_output}")
    print(json.dumps(summary["best_checkpoint"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
