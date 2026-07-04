#!/usr/bin/env python3
"""Run the Qwen3.5-0.8B baseline for AgentDoG-Lite trajectory safety data.

This script is intended to be executed on a GPU server. The local workstation is
only for code editing and light validation unless --allow-local-run is explicit.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, TextIO


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = Path("prompts/baseline_json.txt")
VALID_LABELS = {"safe": 0, "unsafe": 1}


class ProgressReporter:
    def __init__(
        self,
        total: int,
        enabled: bool = True,
        stream: TextIO | None = None,
        width: int = 28,
    ) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self._printed = False

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        done = min(max(done, 0), self.total)
        filled = int(self.width * done / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100 * done / self.total
        self.stream.write(f"\rProgress [{bar}] {done}/{self.total} ({percent:5.1f}%)")
        self.stream.flush()
        self._printed = True

    def finish(self) -> None:
        if self.enabled and self._printed:
            self.stream.write("\n")
            self.stream.flush()


def should_block_inference(
    allow_local_run: bool,
    platform_name: str | None = None,
    cuda_available: bool | None = None,
) -> bool:
    if allow_local_run:
        return False
    if platform_name is None:
        platform_name = platform.system()
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    return platform_name.lower() != "linux" or not cuda_available


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index} is not an object")
        records.append(item)
    return records


def compact_json(value: Any, max_chars: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def format_tools(record: dict[str, Any], max_tool_chars: int = 8000) -> str:
    tools = record.get("tool_used") or record.get("tools") or []
    if not tools:
        return "(none provided)"

    lines: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "(unnamed)")
            description = str(tool.get("description", "")).strip()
            params = tool.get("parameters", {})
            lines.append(f"- {name}: {description}\n  parameters: {compact_json(params, 900)}")
        else:
            lines.append(f"- {tool}")

    text = "\n".join(lines)
    if len(text) <= max_tool_chars:
        return text
    return text[:max_tool_chars] + "\n...[tools truncated]"


def iter_turns(contents: Any) -> list[dict[str, Any]]:
    if not isinstance(contents, list):
        return []
    if contents and all(isinstance(item, dict) for item in contents):
        return contents

    turns: list[dict[str, Any]] = []
    for block in contents:
        if isinstance(block, list):
            turns.extend(item for item in block if isinstance(item, dict))
        elif isinstance(block, dict):
            turns.append(block)
    return turns


def format_trajectory(record: dict[str, Any], max_chars: int = 24000) -> str:
    parts: list[str] = []
    if record.get("profile"):
        parts.append(f"[PROFILE] {record['profile']}")
    if record.get("goal"):
        parts.append(f"[TASK] {record['goal']}")

    for turn in iter_turns(record.get("contents")):
        role = str(turn.get("role", "unknown")).upper()
        if turn.get("content") is not None:
            parts.append(f"[{role}] {turn.get('content')}")
        if turn.get("thought"):
            parts.append(f"[{role}_THOUGHT] {turn.get('thought')}")
        if turn.get("action"):
            parts.append(f"[{role}_ACTION] {turn.get('action')}")

    text = "\n".join(parts).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 3]
    tail = text[-(max_chars - len(head) - 40) :]
    return head + "\n...[middle truncated deterministically]...\n" + tail


def build_prompt(template: str, record: dict[str, Any]) -> str:
    return (
        template.replace("{trajectory}", format_trajectory(record))
        .replace("{tools}", format_tools(record))
    )


def parse_judgment(text: str) -> str | None:
    if not text:
        return None

    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            value = str(obj.get("judgment", "")).strip().lower()
            if value in VALID_LABELS:
                return value
        except Exception:
            pass

    tag_match = re.search(r"<Judgment>\s*(safe|unsafe)\s*</Judgment>", text, flags=re.I)
    if tag_match:
        return tag_match.group(1).lower()

    labels = re.findall(r"\b(unsafe|safe)\b", text, flags=re.I)
    if len(labels) == 1:
        return labels[0].lower()
    return None


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    invalid = sum(1 for row in rows if row.get("prediction") not in VALID_LABELS)
    tp = fp = tn = fn = 0

    for row in rows:
        label = row.get("label")
        prediction = row.get("prediction")
        pred_int = VALID_LABELS.get(prediction)
        if label not in (0, 1):
            continue
        if pred_int is None:
            if label == 1:
                fn += 1
            continue
        if pred_int == 1 and label == 1:
            tp += 1
        elif pred_int == 1 and label == 0:
            fp += 1
        elif pred_int == 0 and label == 0:
            tn += 1
        elif pred_int == 0 and label == 1:
            fn += 1

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    avg_output_tokens = (
        sum(int(row.get("output_tokens") or 0) for row in rows) / total if total else 0.0
    )

    return {
        "total": total,
        "invalid": invalid,
        "invalid_rate": invalid / total if total else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_output_tokens": avg_output_tokens,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def run_generation(args: argparse.Namespace, records: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    selected_records = records[: args.limit] if args.limit else records
    progress = ProgressReporter(
        total=len(selected_records),
        enabled=getattr(args, "progress", "bar") != "none",
    )
    for index, record in enumerate(selected_records, start=1):
        prompt = build_prompt(template, record)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        prediction = parse_judgment(raw_output)
        row = {
            "id": record.get("id", index - 1),
            "label": record.get("label"),
            "prediction": prediction,
            "prediction_int": VALID_LABELS.get(prediction),
            "raw_output": raw_output,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(generated_ids.shape[-1]),
        }
        rows.append(row)
        progress.update(index)
    progress.finish()
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSON array dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for predictions and metrics.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local model path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test sample count.")
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--progress",
        choices=["bar", "none"],
        default="bar",
        help="Terminal progress display. Per-sample details are always saved to predictions.jsonl.",
    )
    parser.add_argument(
        "--allow-local-run",
        action="store_true",
        help="Override the workstation guard. Use only if you intentionally want local inference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if should_block_inference(args.allow_local_run):
        print(
            "Refusing to run real Qwen3.5-0.8B inference here. This repository rule is strict: "
            "the local machine is only for code editing/light validation; run model inference on the GPU server. "
            "Pass --allow-local-run only for an intentional override.",
            file=sys.stderr,
        )
        return 2

    records = load_json_records(args.input)
    template = args.prompt.read_text(encoding="utf-8")
    rows = run_generation(args, records, template)
    metrics = compute_metrics(rows)

    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model": args.model,
            "input": str(args.input),
            "prompt": str(args.prompt),
            "limit": args.limit,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(f"Saved predictions to {args.output_dir / 'predictions.jsonl'}")
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
