#!/usr/bin/env python3
"""Run Qwen3.5-0.8B + LoRA adapter inference for AgentDoG-Lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_baseline_qwen35_08b import (
    DEFAULT_PROMPT,
    VALID_LABELS,
    ProgressReporter,
    build_prompt,
    iter_batches,
    load_json_records,
    parse_judgment,
    should_block_inference,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"


def run_generation(args: argparse.Namespace, records: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    selected_records = records[: args.limit] if args.limit else records
    rows: list[dict[str, Any]] = []
    progress = ProgressReporter(
        total=len(selected_records),
        enabled=args.progress != "none",
    )
    completed = 0

    for batch in iter_batches(selected_records, args.batch_size):
        prompts = [build_prompt(template, record) for record in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
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

        prompt_width = inputs["input_ids"].shape[-1]
        for batch_index, record in enumerate(batch):
            generated_ids = output_ids[batch_index][prompt_width:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            prediction = parse_judgment(raw_output)
            rows.append(
                {
                    "id": record.get("id", completed + batch_index),
                    "label": record.get("label"),
                    "prediction": prediction,
                    "prediction_int": VALID_LABELS.get(prediction),
                    "raw_output": raw_output,
                    "input_tokens": int(inputs["attention_mask"][batch_index].sum().item()),
                    "output_tokens": int(generated_ids.shape[-1]),
                }
            )

        completed += len(batch)
        progress.update(completed)
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
    parser.add_argument("--input", required=True, type=Path, help="Input JSON array test set.")
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSONL path.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--progress", choices=["bar", "none"], default="bar")
    parser.add_argument(
        "--allow-local-run",
        action="store_true",
        help="Override the workstation guard. Use only for intentional local inference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if should_block_inference(args.allow_local_run):
        print(
            "Refusing to run real model inference here. Run this script on the GPU server, "
            "or pass --allow-local-run only for an intentional override.",
            file=sys.stderr,
        )
        return 2

    records = load_json_records(args.input)
    template = args.prompt.read_text(encoding="utf-8")
    rows = run_generation(args, records, template)
    write_jsonl(args.output, rows)
    write_json(
        args.output.with_name("run_config.json"),
        {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "input": str(args.input),
            "output": str(args.output),
            "prompt": str(args.prompt),
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(f"Saved predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
