#!/usr/bin/env python3
"""Run Qwen3.5-0.8B + LoRA adapter inference for AgentDoG-Lite."""

from __future__ import annotations

import argparse
import json
import shutil
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
from prepare_agentdog_data import CLASSIFICATION_INSTRUCTION
from prepare_sft1k_selected_data import RICH_CLASSIFICATION_INSTRUCTION


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LANGUAGE_MODEL_KEY_SEGMENT = "base_model.model.model.language_model.layers."
CAUSAL_LM_KEY_SEGMENT = "base_model.model.model.layers."


def build_sft_prompt(record: dict[str, Any]) -> str:
    trajectory = build_prompt(
        "{trajectory}\n\nAvailable tools:\n{tools}",
        record,
    )
    return f"{CLASSIFICATION_INSTRUCTION}\n\n{trajectory}".strip()


def build_rich_prompt(record: dict[str, Any]) -> str:
    trajectory = build_prompt(
        "{trajectory}\n\nAvailable tools:\n{tools}",
        record,
    )
    return f"{RICH_CLASSIFICATION_INSTRUCTION}\n\n<TRAJECTORY>\n{trajectory}\n</TRAJECTORY>".strip()


def remap_adapter_key_for_causal_lm(key: str) -> str:
    return key.replace(LANGUAGE_MODEL_KEY_SEGMENT, CAUSAL_LM_KEY_SEGMENT, 1)


def maybe_prepare_causal_lm_adapter(adapter: str, model_loader: str) -> str:
    if not adapter or model_loader != "causal-lm":
        return adapter

    adapter_path = Path(adapter)
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.exists():
        return adapter

    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        return adapter

    state = load_file(weights_path, device="cpu")
    if not any(LANGUAGE_MODEL_KEY_SEGMENT in key for key in state):
        return adapter

    cache_dir = adapter_path / "causal_lm_adapter"
    cache_weights = cache_dir / "adapter_model.safetensors"
    if cache_weights.exists():
        return str(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in adapter_path.iterdir():
        if item.is_file() and item.name != "adapter_model.safetensors":
            shutil.copy2(item, cache_dir / item.name)

    remapped = {remap_adapter_key_for_causal_lm(key): value for key, value in state.items()}
    save_file(remapped, cache_weights)
    return str(cache_dir)


def apply_chat_prompt(tokenizer: Any, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_generation_prompt(
    tokenizer: Any,
    record: dict[str, Any],
    template: str,
    prompt_style: str,
) -> str:
    if prompt_style == "sft":
        return apply_chat_prompt(tokenizer, build_sft_prompt(record))
    if prompt_style == "rich":
        return apply_chat_prompt(tokenizer, build_rich_prompt(record))
    return build_prompt(template, record)


def load_base_model(args: argparse.Namespace, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    model_cls = (
        AutoModelForImageTextToText
        if args.model_loader == "image-text-to-text"
        else AutoModelForCausalLM
    )
    return model_cls.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )


def run_generation(args: argparse.Namespace, records: list[dict[str, Any]], template: str) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = load_base_model(args, torch)
    if args.adapter:
        from peft import PeftModel

        adapter_path = maybe_prepare_causal_lm_adapter(args.adapter, args.model_loader)
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()

    selected_records = records[: args.limit] if args.limit else records
    rows: list[dict[str, Any]] = []
    progress = ProgressReporter(
        total=len(selected_records),
        enabled=args.progress != "none",
    )
    completed = 0

    for batch in iter_batches(selected_records, args.batch_size):
        prompts = [
            build_generation_prompt(tokenizer, record, template, args.prompt_style)
            for record in batch
        ]
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
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads during inference. Defaults to cached files only.",
    )
    parser.add_argument(
        "--model-loader",
        choices=["image-text-to-text", "causal-lm"],
        default="causal-lm",
        help="Use text CausalLM by default. LLaMA-Factory adapters are remapped when needed.",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help="LoRA adapter directory. Omit to evaluate the base model with the same prompt.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, type=Path)
    parser.add_argument(
        "--prompt-style",
        choices=["sft", "rich", "baseline"],
        default="sft",
        help="Use SFT or rich training-style chat prompts; baseline keeps the old prompt file.",
    )
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
            "model_loader": args.model_loader,
            "adapter": args.adapter,
            "input": str(args.input),
            "output": str(args.output),
            "prompt": str(args.prompt),
            "prompt_style": args.prompt_style,
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
