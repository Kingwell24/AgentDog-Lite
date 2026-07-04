#!/usr/bin/env python3
"""Generic Qwen3.5-0.8B inference smoke/full test for AgentDoG-Lite data.

Use this on the GPU server. It supports either local JSON files or raw JSON
files downloaded from a Hugging Face dataset repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


BASELINE_PATH = Path(__file__).with_name("run_baseline_qwen35_08b.py")
SPEC = importlib.util.spec_from_file_location("baseline", BASELINE_PATH)
baseline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline)


DEFAULT_HF_REPO = "AI45Research/2026_summer_camp_teseset"
DEFAULT_HF_FILES = [
    "summer_camp_ATBench300.json",
    "summer_camp_rjudge.json",
]
DEFAULT_PROMPT = Path("prompts/qwen35_08b_infer_json.txt")


def stem_for_output(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_hf_files(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_HF_FILES)
    files: list[str] = []
    for value in values:
        files.extend(part.strip() for part in value.split(",") if part.strip())
    return files


def resolve_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(item) for item in args.input]

    if args.source != "hf":
        raise ValueError("--input is required when --source local")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for --source hf. Install with: "
            "python -m pip install -U huggingface_hub"
        ) from exc

    files = parse_hf_files(args.hf_file)
    downloaded: list[Path] = []
    for filename in files:
        local_path = hf_hub_download(
            repo_id=args.hf_repo,
            repo_type="dataset",
            filename=filename,
            local_dir=args.cache_dir,
        )
        downloaded.append(Path(local_path))
    return downloaded


def run_one_file(
    args: argparse.Namespace,
    input_path: Path,
    template: str,
) -> dict[str, Any]:
    records = baseline.load_json_records(input_path)
    rows = baseline.run_generation(args, records, template)
    metrics = baseline.compute_metrics(rows)

    output_dir = args.output_dir / stem_for_output(input_path)
    baseline.write_jsonl(output_dir / "predictions.jsonl", rows)
    baseline.write_json(output_dir / "metrics.json", metrics)
    baseline.write_json(
        output_dir / "run_config.json",
        {
            "model": args.model,
            "source": args.source,
            "hf_repo": args.hf_repo,
            "input": str(input_path),
            "prompt": str(args.prompt),
            "limit": args.limit,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    if args.print_raw:
        for row in rows:
            print(f"\n--- RAW id={row.get('id')} label={row.get('label')} ---")
            print(row.get("raw_output", ""))
    return {"input": str(input_path), "output_dir": str(output_dir), "metrics": metrics}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["hf", "local"],
        default="hf",
        help="Use Hugging Face raw JSON files or local JSON files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Local JSON file. May be repeated. Required for --source local.",
    )
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument(
        "--hf-file",
        action="append",
        help="HF dataset file name. May be repeated or comma-separated.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/2026_summer_camp_teseset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35_08b_infer_test"))
    parser.add_argument("--model", default=baseline.DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--progress",
        choices=["bar", "none"],
        default="bar",
        help="Terminal progress display. Per-sample details are saved to predictions.jsonl.",
    )
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument(
        "--allow-local-run",
        action="store_true",
        help="Override local-machine inference guard. Do not use on the coding workstation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if baseline.should_block_inference(args.allow_local_run):
        print(
            "Refusing to run real Qwen3.5-0.8B inference here. Run this on the GPU server.",
            file=sys.stderr,
        )
        return 2

    template = args.prompt.read_text(encoding="utf-8")
    input_files = resolve_input_files(args)

    summaries = []
    for input_path in input_files:
        print(f"\n=== Running {input_path} ===", flush=True)
        summaries.append(run_one_file(args, input_path, template))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
