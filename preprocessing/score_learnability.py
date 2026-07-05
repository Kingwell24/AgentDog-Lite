#!/usr/bin/env python3
"""Score learnability with the Qwen3.5-0.8B baseline (GPU server only).

A gradient-free proxy for the influence-function purification signal in the
AgentDoG 1.5 paper (§3.2.2). For each pool sample we read the baseline model's
first-token preference between "safe" and "unsafe", then score how *informative*
the sample is for fine-tuning:

    learnability = 0.5 * uncertainty + 0.5 * error
    uncertainty  = 1 - |2p - 1|          (peaks when the model is on the fence)
    error        = 1 - correct_prob      (high when the baseline is wrong)

Intuition: samples the baseline already nails are redundant (low score); samples
right at the decision boundary are where LoRA can actually move the model (high
score). Output feeds step5's selection as `scores/learnability.json`.

This performs real model inference, so per AGENTS.md it must run on the GPU
server. It refuses to run locally unless --allow-local-run is passed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from common import extract_trajectory_from_instruction, load_json, save_json
import config

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = config.REPO_ROOT / "prompts" / "baseline_json.txt"


def should_block(allow_local_run: bool, cuda_available: bool) -> bool:
    if allow_local_run:
        return False
    return not cuda_available  # block CPU-only local machines


def build_prompt(template: str, trajectory: str) -> str:
    return template.replace("{trajectory}", trajectory).replace("{tools}", "(see trajectory)")


def candidate_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids whose decoded text starts with safe / unsafe."""
    safe_ids, unsafe_ids = [], []
    for variant in ["safe", " safe", "Safe", " Safe", '"safe', ' "safe']:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            safe_ids.append(ids[0])
    for variant in ["unsafe", " unsafe", "Unsafe", " Unsafe", '"unsafe', ' "unsafe']:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            unsafe_ids.append(ids[0])
    return sorted(set(safe_ids)), sorted(set(unsafe_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Learnability scoring (GPU)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--input", default=None, help="defaults to step2 valid_pool.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=6144,
                        help="prompt truncation length in tokens (8GB GPUs: keep <=6144;"
                             " attention memory grows with length and OOMs on long tails)")
    parser.add_argument("--missing-only", action="store_true",
                        help="only score uids absent from existing learnability.json"
                             " and merge results into it")
    parser.add_argument("--allow-local-run", action="store_true")
    args = parser.parse_args()

    import os

    # avoid fragmentation OOM on small (8GB) GPUs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    if should_block(args.allow_local_run, torch.cuda.is_available()):
        sys.exit(
            "Refusing to run inference on a non-GPU/local machine (see AGENTS.md).\n"
            "Run on the GPU server, or pass --allow-local-run to override."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    config.ensure_dirs()
    in_path = args.input or (config.STEP_DIRS["step2"] / "valid_pool.json")
    pool = load_json(in_path)
    existing: dict[str, float] = {}
    if args.missing_only:
        score_path = config.STEP_DIRS["scores"] / "learnability.json"
        if score_path.exists():
            existing = load_json(score_path)
        pool = [r for r in pool if r["uid"] not in existing]
        print(f"missing-only: {len(existing)} already scored, {len(pool)} to score")
    if args.limit:
        pool = pool[: args.limit]
    template = Path(args.prompt).read_text(encoding="utf-8")
    print(f"Scoring {len(pool)} samples with {args.model} on {platform.node()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = "bfloat16" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    safe_ids, unsafe_ids = candidate_token_ids(tokenizer)

    def checkpoint(scores: dict[str, float]) -> None:
        merged = {**existing, **scores} if existing else dict(scores)
        save_json(merged, config.STEP_DIRS["scores"] / "learnability.json")

    import time

    t0 = time.time()
    learn: dict[str, float] = {}
    detail = []
    for i, rec in enumerate(pool):
        traj = extract_trajectory_from_instruction(rec["instruction"])
        prompt = build_prompt(template, traj)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logits = None
        for attempt_len in (args.max_length, args.max_length // 2, args.max_length // 4):
            inp = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=attempt_len).to(model.device)
            try:
                with torch.no_grad():
                    try:
                        # only materialize logits for the final position: turns a
                        # [seq_len, vocab] tensor (GBs on long prompts) into [1, vocab]
                        logits = model(**inp, logits_to_keep=1).logits[0, -1]
                    except TypeError:  # older transformers without logits_to_keep
                        logits = model(**inp).logits[0, -1]
                break
            except torch.OutOfMemoryError:
                # long-tail sample blew attention memory: shrink and retry
                torch.cuda.empty_cache()
                print(f"  [OOM] {rec['uid']} at len<={attempt_len}, retrying shorter...",
                      flush=True)
        if logits is None:
            # unscoreable on this GPU: neutral score, keep going
            learn[rec["uid"]] = 0.5
            detail.append({"uid": rec["uid"], "label": rec["label"], "p_unsafe": None,
                           "correct_prob": None, "learnability": 0.5, "oom": True})
            continue
        probs = torch.softmax(logits.float(), dim=-1)
        p_safe = float(probs[safe_ids].sum())
        p_unsafe = float(probs[unsafe_ids].sum())
        denom = p_safe + p_unsafe or 1e-9
        p = p_unsafe / denom  # P(unsafe | safe or unsafe)
        gold_unsafe = rec["label"] == "unsafe"
        correct_prob = p if gold_unsafe else (1 - p)
        uncertainty = 1 - abs(2 * p - 1)
        error = 1 - correct_prob
        score = 0.5 * uncertainty + 0.5 * error
        learn[rec["uid"]] = round(score, 4)
        detail.append(
            {"uid": rec["uid"], "label": rec["label"], "p_unsafe": round(p, 4),
             "correct_prob": round(correct_prob, 4), "learnability": round(score, 4)}
        )
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (len(pool) - i - 1) / rate / 60
            print(f"  {i + 1}/{len(pool)}  ({rate:.1f}/s, 剩余约{eta_min:.0f}分钟)",
                  flush=True)
        if (i + 1) % 200 == 0:
            checkpoint(learn)  # 断点存档: 中断最多丢200条
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    checkpoint(learn)
    with open(config.STEP_DIRS["scores"] / "learnability_detail.jsonl", "w", encoding="utf-8") as fh:
        for row in detail:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote scores/learnability.json ({len(learn)} rows)")


if __name__ == "__main__":
    main()
