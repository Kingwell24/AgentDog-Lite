#!/usr/bin/env python3
"""Semantic (embedding-level) leakage check between the training pool and the
held-out test sets — the second line of defense after step1's lexical MinHash.

MinHash catches near-verbatim template clones; it misses paraphrased or
re-serialized duplicates. Here we embed every trajectory with the base model's
own encoder (mean-pooled last hidden state of Qwen3.5-0.8B — no extra model
download) and flag any train/test pair whose cosine similarity exceeds
--threshold. Flagged uids can then be removed from the pool via --apply.

GPU strongly recommended (per AGENTS.md this is a server job; the local-GPU
override is a human decision).

Outputs:
  scores/semantic_leakage.json     [{uid, test_name, test_idx, cosine}]
  step1/semantic_leak_report.md
  (with --apply) step2/valid_pool.json is rewritten without flagged uids.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import (
    extract_trajectory_from_instruction,
    load_json,
    normalize_light,
    save_json,
    test_sample_to_text,
    write_text,
)
import config

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
MAX_TOKENS = 2048  # truncate trajectories for embedding; heads carry the scenario


def embed_texts(texts: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_TOKENS).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        chunks.append(pooled.cpu().numpy())
        done = min(start + batch_size, len(texts))
        if done % (batch_size * 10) < batch_size:
            print(f"  embedded {done}/{len(texts)}")
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding-level leakage check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="cosine >= this flags a semantic duplicate")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--apply", action="store_true",
                        help="also remove flagged uids from step2 valid_pool.json")
    parser.add_argument("--allow-local-run", action="store_true")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_local_run:
        sys.exit("No CUDA device. Run on the GPU server or pass --allow-local-run (slow).")

    config.ensure_dirs()
    pool = load_json(config.STEP_DIRS["step2"] / "valid_pool.json")
    pool_texts = [normalize_light(extract_trajectory_from_instruction(r["instruction"]))
                  for r in pool]

    test_texts: list[str] = []
    test_meta: list[tuple[str, int]] = []
    for name, path in config.TEST_FILES.items():
        data = load_json(path)
        for idx, sample in enumerate(data):
            test_texts.append(normalize_light(test_sample_to_text(sample)))
            test_meta.append((name, idx))
    print(f"Embedding {len(pool_texts)} train + {len(test_texts)} test on {device}")

    train_emb = embed_texts(pool_texts, args.model, args.batch_size, device)
    test_emb = embed_texts(test_texts, args.model, args.batch_size, device)

    sims = train_emb @ test_emb.T  # (train, test) cosine matrix
    best_j = sims.argmax(axis=1)
    best_s = sims[np.arange(len(pool)), best_j]

    flagged = []
    for i in np.where(best_s >= args.threshold)[0]:
        name, idx = test_meta[int(best_j[i])]
        flagged.append({"uid": pool[i]["uid"], "test_name": name, "test_idx": idx,
                        "cosine": round(float(best_s[i]), 4)})
    flagged.sort(key=lambda r: -r["cosine"])
    save_json(flagged, config.STEP_DIRS["scores"] / "semantic_leakage.json")

    report = [
        "# Semantic Leakage Report (embedding-level)",
        "",
        f"- encoder: {args.model} (mean-pooled last hidden state)",
        f"- threshold: cosine >= {args.threshold}",
        f"- train pool: {len(pool)}, test: {len(test_texts)}",
        f"- flagged pairs: **{len(flagged)}**",
        f"- similarity distribution (train->nearest test): "
        f"p50={np.percentile(best_s,50):.3f} p95={np.percentile(best_s,95):.3f} "
        f"p99={np.percentile(best_s,99):.3f} max={best_s.max():.3f}",
        "",
    ]
    if flagged[:20]:
        report += ["| uid | test | cosine |", "| --- | --- | ---: |"]
        report += [f"| {f['uid']} | {f['test_name']}#{f['test_idx']} | {f['cosine']} |"
                   for f in flagged[:20]]
    write_text("\n".join(report), config.STEP_DIRS["step1"] / "semantic_leak_report.md")
    print("\n".join(report[:10]))

    if args.apply and flagged:
        bad = {f["uid"] for f in flagged}
        cleaned = [r for r in pool if r["uid"] not in bad]
        save_json(cleaned, config.STEP_DIRS["step2"] / "valid_pool.json")
        print(f"Applied: valid_pool.json {len(pool)} -> {len(cleaned)} (re-run step5)")


if __name__ == "__main__":
    main()
