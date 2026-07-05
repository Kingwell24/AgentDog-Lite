#!/usr/bin/env python3
"""Step 0 - Inventory & merge.

Answers the first factual question the pipeline branches on:
  Are BinarySafety (4000) and FineGrainedTaxonomy (4000) the SAME trajectories
  with two annotations, or two disjoint batches?

Also builds a single unified pool (`merged_pool.json`) with per-sample metadata
(binary label + taxonomy triple + trajectory hash + length) that every later
step consumes.

Local-safe: pure text processing, no model, no API.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

from common import (
    extract_trajectory_from_instruction,
    load_json,
    normalize_light,
    save_json,
    sha1,
    write_text,
)
import config

_RISK_RE = re.compile(r"Risk Source:\s*(.+)")
_MODE_RE = re.compile(r"Failure Mode:\s*(.+)")
_HARM_RE = re.compile(r"Real[- ]?World Harm:\s*(.+)")
_BENIGN = {"benign", "none", "n/a", "safe", ""}


def parse_taxonomy(output: str) -> dict[str, str | None]:
    def grab(rx: re.Pattern[str]) -> str | None:
        m = rx.search(output or "")
        return m.group(1).strip() if m else None

    return {
        "risk_source": grab(_RISK_RE),
        "failure_mode": grab(_MODE_RE),
        "harm": grab(_HARM_RE),
    }


def binary_label(source: str, output: str, taxonomy: dict[str, str | None]) -> str:
    """Derive a safe/unsafe label regardless of coarse/fine source."""
    text = (output or "").strip().lower()
    if source == "BinarySafety":
        return "unsafe" if "unsafe" in text else "safe"
    # FineGrained: a real (non-benign) risk source => unsafe
    rs = (taxonomy.get("risk_source") or "").strip().lower()
    return "safe" if rs in _BENIGN else "unsafe"


def build_records(samples: list[dict], source: str) -> list[dict]:
    records = []
    for idx, s in enumerate(samples):
        instruction = s.get("instruction", "")
        traj = extract_trajectory_from_instruction(instruction)
        taxonomy = parse_taxonomy(s.get("output", "")) if source == "FineGrained" else {
            "risk_source": None,
            "failure_mode": None,
            "harm": None,
        }
        records.append(
            {
                "uid": f"{'bin' if source == 'BinarySafety' else 'fg'}-{idx}",
                "source_file": source,
                "orig_id": s.get("id", idx),
                "instruction": instruction,
                "output": s.get("output", ""),
                "label": binary_label(source, s.get("output", ""), taxonomy),
                **taxonomy,
                "traj_chars": len(traj),
                "traj_hash": sha1(normalize_light(traj)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory & merge the 1.0 pool")
    parser.add_argument("--near-threshold", type=float, default=0.9,
                        help="not used here; kept for symmetry with step1")
    parser.parse_args()

    config.ensure_dirs()
    out_dir = config.STEP_DIRS["step0"]

    binary = load_json(config.BINARY_FILE)
    fine = load_json(config.FINEGRAINED_FILE)
    print(f"Loaded BinarySafety={len(binary)}  FineGrained={len(fine)}")

    bin_records = build_records(binary, "BinarySafety")
    fg_records = build_records(fine, "FineGrained")
    pool = bin_records + fg_records

    # --- Overlap: same trajectories across the two files? -------------------
    bin_hashes = {r["traj_hash"] for r in bin_records}
    fg_hashes = {r["traj_hash"] for r in fg_records}
    exact_overlap = bin_hashes & fg_hashes
    overlap_ratio = len(exact_overlap) / max(len(bin_hashes), 1)
    same_batch = overlap_ratio > 0.5

    # --- Distributions ------------------------------------------------------
    def label_dist(records: list[dict]) -> dict[str, int]:
        return dict(Counter(r["label"] for r in records))

    fg_risk = Counter(r["risk_source"] for r in fg_records if r["risk_source"])

    # --- Report -------------------------------------------------------------
    lines = [
        "# Step 0 - Inventory & Merge Report",
        "",
        f"- BinarySafety rows: **{len(bin_records)}**  (labels: {label_dist(bin_records)})",
        f"- FineGrained rows: **{len(fg_records)}**  (labels: {label_dist(fg_records)})",
        f"- Total pool rows: **{len(pool)}**",
        "",
        "## Same-batch check (the branch decision)",
        f"- Unique trajectory hashes: binary={len(bin_hashes)}, fine={len(fg_hashes)}",
        f"- Exact trajectory overlap (binary ∩ fine): **{len(exact_overlap)}**"
        f" ({overlap_ratio:.1%} of binary)",
        "",
        f"**Verdict: {'SAME batch (two annotations of the same trajectories)' if same_batch else 'DISJOINT batches (8000 distinct trajectories)'}**",
        "",
        "### What this means for later steps",
    ]
    if same_batch:
        lines += [
            "- Merge on trajectory: each trajectory can carry BOTH the safe/unsafe"
            " label and the 3-dim taxonomy.",
            "- Step 4 (CoT): feed the taxonomy triple to the teacher as an oracle"
            " hint -> more accurate coarse rationales.",
            "- Real distinct trajectories ~= 4000, not 8000.",
        ]
    else:
        lines += [
            "- FineGrained is a separate (mostly unsafe) batch; do NOT dump it into"
            " the binary training mix blindly or the safe:unsafe ratio skews.",
            "- Use FineGrained as an unsafe-diversity source + CoT oracle material.",
        ]
    lines += [
        "",
        "## FineGrained risk-source distribution (top 12)",
        "",
        "| risk_source | count |",
        "| --- | ---: |",
    ]
    for name, count in fg_risk.most_common(12):
        lines.append(f"| {name} | {count} |")
    lines.append("")

    write_text("\n".join(lines), out_dir / "inventory_report.md")
    save_json(pool, out_dir / "merged_pool.json")

    print("\n".join(lines[:16]))
    print(f"\nWrote:\n  {out_dir / 'inventory_report.md'}\n  {out_dir / 'merged_pool.json'}")


if __name__ == "__main__":
    main()
