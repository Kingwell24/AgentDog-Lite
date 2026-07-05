#!/usr/bin/env python3
"""Step 5 - Scored selection & compression to ~1k.

Combines up to four signals into a per-sample score, then selects a compact,
balanced training subset under hard quotas. Signals that require a model/API are
OPTIONAL: if their score file is absent, that signal is treated as neutral, so
this step runs end-to-end locally today and simply gets sharper once the GPU
`learnability` and API `confidence` scores are dropped into `outputs/.../scores/`.

Score:   S(z) = w_learn*learnability + w_rare*rarity + w_conf*confidence
Quotas:  safe:unsafe ~ 1:1  (capped by availability)
         within safe: >= --safe-risk-frac are "risk present but handled"
         within unsafe: soft-balanced across risk_source with a per-class floor

Emits one subset per size in --sizes so the training teammate can plot the
"data size vs performance/token" curve.

Local-safe: numpy only.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import numpy as np

from common import (
    extract_trajectory_from_instruction,
    load_json,
    save_json,
    write_text,
)
import config

# cue words that suggest a risk was actually present in the trajectory
_RISK_CUES = re.compile(
    r"ignore (all|previous|the above)|prompt injection|sudo|rm -rf|api[_ ]?key|"
    r"password|secret|credential|\.env|private key|exfiltrat|override|jailbreak|"
    r"disregard|bypass|token|ssh|/etc/passwd|base64",
    re.IGNORECASE,
)


def rank_norm(values: list[float]) -> dict[int, float]:
    """Rank-normalize to [0,1] (robust to outliers). Ties share averaged rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = {}
    n = max(len(values) - 1, 1)
    for rank, idx in enumerate(order):
        out[idx] = rank / n
    return out


def load_optional_scores(name: str) -> dict[str, float]:
    path = config.STEP_DIRS["scores"] / f"{name}.json"
    if path.exists():
        print(f"  using {name} scores: {path}")
        return load_json(path)
    print(f"  {name} scores absent -> neutral 0.5")
    return {}


def risk_present(rec: dict) -> bool:
    if rec["label"] == "unsafe":
        return True
    return bool(_RISK_CUES.search(extract_trajectory_from_instruction(rec["instruction"])))


def compute_scores(pool: list[dict], weights: dict[str, float]) -> list[float]:
    learn = load_optional_scores("learnability")
    conf = load_optional_scores("label_confidence")

    # rarity from the full joint taxonomy tuple (risk_source, failure_mode, harm)
    # per the paper's "both marginal categories and joint risk tuples" monitoring;
    # samples without taxonomy (binary source) share one bucket.
    joint = Counter(
        (r.get("risk_source"), r.get("failure_mode"), r.get("harm")) for r in pool
    )
    rarity_raw = [
        1.0 / joint[(r.get("risk_source"), r.get("failure_mode"), r.get("harm"))]
        for r in pool
    ]
    rarity = rank_norm(rarity_raw)

    scores = []
    for i, r in enumerate(pool):
        s = (
            weights["learn"] * learn.get(r["uid"], 0.5)
            + weights["rare"] * rarity[i]
            + weights["conf"] * conf.get(r["uid"], 0.5)
        )
        scores.append(s)
    return scores


def select(pool: list[dict], scores: list[float], target: int, safe_risk_frac: float,
           risk_floor: int) -> list[int]:
    idx_safe = [i for i, r in enumerate(pool) if r["label"] == "safe"]
    idx_unsafe = [i for i, r in enumerate(pool) if r["label"] == "unsafe"]

    safe_target = min(target // 2, len(idx_safe))
    unsafe_target = min(target - safe_target, len(idx_unsafe))

    by_score = lambda i: scores[i]

    # --- safe: fill "risk present" quota first, then benign ----------------
    safe_risk = sorted([i for i in idx_safe if risk_present(pool[i])], key=by_score, reverse=True)
    safe_benign = sorted([i for i in idx_safe if not risk_present(pool[i])], key=by_score, reverse=True)
    n_risk = min(len(safe_risk), int(round(safe_target * safe_risk_frac)))
    picked_safe = safe_risk[:n_risk]
    picked_safe += (safe_benign + safe_risk[n_risk:])[: safe_target - len(picked_safe)]

    # --- unsafe: soft balance across risk_source with a floor --------------
    # Within each risk_source cell, order by round-robin over the
    # (failure_mode, harm) sub-cells (highest score first inside each sub-cell)
    # so the joint taxonomy tuples stay diverse, per the paper's joint-tuple
    # balancing. Plain score-sort would let one dominant fm/harm combo fill the
    # whole cell.
    cells: dict[object, list[int]] = defaultdict(list)
    for i in idx_unsafe:
        cells[pool[i].get("risk_source") or "unknown"].append(i)
    for key in cells:
        subcells: dict[object, list[int]] = defaultdict(list)
        for i in cells[key]:
            subcells[(pool[i].get("failure_mode"), pool[i].get("harm"))].append(i)
        for sub in subcells.values():
            sub.sort(key=by_score, reverse=True)
        order = sorted(subcells.values(), key=lambda s: -by_score(s[0]))
        interleaved: list[int] = []
        depth = 0
        while len(interleaved) < len(cells[key]):
            for sub in order:
                if depth < len(sub):
                    interleaved.append(sub[depth])
            depth += 1
        cells[key] = interleaved

    picked_unsafe: list[int] = []
    # floor pass: guarantee coverage of every risk_source
    for key, members in cells.items():
        picked_unsafe += members[: min(risk_floor, len(members))]
    # soft-balance pass: allocate remaining by sqrt(freq) weight
    remaining = unsafe_target - len(picked_unsafe)
    if remaining > 0:
        weights = {k: np.sqrt(len(v)) for k, v in cells.items()}
        wsum = sum(weights.values()) or 1.0
        pos = {k: min(risk_floor, len(v)) for k, v in cells.items()}
        alloc = {k: int(remaining * w / wsum) for k, w in weights.items()}
        for key, members in cells.items():
            take = members[pos[key]: pos[key] + alloc[key]]
            picked_unsafe += take
            pos[key] += len(take)
        # top up any shortfall by global score
        if len(picked_unsafe) < unsafe_target:
            chosen = set(picked_unsafe)
            rest = sorted((i for i in idx_unsafe if i not in chosen), key=by_score, reverse=True)
            picked_unsafe += rest[: unsafe_target - len(picked_unsafe)]

    picked_unsafe = picked_unsafe[:unsafe_target]
    return picked_safe + picked_unsafe


def summarize(pool: list[dict], chosen: list[int]) -> list[str]:
    recs = [pool[i] for i in chosen]
    labels = Counter(r["label"] for r in recs)
    unsafe_recs = [r for r in recs if r["label"] == "unsafe"]
    risk = Counter(r.get("risk_source") or "unknown" for r in unsafe_recs)
    modes = Counter(r.get("failure_mode") for r in unsafe_recs if r.get("failure_mode"))
    harms = Counter(r.get("harm") for r in unsafe_recs if r.get("harm"))
    joints = Counter(
        (r.get("risk_source"), r.get("failure_mode"), r.get("harm"))
        for r in unsafe_recs
        if r.get("risk_source")
    )
    n_safe_risk = sum(1 for r in recs if r["label"] == "safe" and risk_present(r))
    n_safe = labels.get("safe", 0)
    lengths = [r.get("traj_chars", 0) for r in recs]
    out = [
        f"- total: **{len(recs)}**  labels: {dict(labels)}",
        f"- safe with risk-present-but-handled: {n_safe_risk}/{n_safe}"
        f" ({(n_safe_risk / n_safe if n_safe else 0):.0%})",
        f"- trajectory length chars: min={min(lengths) if lengths else 0} "
        f"median={int(np.median(lengths)) if lengths else 0} max={max(lengths) if lengths else 0}",
        f"- unsafe coverage: risk_source={len(risk)} | failure_mode={len(modes)}"
        f" | harm={len(harms)} | joint tuples={len(joints)}",
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scored selection & compression")
    parser.add_argument("--sizes", default="500,1000,2000,full",
                        help="comma-separated target sizes; 'full' = keep all")
    parser.add_argument("--safe-risk-frac", type=float, default=0.4,
                        help="min fraction of safe picks that must be risk-present")
    parser.add_argument("--risk-floor", type=int, default=25,
                        help="min unsafe picks per risk_source class")
    parser.add_argument("--w-learn", type=float, default=0.5)
    parser.add_argument("--w-rare", type=float, default=0.3)
    parser.add_argument("--w-conf", type=float, default=0.2)
    args = parser.parse_args()

    config.ensure_dirs()
    out_dir = config.STEP_DIRS["step5"]
    pool = load_json(config.STEP_DIRS["step2"] / "valid_pool.json")
    print(f"Loaded valid pool: {len(pool)}")

    weights = {"learn": args.w_learn, "rare": args.w_rare, "conf": args.w_conf}
    scores = compute_scores(pool, weights)

    n_safe = sum(1 for r in pool if r["label"] == "safe")
    n_unsafe = len(pool) - n_safe
    report = [
        "# Step 5 - Selection & Compression Report",
        "",
        f"- available: safe={n_safe}, unsafe={n_unsafe}",
        f"- score weights: {weights}",
        f"- safe-risk-frac={args.safe_risk_frac}, risk-floor={args.risk_floor}",
        "",
    ]

    for token in args.sizes.split(","):
        token = token.strip()
        if token == "full":
            chosen = list(range(len(pool)))
            tag = "full"
        else:
            target = int(token)
            chosen = select(pool, scores, target, args.safe_risk_frac, args.risk_floor)
            tag = str(target)
        subset = [pool[i] for i in chosen]
        save_json(subset, out_dir / f"selected_{tag}.json")
        report.append(f"## size = {tag}  (actual {len(subset)})")
        report += summarize(pool, chosen)
        report.append("")
        print(f"  selected_{tag}: {len(subset)}")

    write_text("\n".join(report), out_dir / "selection_report.md")
    print(f"\nWrote subsets + {out_dir / 'selection_report.md'}")


if __name__ == "__main__":
    main()
