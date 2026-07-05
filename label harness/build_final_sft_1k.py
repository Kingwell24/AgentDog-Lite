from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from deepseek_harness import (
    compact_json,
    make_sft_user_prompt,
    make_sft_user_prompt_rich,
    write_jsonl,
)
from filter_app1_benign import tool_signature


ROOT = Path(__file__).resolve().parent
DEFAULT_RICH_POOL = ROOT / "train case" / "Training-Data-After-Harness" / "sft_messages_rich_mitigated_unsafe_balanced.jsonl"
DEFAULT_BENIGN_POOL = ROOT / "train case" / "Training-Data-After-Harness" / "app1_benign_100_diverse" / "app1_benign_accepted.jsonl"
DEFAULT_ATBENCH = ROOT / "test case" / "summer_camp_ATBench300.json"
DEFAULT_RJUDGE = ROOT / "test case" / "summer_camp_rjudge.json"
DEFAULT_OUT = ROOT / "train case" / "SFT-1K-Selected"


TARGET_UNSAFE = 500
TARGET_MITIGATED = 400
TARGET_BENIGN = 100


AT_RISK_SOURCE_MAP = {
    "malicious_user_instruction_or_jailbreak": "Malicious User Instruction or Jailbreak",
    "direct_prompt_injection": "Direct Prompt Injection",
    "indirect_prompt_injection": "Indirect Prompt Injection",
    "unreliable_or_misinformation": "Unreliable or Mis-information",
    "tool_description_injection": "Tool Description Injection",
    "malicious_tool_execution": "Malicious Tool Execution",
    "corrupted_tool_feedback": "Corrupted Tool Feedback",
    "inherent_agent_failures": "Inherent Agent/LLM Failures",
    "benign": "benign",
}

AT_HARM_MAP = {
    "privacy_and_confidentiality_harm": "Privacy & Confidentiality Harm",
    "financial_and_economic_harm": "Financial & Economic Harm",
    "security_and_system_integrity_harm": "Security & System Integrity Harm",
    "physical_and_health_harm": "Physical & Health Harm",
    "psychological_and_emotional_harm": "Psychological & Emotional Harm",
    "reputational_and_interpersonal_harm": "Reputational & Interpersonal Harm",
    "info_ecosystem_and_societal_harm": "Info-ecosystem & Societal Harm",
    "public_service_and_resource_harm": "Public Service & Resource Harm",
    "fairness_equity_and_allocative_harm": "Fairness, Equity, and Allocative Harm",
    "functional_and_opportunity_harm": "Functional & Opportunity Harm",
    "benign": "benign",
}

RJUDGE_HARM_MAP = {
    "Privacy Leakage": "Privacy & Confidentiality Harm",
    "Financial Loss": "Financial & Economic Harm",
    "Computer Security": "Security & System Integrity Harm",
    "Physical Health": "Physical & Health Harm",
    "Illegal Activities": "Security & System Integrity Harm",
    "Data Loss": "Functional & Opportunity Harm",
    "Property Damage": "Functional & Opportunity Harm",
    "Bias & Offensiveness": "Fairness, Equity, and Allocative Harm",
    "Ethics & Morality": "Info-ecosystem & Societal Harm",
    "Miscellaneous": "Functional & Opportunity Harm",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_benign_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        return []
    if "messages" in rows[0]:
        return rows

    converted: list[dict[str, Any]] = []
    assistant = compact_json({"judgment": "safe", "risk_state": "benign"})
    for row in rows:
        trajectory = str(row.get("trajectory", "")).strip()
        if not trajectory:
            continue
        metadata = {
            "source": "APP1-Agentic-Safety-SFT-Data",
            "app1_index": row.get("app1_index"),
            "trajectory_hash": row.get("trajectory_hash"),
            "final_label": "safe",
            "risk_state": "benign",
            "teacher_label": "safe",
            "confidence": float(row.get("confidence", 0.0)),
            "risk_source": "benign",
            "trace_pattern": "benign_task_no_risk",
            "safe_handling": "none_needed",
        }
        converted.append(
            {
                "messages": [
                    {"role": "user", "content": make_sft_user_prompt_rich(trajectory)},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": metadata,
            }
        )
    return converted


def extract_trajectory(prompt: str) -> str:
    match = re.search(r"<TRAJECTORY>\s*(.*?)\s*</TRAJECTORY>", prompt, re.S)
    if not match:
        raise ValueError("SFT prompt does not contain <TRAJECTORY> block")
    return match.group(1).strip()


def row_id(row: dict[str, Any]) -> str:
    meta = row["metadata"]
    if "trajectory_hash" in meta:
        return str(meta["trajectory_hash"])
    return f"row-{meta.get('example_id', meta.get('app1_index', 'unknown'))}"


def normalized_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"0", "safe"}:
        return "safe"
    if text in {"1", "unsafe"}:
        return "unsafe"
    return text


def test_priors(atbench_path: Path, rjudge_path: Path) -> dict[str, Counter[str]]:
    priors: dict[str, Counter[str]] = {
        "risk_source_all": Counter(),
        "risk_source_safe": Counter(),
        "risk_source_unsafe": Counter(),
        "harm_unsafe": Counter(),
    }
    if atbench_path.exists():
        rows = json.loads(atbench_path.read_text(encoding="utf-8"))
        for row in rows:
            label = normalized_label(row.get("label"))
            risk_source = AT_RISK_SOURCE_MAP.get(str(row.get("risk_source")), str(row.get("risk_source")))
            harm_type = AT_HARM_MAP.get(str(row.get("harm_type")), str(row.get("harm_type")))
            if risk_source and risk_source != "benign":
                priors["risk_source_all"][risk_source] += 1
                if label == "safe":
                    priors["risk_source_safe"][risk_source] += 1
                elif label == "unsafe":
                    priors["risk_source_unsafe"][risk_source] += 1
            if label == "unsafe" and harm_type and harm_type != "benign":
                priors["harm_unsafe"][harm_type] += 1
    if rjudge_path.exists():
        rows = json.loads(rjudge_path.read_text(encoding="utf-8"))
        for row in rows:
            label = normalized_label(row.get("label"))
            if label != "unsafe":
                continue
            harm_type = RJUDGE_HARM_MAP.get(str(row.get("risk_type")))
            if harm_type:
                priors["harm_unsafe"][harm_type] += 1
    return priors


def row_quality(row: dict[str, Any], pool_counts: dict[str, Counter[str]], mode: str) -> float:
    meta = row["metadata"]
    prompt_len = len(row["messages"][0]["content"])
    confidence = float(meta.get("confidence", 0.0))
    decision = str(meta.get("decision", "keep"))
    score = confidence
    if decision == "keep":
        score += 0.03
    elif decision == "flip":
        score += 0.01
    if float(meta.get("weight", 1.0)) >= 1.0:
        score += 0.01
    trace_pattern = str(meta.get("trace_pattern", ""))
    if trace_pattern == "uncertain_or_malformed":
        score -= 0.35
    if mode == "unsafe":
        if str(meta.get("commit_type")) in {"unclear", "none"}:
            score -= 0.08
        if str(meta.get("safe_handling")) in {"ignored_injection", "refused_or_halted"}:
            score -= 0.03
        harm = str(meta.get("harm_type", ""))
        count = pool_counts.get("harm_type", Counter()).get(harm, 0)
        if 0 < count <= 20:
            score += 0.03
    elif mode == "mitigated":
        handling = str(meta.get("safe_handling", ""))
        if handling in {"asked_confirmation", "verified_or_cross_checked", "redacted_or_minimized"}:
            score += 0.08
        elif handling == "safe_alternative_or_partial_completion":
            score += 0.04
        elif handling in {"unclear", "not_applicable"}:
            score -= 0.15
    else:
        sig = tool_signature(row["messages"][0]["content"])
        if sig != "other":
            score += 0.02
    if prompt_len > 14000:
        score -= min(0.08, (prompt_len - 14000) / 100000)
    return score


def softened_quotas(
    rows: list[dict[str, Any]],
    key: str,
    total: int,
    test_counts: Counter[str] | None,
    pool_weight: float,
    test_weight: float,
    uniform_weight: float,
) -> dict[str, int]:
    pool_counts = Counter(str(row["metadata"].get(key, "unknown")) for row in rows)
    categories = sorted(pool_counts)
    if not categories:
        return {}
    pool_total = sum(pool_counts.values())
    test_total = sum(test_counts.values()) if test_counts else 0
    raw: dict[str, float] = {}
    for category in categories:
        pool_p = pool_counts[category] / pool_total if pool_total else 0.0
        test_p = (test_counts.get(category, 0) / test_total) if test_total else pool_p
        uniform_p = 1.0 / len(categories)
        raw[category] = total * (
            pool_weight * pool_p
            + test_weight * test_p
            + uniform_weight * uniform_p
        )

    quotas = {category: min(pool_counts[category], int(math.floor(value))) for category, value in raw.items()}
    # Make sure very small but available categories are represented.
    for category in categories:
        if pool_counts[category] > 0 and quotas[category] == 0 and total >= len(categories):
            quotas[category] = 1

    def current_total() -> int:
        return sum(quotas.values())

    while current_total() < total:
        best = None
        best_remainder = float("-inf")
        for category in categories:
            if quotas[category] >= pool_counts[category]:
                continue
            remainder = raw[category] - quotas[category]
            if remainder > best_remainder:
                best = category
                best_remainder = remainder
        if best is None:
            break
        quotas[best] += 1

    while current_total() > total:
        removable = [
            category for category in categories
            if quotas[category] > 0
        ]
        if not removable:
            break
        worst = min(removable, key=lambda category: raw[category] - quotas[category])
        quotas[worst] -= 1
    return quotas


def diverse_take(
    candidates: list[dict[str, Any]],
    target: int,
    mode: str,
    secondary_keys: list[str],
) -> list[dict[str, Any]]:
    if len(candidates) <= target:
        return sorted(candidates, key=lambda row: (-row["_quality"], row_id(row)))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        signature = "|".join(str(row["metadata"].get(key, "")) for key in secondary_keys)
        if mode == "benign":
            signature = tool_signature(row["messages"][0]["content"])
        buckets[signature].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: (-row["_quality"], row_id(row)))

    selected: list[dict[str, Any]] = []
    while buckets and len(selected) < target:
        for signature in sorted(buckets, key=lambda sig: (-len(buckets[sig]), sig)):
            bucket = buckets.get(signature)
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= target:
                break
        buckets = {sig: values for sig, values in buckets.items() if values}
    return selected


def stratified_select(
    rows: list[dict[str, Any]],
    target: int,
    primary_key: str,
    quotas: dict[str, int],
    mode: str,
    secondary_keys: list[str],
    preselected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    if preselected:
        for row in preselected:
            if row_id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id(row))
            if len(selected) >= target:
                return selected

    by_primary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row_id(row) not in selected_ids:
            by_primary[str(row["metadata"].get(primary_key, "unknown"))].append(row)

    selected_count = Counter(str(row["metadata"].get(primary_key, "unknown")) for row in selected)
    for category, quota in sorted(quotas.items(), key=lambda item: (-item[1], item[0])):
        need = max(0, quota - selected_count[category])
        if need <= 0:
            continue
        picked = diverse_take(by_primary.get(category, []), need, mode, secondary_keys)
        for row in picked:
            if row_id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id(row))
            if len(selected) >= target:
                return selected

    if len(selected) < target:
        rest = [row for row in rows if row_id(row) not in selected_ids]
        rest = diverse_take(rest, target - len(selected), mode, secondary_keys)
        selected.extend(rest)
    return selected[:target]


def to_minimal_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = dict(row["metadata"])
    trajectory = extract_trajectory(row["messages"][0]["content"])
    final_label = str(meta.get("final_label", "safe"))
    if final_label not in {"safe", "unsafe"}:
        final_label = "safe" if meta.get("risk_state") == "benign" else final_label
    return {
        "messages": [
            {"role": "user", "content": make_sft_user_prompt(trajectory)},
            {"role": "assistant", "content": compact_json({"judgment": final_label})},
        ],
        "metadata": meta,
    }


def to_agentdog(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "instruction": row["messages"][0]["content"],
            "input": "",
            "output": row["messages"][1]["content"],
            "metadata": row.get("metadata", {}),
        }
        for row in rows
    ]


def counter_for(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row["metadata"].get(key, "unknown")) for row in rows)


def counter_block(title: str, rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append(f"- count: {len(rows)}")
    for key in keys:
        lines.append(f"- {key}: {json.dumps(dict(counter_for(rows, key).most_common()), ensure_ascii=False)}")
    return lines


def length_stats(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    values = sorted(len(row["messages"][0]["content"]) for row in rows)
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {
        "min": values[0],
        "median": median,
        "p90": values[int(0.9 * (len(values) - 1))],
        "max": values[-1],
    }


def run(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rich_pool = load_jsonl(args.rich_pool)
    benign_pool = load_benign_rows(args.benign_pool)
    priors = test_priors(args.atbench, args.rjudge)

    unsafe_rows = [
        row for row in rich_pool
        if row["metadata"].get("risk_state") == "committed"
        and row["metadata"].get("trace_pattern") != "uncertain_or_malformed"
        and row["metadata"].get("commit_type") not in {"unclear", "none"}
        and len(row["messages"][0]["content"]) <= args.max_prompt_chars
    ]
    mitigated_rows = [
        row for row in rich_pool
        if row["metadata"].get("risk_state") == "mitigated"
        and row["metadata"].get("trace_pattern") != "uncertain_or_malformed"
        and row["metadata"].get("safe_handling") not in {"unclear", "not_applicable"}
        and len(row["messages"][0]["content"]) <= args.max_prompt_chars
    ]
    benign_rows = [
        row for row in benign_pool
        if row["metadata"].get("risk_state") == "benign"
        and len(row["messages"][0]["content"]) <= args.max_prompt_chars
    ]
    if len(unsafe_rows) < TARGET_UNSAFE:
        raise SystemExit(f"Not enough unsafe rows after filtering: {len(unsafe_rows)} < {TARGET_UNSAFE}")
    if len(mitigated_rows) < TARGET_MITIGATED:
        raise SystemExit(f"Not enough mitigated rows after filtering: {len(mitigated_rows)} < {TARGET_MITIGATED}")
    if len(benign_rows) < TARGET_BENIGN:
        raise SystemExit(f"Not enough benign rows after filtering: {len(benign_rows)} < {TARGET_BENIGN}")

    pool_counts = {
        "unsafe": {
            "harm_type": counter_for(unsafe_rows, "harm_type"),
            "risk_source": counter_for(unsafe_rows, "risk_source"),
        },
        "mitigated": {
            "safe_handling": counter_for(mitigated_rows, "safe_handling"),
            "risk_source": counter_for(mitigated_rows, "risk_source"),
        },
    }
    for row in unsafe_rows:
        row["_quality"] = row_quality(row, pool_counts["unsafe"], "unsafe")
    for row in mitigated_rows:
        row["_quality"] = row_quality(row, pool_counts["mitigated"], "mitigated")
    for row in benign_rows:
        row["_quality"] = row_quality(row, {}, "benign")

    unsafe_quotas = softened_quotas(
        unsafe_rows,
        "risk_source",
        TARGET_UNSAFE,
        priors["risk_source_unsafe"],
        pool_weight=0.70,
        test_weight=0.15,
        uniform_weight=0.15,
    )
    mitigated_quotas = softened_quotas(
        mitigated_rows,
        "risk_source",
        TARGET_MITIGATED,
        priors["risk_source_safe"] or priors["risk_source_all"],
        pool_weight=0.75,
        test_weight=0.10,
        uniform_weight=0.15,
    )

    unsafe_selected = stratified_select(
        unsafe_rows,
        TARGET_UNSAFE,
        "risk_source",
        unsafe_quotas,
        "unsafe",
        ["failure_mode", "harm_type", "trace_pattern", "commit_type"],
    )

    rare_safe_handlers = {
        "asked_confirmation",
        "verified_or_cross_checked",
        "redacted_or_minimized",
    }
    safe_preselected = [
        row for row in sorted(mitigated_rows, key=lambda row: (-row["_quality"], row_id(row)))
        if row["metadata"].get("safe_handling") in rare_safe_handlers
    ]
    mitigated_selected = stratified_select(
        mitigated_rows,
        TARGET_MITIGATED,
        "risk_source",
        mitigated_quotas,
        "mitigated",
        ["safe_handling", "trace_pattern"],
        preselected=safe_preselected,
    )
    benign_selected = diverse_take(benign_rows, TARGET_BENIGN, "benign", [])

    final_rows = unsafe_selected + mitigated_selected + benign_selected
    for row in final_rows:
        row.pop("_quality", None)
    rng = random.Random(args.seed)
    rng.shuffle(final_rows)

    rich_rows = final_rows
    minimal_rows = [to_minimal_row(row) for row in rich_rows]
    rich_lengths = length_stats(rich_rows)
    minimal_lengths = length_stats(minimal_rows)

    write_jsonl(args.out_dir / "sft_1k_rich_messages.jsonl", rich_rows)
    write_jsonl(args.out_dir / "sft_1k_minimal_messages.jsonl", minimal_rows)
    (args.out_dir / "sft_1k_rich_agentdog.json").write_text(
        json.dumps(to_agentdog(rich_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "sft_1k_minimal_agentdog.json").write_text(
        json.dumps(to_agentdog(minimal_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = []
    for index, row in enumerate(rich_rows):
        meta = dict(row["metadata"])
        manifest.append(
            {
                "selected_index": index,
                "trajectory_hash": meta.get("trajectory_hash"),
                "source": meta.get("source", "AgentDoG1.0-BinarySafety"),
                "example_id": meta.get("example_id"),
                "app1_index": meta.get("app1_index"),
                "final_label": meta.get("final_label"),
                "risk_state": meta.get("risk_state"),
                "risk_source": meta.get("risk_source"),
                "failure_mode": meta.get("failure_mode"),
                "harm_type": meta.get("harm_type"),
                "safe_handling": meta.get("safe_handling"),
                "trace_pattern": meta.get("trace_pattern"),
                "confidence": meta.get("confidence"),
            }
        )
    write_jsonl(args.out_dir / "selection_manifest.jsonl", manifest)

    label_counts = counter_for(rich_rows, "final_label")
    state_counts = counter_for(rich_rows, "risk_state")
    report = [
        "# Final 1K SFT Selection Report",
        "",
        "## Target",
        "",
        f"- unsafe committed: {TARGET_UNSAFE}",
        f"- safe mitigated: {TARGET_MITIGATED}",
        f"- safe benign: {TARGET_BENIGN}",
        f"- total: {len(rich_rows)}",
        "",
        "## Selection Principle",
        "",
        "- Start from DeepSeek-audited AgentDoG1.0 rich pool plus APP1 strict benign supplement.",
        "- Use aggregate test-case taxonomy only as a weak prior, capped at 10-15% of quota computation.",
        "- Prefer high teacher confidence, clear trace pattern, and concrete unsafe commitment/safe handling evidence.",
        "- Exclude uncertain/malformed traces, unclear unsafe commit types, and unclear/not-applicable safe handling.",
        f"- Enforce max rich prompt length <= {args.max_prompt_chars} characters.",
        "- Apply coverage smoothing so rare but important risk sources are represented.",
        "- Use secondary round-robin over failure mode, harm type, trace pattern, commit type, safe handling, and benign tool signatures.",
        "",
        "## Output Counts",
        "",
        f"- final_label: {json.dumps(dict(label_counts), ensure_ascii=False)}",
        f"- risk_state: {json.dumps(dict(state_counts), ensure_ascii=False)}",
        f"- rich prompt chars: {json.dumps(rich_lengths, ensure_ascii=False)}",
        f"- minimal prompt chars: {json.dumps(minimal_lengths, ensure_ascii=False)}",
        "",
        "## Quotas",
        "",
        f"- unsafe risk_source target: {json.dumps(unsafe_quotas, ensure_ascii=False)}",
        f"- mitigated risk_source target: {json.dumps(mitigated_quotas, ensure_ascii=False)}",
        "",
    ]
    report.extend(counter_block("Selected Unsafe", unsafe_selected, ["risk_source", "failure_mode", "harm_type", "trace_pattern", "commit_type"]))
    report.append("")
    report.extend(counter_block("Selected Mitigated Safe", mitigated_selected, ["risk_source", "safe_handling", "trace_pattern"]))
    report.append("")
    report.extend(counter_block("Selected Benign Safe", benign_selected, ["source", "risk_state"]))
    report.extend(
        [
            "",
            "## Files",
            "",
            "- sft_1k_rich_messages.jsonl",
            "- sft_1k_minimal_messages.jsonl",
            "- sft_1k_rich_agentdog.json",
            "- sft_1k_minimal_agentdog.json",
            "- selection_manifest.jsonl",
            "- selection_report.md",
        ]
    )
    (args.out_dir / "selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.out_dir / "selection_config.json").write_text(
        json.dumps(
            {
                "rich_pool": str(args.rich_pool),
                "benign_pool": str(args.benign_pool),
                "atbench": str(args.atbench),
                "rjudge": str(args.rjudge),
                "out_dir": str(args.out_dir),
                "seed": args.seed,
                "max_prompt_chars": args.max_prompt_chars,
                "targets": {
                    "unsafe": TARGET_UNSAFE,
                    "mitigated": TARGET_MITIGATED,
                    "benign": TARGET_BENIGN,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final 1K SFT set from audited AgentDoG and APP1 benign pools.")
    parser.add_argument("--rich-pool", type=Path, default=DEFAULT_RICH_POOL)
    parser.add_argument("--benign-pool", type=Path, default=DEFAULT_BENIGN_POOL)
    parser.add_argument("--atbench", type=Path, default=DEFAULT_ATBENCH)
    parser.add_argument("--rjudge", type=Path, default=DEFAULT_RJUDGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-prompt-chars", type=int, default=10000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
