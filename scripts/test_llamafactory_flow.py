import importlib.util
import json
import tempfile
from pathlib import Path


def load_module(name: str, filename: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_agentdog_data", "prepare_agentdog_data.py")
evaluate = load_module("evaluate", "evaluate.py")


def test_prepare_records_balances_safe_against_total_unsafe():
    binary_rows = [
        {
            "instruction": "<BEGIN TRAJECTORY>\nT safe\n<END TRAJECTORY>",
            "input": "",
            "output": "safe",
        },
        {
            "instruction": "<BEGIN TRAJECTORY>\nT unsafe\n<END TRAJECTORY>",
            "input": "",
            "output": "unsafe",
        },
    ]
    fine_rows = [
        {
            "instruction": "<BEGIN TRAJECTORY>\nT fine 1\n<END TRAJECTORY>",
            "input": "",
            "output": (
                "Risk Source: Direct Prompt Injection\n"
                "Failure Mode: Choosing Malicious Tool\n"
                "Real World Harm: Financial & Economic Harm"
            ),
        },
        {
            "instruction": "<BEGIN TRAJECTORY>\nT fine 2\n<END TRAJECTORY>",
            "input": "",
            "output": (
                "Risk Source: Tool Description Injection\n"
                "Failure Mode: Failure to Validate Tool Outputs\n"
                "Real World Harm: Privacy & Confidentiality Harm"
            ),
        },
    ]

    records, stats, rejected = prepare.prepare_records(binary_rows, fine_rows, seed=7)

    assert rejected == []
    assert stats["final_safe"] == 3
    assert stats["final_unsafe"] == 3
    assert len(records) == 6
    assert {row["output"] for row in records if "T safe" in row["input"]} == {
        '{"judgment":"safe"}'
    }
    assert any(
        json.loads(row["output"]).get("risk_source") == "Direct Prompt Injection"
        for row in records
    )


def test_prepare_records_reports_rejected_rows():
    records, stats, rejected = prepare.prepare_records(
        [{"instruction": "missing tags", "input": "", "output": "maybe"}],
        [],
        seed=42,
    )

    assert records == []
    assert stats["rejected_count"] == 1
    assert rejected[0]["source"] == "binary"
    assert "trajectory" in rejected[0]["reason"] or "label" in rejected[0]["reason"]


def test_evaluate_metrics_parse_raw_output_and_count_invalid():
    rows = [
        {"id": 1, "label": 1, "raw_output": '{"judgment":"unsafe"}', "output_tokens": 4},
        {"id": 2, "label": 0, "raw_output": '{"judgment":"unsafe"}', "output_tokens": 2},
        {"id": 3, "label": 0, "raw_output": '{"judgment":"safe"}', "output_tokens": 6},
        {"id": 4, "label": 1, "raw_output": "not json", "output_tokens": 0},
    ]

    normalized = [evaluate.normalize_prediction_row(row) for row in rows]
    metrics = evaluate.compute_metrics(normalized)

    assert metrics["total"] == 4
    assert metrics["invalid"] == 1
    assert metrics["accuracy"] == 0.5
    assert round(metrics["precision"], 6) == 0.5
    assert round(metrics["recall"], 6) == 0.5
    assert round(metrics["f1"], 6) == 0.5
    assert metrics["avg_output_tokens"] == 3.0


def test_evaluate_tags_pending_run_dir_and_mirrors_adapter_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        run_dir = root / "demo__acc-pending__f1-pending"
        adapter_root = root / "adapters"
        run_dir.mkdir()
        (run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (run_dir / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")

        tagged = evaluate.tag_run_dir(
            run_dir,
            {"accuracy": 0.7421, "f1": 0.6814},
            adapter_output_root=adapter_root,
        )

        assert tagged.name == "demo__acc-0.742__f1-0.681"
        assert tagged.exists()
        mirrored = adapter_root / tagged.name
        assert (mirrored / "adapter_config.json").exists()
        assert (mirrored / "adapter_model.safetensors").exists()


if __name__ == "__main__":
    test_prepare_records_balances_safe_against_total_unsafe()
    test_prepare_records_reports_rejected_rows()
    test_evaluate_metrics_parse_raw_output_and_count_invalid()
    test_evaluate_tags_pending_run_dir_and_mirrors_adapter_files()
    print("llamafactory flow tests passed")
