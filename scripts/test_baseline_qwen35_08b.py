import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_baseline_qwen35_08b.py")
SPEC = importlib.util.spec_from_file_location("baseline", MODULE_PATH)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def test_parse_judgment_prefers_json():
    assert baseline.parse_judgment('{"judgment":"unsafe"}') == "unsafe"
    assert baseline.parse_judgment('prefix {"judgment": "safe"} suffix') == "safe"


def test_parse_judgment_accepts_tag_and_plain_fallbacks():
    assert baseline.parse_judgment("<Judgment> unsafe </Judgment>") == "unsafe"
    assert baseline.parse_judgment("The trajectory is SAFE.") == "safe"
    assert baseline.parse_judgment("unclear") is None


def test_compute_metrics_counts_invalid_predictions():
    rows = [
        {"label": 1, "prediction": "unsafe", "output_tokens": 4},
        {"label": 0, "prediction": "unsafe", "output_tokens": 2},
        {"label": 0, "prediction": "safe", "output_tokens": 6},
        {"label": 1, "prediction": None, "output_tokens": 0},
    ]

    metrics = baseline.compute_metrics(rows)

    assert metrics["total"] == 4
    assert metrics["invalid"] == 1
    assert metrics["accuracy"] == 0.5
    assert round(metrics["precision"], 6) == 0.5
    assert round(metrics["recall"], 6) == 0.5
    assert round(metrics["f1"], 6) == 0.5
    assert metrics["avg_output_tokens"] == 3.0


def test_local_inference_guard_blocks_non_server_without_override():
    assert baseline.should_block_inference(
        allow_local_run=False,
        platform_name="Windows",
        cuda_available=False,
    )
    assert not baseline.should_block_inference(
        allow_local_run=True,
        platform_name="Windows",
        cuda_available=False,
    )
    assert not baseline.should_block_inference(
        allow_local_run=False,
        platform_name="Linux",
        cuda_available=True,
    )


if __name__ == "__main__":
    test_parse_judgment_prefers_json()
    test_parse_judgment_accepts_tag_and_plain_fallbacks()
    test_compute_metrics_counts_invalid_predictions()
    test_local_inference_guard_blocks_non_server_without_override()
    print("baseline unit tests passed")
