import importlib.util
import io
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


def test_build_prompt_keeps_literal_json_examples():
    template = (
        'Return {"judgment":"safe"} or {"judgment":"unsafe"}.\n'
        "<BEGIN>{trajectory}</BEGIN>\n"
        "<TOOLS>{tools}</TOOLS>"
    )
    record = {
        "id": 1,
        "label": 0,
        "contents": [[{"role": "user", "content": "hello"}]],
        "tool_used": [{"name": "noop", "description": "No operation."}],
    }

    prompt = baseline.build_prompt(template, record)

    assert '{"judgment":"safe"}' in prompt
    assert "[USER] hello" in prompt
    assert "noop" in prompt


def test_progress_reporter_hides_sample_details():
    stream = io.StringIO()
    progress = baseline.ProgressReporter(total=3, stream=stream)

    progress.update(1)
    progress.update(3)
    progress.finish()

    output = stream.getvalue()
    assert "3/3" in output
    assert "id=" not in output
    assert "label=" not in output
    assert "pred=" not in output
    assert "unsafe" not in output
    assert "raw_output" not in output


def test_progress_reporter_can_be_disabled():
    stream = io.StringIO()
    progress = baseline.ProgressReporter(total=2, enabled=False, stream=stream)

    progress.update(1)
    progress.finish()

    assert stream.getvalue() == ""


def test_iter_batches_splits_records_without_dropping_tail():
    records = [{"id": index} for index in range(5)]

    batches = list(baseline.iter_batches(records, batch_size=2))

    assert [[item["id"] for item in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_iter_batches_rejects_invalid_batch_size():
    try:
        list(baseline.iter_batches([{"id": 1}], batch_size=0))
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("batch_size=0 should fail")


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
    test_build_prompt_keeps_literal_json_examples()
    test_progress_reporter_hides_sample_details()
    test_progress_reporter_can_be_disabled()
    test_iter_batches_splits_records_without_dropping_tail()
    test_iter_batches_rejects_invalid_batch_size()
    test_local_inference_guard_blocks_non_server_without_override()
    print("baseline unit tests passed")
