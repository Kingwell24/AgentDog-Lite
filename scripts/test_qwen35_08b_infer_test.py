import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_qwen35_08b_infer_test.py")
SPEC = importlib.util.spec_from_file_location("infer_test", MODULE_PATH)
infer_test = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer_test)


def test_parse_hf_files_defaults_and_commas():
    assert infer_test.parse_hf_files(None) == [
        "summer_camp_ATBench300.json",
        "summer_camp_rjudge.json",
    ]
    assert infer_test.parse_hf_files(["a.json,b.json", "c.json"]) == [
        "a.json",
        "b.json",
        "c.json",
    ]


def test_stem_for_output_handles_json_suffixes():
    assert infer_test.stem_for_output(Path("summer_camp_ATBench300.json")) == "summer_camp_ATBench300"
    assert infer_test.stem_for_output(Path("x.jsonl")) == "x"


if __name__ == "__main__":
    test_parse_hf_files_defaults_and_commas()
    test_stem_for_output_handles_json_suffixes()
    print("generic inference tests passed")
