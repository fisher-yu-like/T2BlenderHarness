from __future__ import annotations


def test_all_supported_vlm_report_names_are_lowercase() -> None:
    from evaluator.openai_vlm import canonical_vlm_name

    assert canonical_vlm_name("gpt-5.6-Luna") == "gpt-5.6-luna"
    assert canonical_vlm_name("gpt-5.6-Terra") == "gpt-5.6-terra"
