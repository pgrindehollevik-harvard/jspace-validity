import json
import random

from jspace import scale_analyze


def test_historical_baseline_fails_only_registered_selectivity():
    row, hits = scale_analyze.historical_row()
    scale_analyze.add_intervals(row, hits, None, random.Random(20260825))
    assert row["profile"] == "qwen3-4b"
    assert not row["passes_v2_equivalent_gate"]
    assert not row["checks"]["wikitext_selectivity"]
    assert all(passed for name, passed in row["checks"].items() if name != "wikitext_selectivity")
    assert row["specificity_ci95"] is not None


def test_locked_analysis_preserves_early_stop_as_missing(tmp_path, monkeypatch):
    result = tmp_path / "extension/results/qwen3-8b/scale_gate.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "metrics": {},
                "role": "secondary",
                "checks": {"lens_validation": False},
                "passes_gate": False,
                "status": "STOP",
                "stop_reason": "lens_validation_failed",
                "environment": {},
                "peak_cuda_gb": 1.0,
            }
        )
    )
    monkeypatch.setattr(scale_analyze, "REPO_ROOT", tmp_path)
    row, hits, wiki = scale_analyze.extension_row("qwen3-8b")
    scale_analyze.add_intervals(row, hits, wiki, random.Random(1))
    assert row["stop_reason"] == "lens_validation_failed"
    assert row["specificity_random_minus_j"] is None
    assert row["specificity_ci95"] is None
