from jspace.metrics import metric_report
from jspace.scale_config import load_scale_config


def control_record(hit: bool) -> dict:
    return {"hit": hit, "seconds": 0.0, "norms": {}}


def passing_state(wikitext_matches: int = 81) -> dict:
    return {
        "stages": {
            "lens_validation": {"report": {"gate": "PASS"}},
            "control": {
                "clean": [control_record(True)] * 55,
                "jspace": ([control_record(False)] * 6 + [control_record(True)] * 49),
                "random": [control_record(True)] * 55,
            },
            "wikitext": {
                "jspace": [
                    {
                        "matches": wikitext_matches,
                        "total": 100,
                        "agreement": wikitext_matches / 100,
                        "seconds": 0.0,
                        "norms": {},
                    }
                ],
                "random": [
                    {
                        "matches": 90,
                        "total": 100,
                        "agreement": 0.9,
                        "seconds": 0.0,
                        "norms": {},
                    }
                ],
            },
            "coherence": [
                {
                    "seconds": 1.0,
                    "summary": {"degenerate": False, "tokens": 10, "norms": {}},
                }
            ],
        }
    }


def test_registered_conjunction_can_pass():
    thresholds = load_scale_config()["thresholds"]
    _, checks = metric_report(passing_state(), thresholds)
    assert all(checks.values())


def test_wikitext_threshold_is_strict():
    thresholds = load_scale_config()["thresholds"]
    _, checks = metric_report(passing_state(wikitext_matches=80), thresholds)
    assert checks["wikitext_selectivity"] is False
