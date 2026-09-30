from jspace.lens_validation import passes_gate, synonyms


def test_registered_synonyms_are_explicit():
    assert "times" in synonyms("multiplication")
    assert synonyms("7") == ["7", "seven"]
    assert synonyms("Mars") == ["Mars"]


def test_lens_gate_requires_noninferiority_and_one_strict_gain():
    report = {
        "order_ops": {
            "jlens": {"1": 0.4, "5": 0.6},
            "logit_lens": {"1": 0.3, "5": 0.6},
        },
        "multihop": {
            "jlens": {"1": 0.2, "5": 0.4},
            "logit_lens": {"1": 0.3, "5": 0.5},
        },
    }
    assert passes_gate(report)
    report["order_ops"]["jlens"]["1"] = 0.3
    assert not passes_gate(report)
