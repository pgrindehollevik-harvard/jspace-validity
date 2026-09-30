from jspace.v3_summary import CORE_4B, summarize_model


def synthetic_result(wiki_j, wiki_r, clean, jarm, j_drop):
    return {
        "profile": "qwen3-32b",
        "band": [24, 34],
        "passes_gate": False,
        "status": "STOP",
        "stop_reason": "noncoherence_gate_failed:wikitext_selectivity",
        "checks": {"wikitext_selectivity": False},
        "lens_validation": {"gate": "PASS"},
        "metrics": {
            "wikitext_jspace": {
                "per_sequence_agreement": wiki_j,
                "top1_agreement": sum(wiki_j) / len(wiki_j),
            },
            "wikitext_random": {
                "per_sequence_agreement": wiki_r,
                "top1_agreement": sum(wiki_r) / len(wiki_r),
            },
            "control_clean": {"hits": clean},
            "control_jspace": {"hits": jarm},
            "positive_control": {"j_drop": j_drop},
        },
    }


def test_h1_confirmed_when_j_uniformly_worse():
    wiki_j = [0.70 + 0.001 * i for i in range(50)]
    wiki_r = [value + 0.10 for value in wiki_j]
    clean = [True] * 50 + [False] * 5
    jarm = list(clean)
    jarm[2] = False  # one core down-flip
    row = summarize_model(synthetic_result(wiki_j, wiki_r, clean, jarm, 1 / 55))
    h1 = row["h1_anti_selectivity"]
    assert h1["sequences_j_worse"] == 50 and h1["sequences_j_better"] == 0
    assert h1["sign_test_p"] < 1e-12
    assert h1["selectivity_index"] > 1
    assert h1["confirmed"] is True
    h4 = row["h4_flip_core"]
    assert h4["applicable"] is True
    assert h4["downflips"] == [2]
    assert h4["core_overlap"] == 1
    assert 0 < h4["overlap_tail_p"] < 1


def test_h1_not_confirmed_on_balanced_deltas():
    wiki_j = [0.8, 0.7] * 25
    wiki_r = [0.7, 0.8] * 25
    clean = [True] * 55
    row = summarize_model(synthetic_result(wiki_j, wiki_r, clean, list(clean), 0.0))
    h1 = row["h1_anti_selectivity"]
    assert h1["confirmed"] is False
    assert row["h4_flip_core"]["applicable"] is False


def test_core_constant_matches_manuscript():
    assert CORE_4B == (2, 14, 17, 18, 34, 35, 39, 46, 48)
