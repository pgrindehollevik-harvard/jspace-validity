from random import Random

import pytest

from jspace.frozen_explore import (
    bootstrap_selectivity_index,
    downflips,
    hypergeometric_overlap,
    jaccard,
    min_detectable_drop,
    overlap_tail_p,
    selectivity_index,
    sign_test_p,
)


def test_sign_test_matches_exact_binomial():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(3, 3) == pytest.approx(1.0)
    # 49 of 50 one-way: p = 2 * (C(50,0) + C(50,1)) / 2**50
    assert sign_test_p(49, 1) == pytest.approx(2 * 51 / 2**50)
    assert sign_test_p(6, 0) == pytest.approx(2 / 64)


def test_selectivity_index_ratio_of_disagreements():
    assert selectivity_index(0.766, 0.851) == pytest.approx(0.234 / 0.149)
    assert selectivity_index(0.9, 0.9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        selectivity_index(0.9, 1.0)


def test_bootstrap_selectivity_index_is_seeded_and_ordered():
    j = [0.7, 0.8, 0.75, 0.72, 0.78]
    r = [0.85, 0.9, 0.88, 0.86, 0.87]
    lo, hi = bootstrap_selectivity_index(j, r, Random(7))
    lo2, hi2 = bootstrap_selectivity_index(j, r, Random(7))
    assert (lo, hi) == (lo2, hi2)
    assert 0 < lo <= hi
    with pytest.raises(ValueError):
        bootstrap_selectivity_index([0.5], [0.5, 0.6], Random(0))


def test_min_detectable_drop_at_registered_n():
    limits = min_detectable_drop(55)
    # 5 all-one-way discordant pairs give p = .0625; 6 give p = .03125.
    assert limits["min_discordant_all_one_way"] == 6
    assert limits["min_detectable_drop"] == pytest.approx(6 / 55)
    by_shape = {(e["down"], e["up"]): e["p"] for e in limits["examples"]}
    assert by_shape[(5, 0)] == pytest.approx(0.0625)
    assert by_shape[(6, 0)] == pytest.approx(0.03125)


def test_downflips_and_jaccard():
    clean = [True, True, False, True]
    arm = [True, False, False, False]
    assert downflips(clean, arm) == [1, 3]
    assert jaccard({1, 3}, {3, 5}) == pytest.approx(1 / 3)


def test_hypergeometric_overlap_null():
    null = hypergeometric_overlap(universe=41, size_a=9, size_b=9)
    assert sum(null["pmf"].values()) == pytest.approx(1.0)
    assert null["expected_overlap"] == pytest.approx(81 / 41)
    # Observing at least 8 shared items out of 9-and-9 in a 41 universe is extreme.
    assert overlap_tail_p(41, 9, 9, 8) < 1e-6
    assert overlap_tail_p(41, 9, 9, 0) == pytest.approx(1.0)
