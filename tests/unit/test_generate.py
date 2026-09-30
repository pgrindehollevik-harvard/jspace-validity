import pytest

from jspace.generate import _merge_norm_stats


def test_norm_aggregation_is_weighted_by_intervention_slots():
    state = {}
    _merge_norm_stats(state, {14: 0.1}, {14: 100})
    _merge_norm_stats(state, {14: 0.9}, {14: 1})
    assert state["norm_counts"]["14"] == 101
    assert state["norms"]["14"] == pytest.approx((0.1 * 100 + 0.9) / 101)
