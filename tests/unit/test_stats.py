import pytest

from jspace.stats import mcnemar_exact_p


def test_exact_mcnemar_extremes():
    assert mcnemar_exact_p([True] * 6, [False] * 6) == pytest.approx(0.03125)
    assert mcnemar_exact_p([True, False], [True, False]) == 1.0


def test_exact_mcnemar_requires_pairing():
    with pytest.raises(ValueError, match="equal length"):
        mcnemar_exact_p([True], [])
