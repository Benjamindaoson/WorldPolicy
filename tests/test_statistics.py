import pytest

from worldpolicy.statistics import paired_counts, wilson_interval


def test_wilson_and_pairing():
    low, high = wilson_interval(9, 12)
    assert .46 < low < .48 and .90 < high < .92
    assert paired_counts([True, False, True], [True, True, False]) == {
        "both_success": 1, "only_a": 1, "only_b": 1, "both_fail": 0}
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
    with pytest.raises(ValueError):
        paired_counts([True], [])
