"""
Tests for evtools.metrics — utility-discounted accuracies and aggregators.
"""

import pytest

from evtools.metrics import (
    discounted_accuracy, utility_score, u65, u80,
    mean_discounted_accuracy, mean_u65, mean_u80, mean_utility_score,
)


# ---------------------------------------------------------------------------
# Per-instance metrics
# ---------------------------------------------------------------------------

class TestDiscountedAccuracy:

    def test_precise_correct(self):
        assert discounted_accuracy(frozenset({"a"}), "a") == 1.0

    def test_partial_correct_size_2(self):
        assert discounted_accuracy(frozenset({"a", "h"}), "a") == 0.5

    def test_partial_correct_size_3(self):
        assert abs(discounted_accuracy(frozenset({"a", "h", "r"}), "a") - 1/3) < 1e-12

    def test_wrong_omega_not_in_d(self):
        assert discounted_accuracy(frozenset({"a", "h"}), "r") == 0.0

    def test_empty_decision(self):
        assert discounted_accuracy(frozenset(), "a") == 0.0


class TestU65:

    def test_precise_correct_is_one(self):
        # x=1: 1.6·1 − 0.6·1 = 1.0
        assert u65(frozenset({"a"}), "a") == pytest.approx(1.0)

    def test_size_2_correct_is_065(self):
        # x=0.5: 1.6·0.5 − 0.6·0.25 = 0.8 − 0.15 = 0.65
        assert u65(frozenset({"a", "h"}), "a") == pytest.approx(0.65)

    def test_size_3_correct(self):
        x = 1/3
        expected = 1.6 * x - 0.6 * x * x
        assert u65(frozenset({"a", "h", "r"}), "a") == pytest.approx(expected)

    def test_wrong_is_zero(self):
        assert u65(frozenset({"a", "h"}), "r") == 0.0

    def test_empty_is_zero(self):
        assert u65(frozenset(), "a") == 0.0


class TestU80:

    def test_precise_correct_is_one(self):
        # x=1: 2.2 − 1.2 = 1.0
        assert u80(frozenset({"a"}), "a") == pytest.approx(1.0)

    def test_size_2_correct_is_080(self):
        # x=0.5: 2.2·0.5 − 1.2·0.25 = 1.1 − 0.3 = 0.8
        assert u80(frozenset({"a", "h"}), "a") == pytest.approx(0.80)

    def test_wrong_is_zero(self):
        assert u80(frozenset({"a", "h"}), "r") == 0.0

    def test_more_risk_averse_than_u65(self):
        # u80 ≥ u65 for all correct partial decisions of size ≥ 2
        for d in [frozenset({"a", "h"}), frozenset({"a", "h", "r"})]:
            assert u80(d, "a") > u65(d, "a")


class TestUtilityScore:

    def test_u65_equivalence(self):
        d = frozenset({"a", "h"})
        assert utility_score(d, "a", a=1.6, b=0.6) == pytest.approx(u65(d, "a"))

    def test_u80_equivalence(self):
        d = frozenset({"a", "h"})
        assert utility_score(d, "a", a=2.2, b=1.2) == pytest.approx(u80(d, "a"))

    def test_zero_when_omega_not_in_d(self):
        assert utility_score(frozenset({"a"}), "r", a=1.6, b=0.6) == 0.0

    def test_custom_coefficients(self):
        # u(x) = 2x − x² with x=0.5 → 1.0 − 0.25 = 0.75
        d = frozenset({"a", "h"})
        assert utility_score(d, "a", a=2.0, b=1.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Batch aggregators (mean over a dataset)
# ---------------------------------------------------------------------------

class TestMeanAggregators:

    PREDS  = [frozenset({"a"}), frozenset({"a", "h"}), frozenset({"r"})]
    LABELS = ["a",              "a",                    "a"]
    # x: [1.0, 0.5, 0.0] → mean_x = 0.5
    # u65: [1.0, 0.65, 0.0] → mean = 0.55
    # u80: [1.0, 0.80, 0.0] → mean = 0.60

    def test_mean_discounted_accuracy(self):
        assert mean_discounted_accuracy(self.PREDS, self.LABELS) == pytest.approx(0.5)

    def test_mean_u65(self):
        assert mean_u65(self.PREDS, self.LABELS) == pytest.approx(0.55)

    def test_mean_u80(self):
        assert mean_u80(self.PREDS, self.LABELS) == pytest.approx(0.60)

    def test_mean_utility_score_matches_mean_u65(self):
        m1 = mean_utility_score(self.PREDS, self.LABELS, a=1.6, b=0.6)
        m2 = mean_u65(self.PREDS, self.LABELS)
        assert m1 == pytest.approx(m2)

    def test_empty_returns_zero(self):
        # Degenerate: no predictions → 0.0 (no division by zero)
        assert mean_u65([], []) == 0.0

    def test_single_correct_precise(self):
        assert mean_u65([frozenset({"a"})], ["a"]) == pytest.approx(1.0)
        assert mean_u80([frozenset({"a"})], ["a"]) == pytest.approx(1.0)
