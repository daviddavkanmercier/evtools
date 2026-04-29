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


# ---------------------------------------------------------------------------
# pl_loss / mean_pl_loss  (E_pl and Ẽ_pl from Mutmainah 2021)
# ---------------------------------------------------------------------------

import numpy as np
from evtools.dsvector import DSVector
from evtools.metrics import pl_loss, mean_pl_loss


FRAME = ["a", "h", "r"]


class TestPlLoss:

    def test_perfect_hard_prediction_gives_zero(self):
        # Prediction is categorical on {a}, true label is "a".
        # pl({a})=1, pl({h})=pl({r})=0; δ = (1, 0, 0). diff = 0 → loss = 0.
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        assert pl_loss([m], ["a"]) == pytest.approx(0.0)

    def test_perfect_wrong_hard_prediction(self):
        # Prediction is categorical on {a}, true label is "h".
        # pl({a})=1, pl({h})=0, pl({r})=0; δ = (0, 1, 0).
        # diff = (1, -1, 0) → squared = 1+1+0 = 2.
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        assert pl_loss([m], ["h"]) == pytest.approx(2.0)

    def test_vacuous_against_hard_label(self):
        # Vacuous: pl = (1, 1, 1) on singletons. δ = (1, 0, 0) for label "a".
        # diff = (0, 1, 1) → squared = 0+1+1 = 2.
        m = DSVector.from_focal(FRAME, {})  # vacuous → m(Ω)=1
        assert pl_loss([m], ["a"]) == pytest.approx(2.0)

    def test_known_result_two_atom_frame(self):
        # Frame = {a, b}, m({a})=0.6, m({a,b})=0.4
        # Then pl({a}) = 1, pl({b}) = 0.4. With label "a" → δ=(1,0).
        # diff = (0, 0.4) → squared = 0.16.
        m = DSVector.from_focal(["a", "b"], {"a": 0.6})  # rest mass to Ω
        assert pl_loss([m], ["a"]) == pytest.approx(0.16)

    def test_perfect_soft_label_self(self):
        # If soft label = prediction, contour functions match → loss = 0.
        m = DSVector.from_focal(FRAME, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3})
        assert pl_loss([m], [m]) == pytest.approx(0.0)

    def test_soft_label_categorical_equals_hard(self):
        # A soft label categorical on {a} should give the same loss as the
        # hard label "a" (its contour function is the indicator on {a}).
        m_pred = DSVector.from_focal(FRAME, {"a": 0.6, "h": 0.2, "r": 0.2})
        soft   = DSVector.from_focal(FRAME, {"a": 1.0})
        loss_hard = pl_loss([m_pred], ["a"])
        loss_soft = pl_loss([m_pred], [soft])
        assert loss_hard == pytest.approx(loss_soft)

    def test_mixed_hard_and_soft_labels(self):
        # Mixing within the same call is supported.
        m1 = DSVector.from_focal(FRAME, {"a": 1.0})
        m2 = DSVector.from_focal(FRAME, {"a": 0.5, "h": 0.5})
        soft_h = DSVector.from_focal(FRAME, {"h": 1.0})
        # m1 vs "a" → 0,    m2 vs soft_h:
        # pl(m2) on singletons = (0.5, 0.5, 0)  (m({a}) + m({a,h}) etc.)
        # δ̃ for soft_h = (0, 1, 0)
        # diff = (0.5, -0.5, 0) → 0.5
        loss_total = pl_loss([m1, m2], ["a", soft_h])
        assert loss_total == pytest.approx(0.5)

    def test_dataset_summed_not_averaged(self):
        # pl_loss sums over instances; mean_pl_loss divides by n.
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        loss3   = pl_loss([m, m, m], ["h", "h", "h"])
        loss1   = pl_loss([m],       ["h"])
        assert loss3 == pytest.approx(3 * loss1)

    def test_lengths_mismatch_raises(self):
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        with pytest.raises(ValueError, match="length"):
            pl_loss([m, m], ["a"])

    def test_invalid_label_type_raises(self):
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        with pytest.raises(TypeError, match="str|DSVector"):
            pl_loss([m], [123])

    def test_soft_label_frame_mismatch_raises(self):
        m_pred = DSVector.from_focal(FRAME, {"a": 1.0})
        m_lbl  = DSVector.from_focal(["x", "y", "z"], {"x": 1.0})
        with pytest.raises(ValueError, match="frame"):
            pl_loss([m_pred], [m_lbl])


class TestMeanPlLoss:

    def test_mean_equals_loss_over_n(self):
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        preds  = [m, m, m]
        labels = ["h", "h", "h"]
        assert mean_pl_loss(preds, labels) == pytest.approx(pl_loss(preds, labels) / 3)

    def test_empty_returns_zero(self):
        assert mean_pl_loss([], []) == 0.0

    def test_perfect_predictions_zero_mean(self):
        m = DSVector.from_focal(FRAME, {"a": 1.0})
        assert mean_pl_loss([m, m], ["a", "a"]) == 0.0
