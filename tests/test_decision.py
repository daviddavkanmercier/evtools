"""
Tests for evtools.decision — maximin, maximax, pignistic, plp,
probability, hurwicz, strong_dominance, weak_dominance.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind
from evtools.decision import (
    maximin, maximax, pignistic_decision, plp_decision, probability_decision,
    hurwicz, strong_dominance, weak_dominance,
)
from evtools.conversions import betp, plp

FRAME = ["a", "h", "r"]

# Categorical: certainty on {a}
M_CAT = DSVector.from_focal(FRAME, {"a": 1.0})

# Vacuous: full ignorance
M_VAC = DSVector.from_focal(FRAME, {})

# Two singletons + Ω
M = DSVector.from_focal(FRAME, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3})

# Sub-normal — not used directly here, BBA only
M_SUB = DSVector.from_sparse(FRAME, {frozenset(): 0.2, frozenset({"a"}): 0.8})


# ---------------------------------------------------------------------------
# maximin / maximax
# ---------------------------------------------------------------------------

class TestMaximin:

    def test_categorical_picks_truth(self):
        idx, atom = maximin(M_CAT)
        assert atom == "a"
        assert idx == 0

    def test_vacuous_lower_zero_picks_first(self):
        # All lower expected utilities are 0 → argmax returns index 0.
        idx, atom = maximin(M_VAC)
        assert idx == 0

    def test_returns_tuple(self):
        idx, atom = maximin(M)
        assert isinstance(idx, int)
        assert isinstance(atom, str)
        assert atom in FRAME

    def test_identity_utility_known_result(self):
        # m({a})=0.3, m({a,h})=0.4, m({a,h,r})=0.3
        # E_-(a) = 0.3·1 + 0.4·min(1,0) + 0.3·min(1,0,0) = 0.3
        # E_-(h) = 0.3·0 + 0.4·min(0,1) + 0.3·min(0,1,0) = 0
        # E_-(r) = 0.3·0 + 0.4·min(0,0) + 0.3·min(0,0,1) = 0
        idx, atom = maximin(M)
        assert atom == "a"

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            maximin(M.to_bel())

    def test_wrong_utility_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            maximin(M, np.eye(2))


class TestMaximax:

    def test_categorical_picks_truth(self):
        idx, atom = maximax(M_CAT)
        assert atom == "a"

    def test_vacuous_upper_one_picks_first(self):
        # E^+(a_i) = m(Ω)·max(U[i,:]) = 1·1 = 1 for all i with identity U
        idx, atom = maximax(M_VAC)
        assert idx == 0

    def test_identity_utility_known_result(self):
        # E^+(a) = 0.3·1 + 0.4·1 + 0.3·1 = 1.0
        # E^+(h) = 0.3·0 + 0.4·1 + 0.3·1 = 0.7
        # E^+(r) = 0.3·0 + 0.4·0 + 0.3·1 = 0.3
        idx, atom = maximax(M)
        assert atom == "a"

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            maximax(M.to_bel())


# ---------------------------------------------------------------------------
# pignistic_decision
# ---------------------------------------------------------------------------

class TestPignisticDecision:

    def test_categorical_picks_truth(self):
        idx, atom = pignistic_decision(M_CAT)
        assert atom == "a"

    def test_vacuous_picks_first(self):
        # BetP uniform → all expected utilities = 1/3 → argmax is index 0
        idx, atom = pignistic_decision(M_VAC)
        assert idx == 0

    def test_known_result(self):
        # BetP({a})=0.6, BetP({h})=0.3, BetP({r})=0.1
        idx, atom = pignistic_decision(M)
        assert atom == "a"

    def test_conflict_raises(self):
        m_conflict = DSVector.from_sparse(FRAME, {frozenset(): 1.0})
        with pytest.raises(ValueError):
            pignistic_decision(m_conflict)

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            pignistic_decision(M.to_bel())

    def test_custom_utility(self):
        # Identity utility → max BetP. Inverted utility → arg-min BetP.
        # BetP order: a > h > r → with U = J - I (off-diagonal 1) we
        # invert preferences and pick the atom with smallest BetP.
        n = len(FRAME)
        U = np.ones((n, n)) - np.eye(n)
        idx, atom = pignistic_decision(M, U)
        assert atom == "r"


# ---------------------------------------------------------------------------
# plp_decision and probability_decision
# ---------------------------------------------------------------------------

class TestPlpDecision:

    def test_categorical_picks_truth(self):
        idx, atom = plp_decision(M_CAT)
        assert atom == "a"

    def test_vacuous_picks_first(self):
        # PlP uniform → all expected utilities equal → argmax = index 0
        idx, atom = plp_decision(M_VAC)
        assert idx == 0

    def test_known_result(self):
        # PlP({a})=0.5, PlP({h})=0.35, PlP({r})=0.15 → argmax = a
        idx, atom = plp_decision(M)
        assert atom == "a"

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            plp_decision(M.to_bel())

    def test_custom_utility(self):
        # Inverted preferences: with U = J - I, picks atom of smallest PlP.
        n = len(FRAME)
        U = np.ones((n, n)) - np.eye(n)
        idx, atom = plp_decision(M, U)
        assert atom == "r"


class TestProbabilityDecision:

    def test_default_transform_is_plp(self):
        # When transform is omitted, behave like plp_decision.
        assert probability_decision(M) == plp_decision(M)

    def test_explicit_betp_matches_pignistic(self):
        assert probability_decision(M, transform=betp) == pignistic_decision(M)

    def test_explicit_plp_matches_plp_decision(self):
        assert probability_decision(M, transform=plp) == plp_decision(M)

    def test_custom_transform_uniform(self):
        # A custom transform that always returns the uniform distribution
        # → all expected utilities equal → argmax returns index 0.
        n = len(FRAME)
        uniform = lambda dense: np.full(n, 1.0 / n)
        idx, atom = probability_decision(M, transform=uniform)
        assert idx == 0

    def test_custom_transform_one_hot_picks_atom(self):
        # Transform returning a one-hot vector at index 2 → forces choice "r"
        # under identity utility.
        n = len(FRAME)
        one_hot_r = lambda dense: np.eye(n)[2]
        idx, atom = probability_decision(M, transform=one_hot_r)
        assert atom == "r"

    def test_transform_wrong_length_raises(self):
        bad = lambda dense: np.zeros(len(FRAME) + 1)
        with pytest.raises(ValueError, match="length|shape"):
            probability_decision(M, transform=bad)

    def test_betp_conflict_raises(self):
        m_conflict = DSVector.from_sparse(FRAME, {frozenset(): 1.0})
        with pytest.raises(ValueError):
            probability_decision(m_conflict, transform=betp)

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            probability_decision(M.to_bel())


# ---------------------------------------------------------------------------
# hurwicz
# ---------------------------------------------------------------------------

class TestHurwicz:

    def test_alpha_one_is_maximin(self):
        idx_h, _ = hurwicz(M, alpha=1.0)
        idx_m, _ = maximin(M)
        assert idx_h == idx_m

    def test_alpha_zero_is_maximax(self):
        idx_h, _ = hurwicz(M, alpha=0.0)
        idx_m, _ = maximax(M)
        assert idx_h == idx_m

    def test_default_alpha_half(self):
        idx, atom = hurwicz(M)
        assert atom in FRAME

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            hurwicz(M, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            hurwicz(M, alpha=-0.1)

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            hurwicz(M.to_bel())


# ---------------------------------------------------------------------------
# strong_dominance / weak_dominance
# ---------------------------------------------------------------------------

class TestStrongDominance:

    def test_categorical_singleton(self):
        # Bel({a})=1, Pl({a})=1, Pl(others)=0 → only {a} non-dominated
        nd = strong_dominance(M_CAT)
        assert nd == frozenset({"a"})

    def test_vacuous_all_atoms(self):
        # Bel singletons all = 0, Pl singletons all = 1.
        # No atom dominates: all are non-dominated.
        nd = strong_dominance(M_VAC)
        assert nd == frozenset(FRAME)

    def test_returns_frozenset_of_str(self):
        nd = strong_dominance(M)
        assert isinstance(nd, frozenset)
        for x in nd:
            assert isinstance(x, str)
            assert x in FRAME

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            strong_dominance(M.to_bel())


class TestWeakDominance:

    def test_categorical_singleton(self):
        nd = weak_dominance(M_CAT)
        assert nd == frozenset({"a"})

    def test_vacuous_empty_under_ge_relation(self):
        # All bel=0 and all pl=1 → every atom satisfies the ≥-comparison
        # against every other, so each one is "dominated" by another and
        # the non-dominated set is empty. This is a consequence of the
        # non-strict ≥ comparison used in the implementation.
        nd = weak_dominance(M_VAC)
        assert nd == frozenset()

    def test_returns_frozenset_of_str(self):
        nd = weak_dominance(M)
        assert isinstance(nd, frozenset)

    def test_subset_of_strong_dominance_set(self):
        # Under this implementation (non-strict ≥) the weakly non-dominated
        # set is a subset of the strongly non-dominated set: any pair (j,k)
        # with bel_j ≥ pl_k also satisfies bel_j ≥ bel_k AND pl_j ≥ pl_k
        # (since pl ≥ bel), so strong dominance implies weak dominance.
        weak = weak_dominance(M)
        strong = strong_dominance(M)
        assert weak <= strong

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            weak_dominance(M.to_bel())


# ---------------------------------------------------------------------------
# Utility-discounted accuracies (Zaffalon et al. 2012)
# ---------------------------------------------------------------------------

from evtools.decision import discounted_accuracy, utility_score, u65, u80


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
        # x=1/3: 1.6/3 − 0.6/9 ≈ 0.5333 − 0.0667 = 0.4667
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
