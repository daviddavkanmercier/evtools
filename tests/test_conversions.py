"""
Tests for evtools.conversions — round-trip consistency checks.

Strategy: start from a valid mass function (m), convert to every other
representation, then convert back to m and assert numerical closeness.
"""

import numpy as np
import pytest
from evtools.conversions import (
    mtob, mtobel, mtopl, mtoq, mtov, mtow,
    btom, beltom, pltom, qtom, vtom, wtom,
)

# Normalised mass function over a 2-atom frame (4 focal elements)
# ∅=0, {a}=1, {b}=2, {a,b}=3
M2 = np.array([0.0, 0.3, 0.5, 0.2])

# Subnormal mass function (m(∅) != 0)
M2_SUB = np.array([0.1, 0.3, 0.4, 0.2])

# Normalised mass function over a 3-atom frame (8 focal elements)
M3 = np.array([0.0, 0.1, 0.2, 0.05, 0.3, 0.1, 0.15, 0.1])


@pytest.mark.parametrize("m", [M2, M2_SUB, M3])
def test_mtob_roundtrip(m):
    assert np.allclose(btom(mtob(m)), m)


@pytest.mark.parametrize("m", [M2, M2_SUB, M3])
def test_mtopl_roundtrip(m):
    assert np.allclose(pltom(mtopl(m)), m)


@pytest.mark.parametrize("m", [M2, M2_SUB, M3])
def test_mtoq_roundtrip(m):
    assert np.allclose(qtom(mtoq(m)), m)


@pytest.mark.parametrize("m", [M2, M3])
def test_mtobel_roundtrip(m):
    assert np.allclose(beltom(mtobel(m)), m)


def test_mtov_roundtrip():
    # v and w require b > 0 everywhere, which means m(∅) > 0 (subnormal BBA).
    # Normalised mass functions (m(∅)=0) are outside the domain of vtob/wtob.
    assert np.allclose(vtom(mtov(M2_SUB)), M2_SUB, atol=1e-10)


def test_mtow_roundtrip():
    assert np.allclose(wtom(mtow(M2_SUB)), M2_SUB, atol=1e-10)


def test_normalised_masses_sum_to_one():
    for m in [M2, M3]:
        assert np.isclose(m.sum(), 1.0)


# ===========================================================================
# betp and plp
# ===========================================================================

import numpy as np
import pytest
from evtools.dsvector import DSVector
from evtools.conversions import betp, plp

FRAME_AHR = ["a", "h", "r"]

M_CAT      = DSVector.from_focal(FRAME_AHR, {"a": 1.0})
M_VAC      = DSVector.from_focal(FRAME_AHR, {})
M_HALF     = DSVector.from_focal(FRAME_AHR, {"a": 0.5, "r": 0.5})
M_MULTI    = DSVector.from_focal(FRAME_AHR, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3})
M_CONFLICT = DSVector.from_sparse(FRAME_AHR, {frozenset(): 1.0})


class TestBetP:

    def test_categorical(self):
        bp = M_CAT.to_betp()
        assert np.isclose(bp[0], 1.0)
        assert np.isclose(bp[1], 0.0)
        assert np.isclose(bp[2], 0.0)

    def test_vacuous_is_uniform(self):
        bp = M_VAC.to_betp()
        assert np.allclose(bp, [1/3, 1/3, 1/3])

    def test_known_result_singletons(self):
        # m({a})=0.5, m({r})=0.5
        bp = M_HALF.to_betp()
        assert np.isclose(bp[0], 0.5)  # a
        assert np.isclose(bp[1], 0.0)  # h
        assert np.isclose(bp[2], 0.5)  # r

    def test_known_result_multi_focal(self):
        # BetP({a}) = 0.3 + 0.4/2 + 0.3/3 = 0.6
        # BetP({h}) =        0.4/2 + 0.3/3 = 0.3
        # BetP({r}) =               0.3/3   = 0.1
        bp = M_MULTI.to_betp()
        assert np.isclose(bp[0], 0.6)
        assert np.isclose(bp[1], 0.3)
        assert np.isclose(bp[2], 0.1)

    def test_sums_to_one(self):
        for m in [M_CAT, M_VAC, M_HALF, M_MULTI]:
            assert np.isclose(m.to_betp().sum(), 1.0)

    def test_returns_ndarray_of_length_n(self):
        bp = M_HALF.to_betp()
        assert isinstance(bp, np.ndarray)
        assert len(bp) == len(FRAME_AHR)

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="contradictory|m\\(∅\\)"):
            M_CONFLICT.to_betp()

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            M_HALF.to_bel().to_betp()

    def test_standalone_matches_method(self):
        assert np.allclose(betp(M_MULTI.dense), M_MULTI.to_betp())


class TestPlP:

    def test_sums_to_one(self):
        for m in [M_CAT, M_VAC, M_HALF, M_MULTI]:
            assert np.isclose(m.to_plp().sum(), 1.0)

    def test_categorical(self):
        pp = M_CAT.to_plp()
        assert np.isclose(pp[0], 1.0)  # only {a} is focal
        assert np.isclose(pp[1], 0.0)
        assert np.isclose(pp[2], 0.0)

    def test_vacuous_is_uniform(self):
        # pl({x}) = 1 for all x when m = vacuous
        pp = M_VAC.to_plp()
        assert np.allclose(pp, [1/3, 1/3, 1/3])

    def test_returns_ndarray_of_length_n(self):
        pp = M_HALF.to_plp()
        assert isinstance(pp, np.ndarray)
        assert len(pp) == len(FRAME_AHR)

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            M_HALF.to_bel().to_plp()

    def test_standalone_matches_method(self):
        assert np.allclose(plp(M_MULTI.dense), M_MULTI.to_plp())

    def test_nonnegative(self):
        assert np.all(M_MULTI.to_plp() >= 0)
