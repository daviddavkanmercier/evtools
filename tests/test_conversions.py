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
