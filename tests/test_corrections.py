"""
Tests for evtools.corrections — discount, contextual_discount, theta_contextual_discount.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind
from evtools.corrections import discount, contextual_discount, theta_contextual_discount

FRAME_AHR = ["a", "h", "r"]

# m({a})=0.5, m({r})=0.5  (Example 1 of Mercier 2008)
M = DSVector.from_focal(FRAME_AHR, {"a": 0.5, "r": 0.5})


# ---------------------------------------------------------------------------
# discount — classical discounting
# ---------------------------------------------------------------------------

def test_discount_known_result():
    # beta=0.6: βm({a})=0.3, βm({r})=0.3, βm(Ω)=0.4
    md = discount(M, 0.6)
    assert np.isclose(md[frozenset({"a"})],          0.3)
    assert np.isclose(md[frozenset({"r"})],          0.3)
    assert np.isclose(md[frozenset({"a","h","r"})],  0.4)
    assert np.isclose(sum(md.sparse.values()),        1.0)


def test_discount_beta1_unchanged():
    assert np.allclose(discount(M, 1.0).dense, M.dense, atol=1e-10)


def test_discount_beta0_vacuous():
    vac = discount(M, 0.0)
    assert np.isclose(vac[frozenset({"a","h","r"})], 1.0)
    assert np.isclose(sum(vac.sparse.values()),       1.0)


def test_discount_kind_m():
    assert discount(M, 0.7).kind == Kind.M


def test_discount_beta_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        discount(M, 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        discount(M, -0.1)


def test_discount_wrong_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        discount(M.to_bel(), 0.3)


# ---------------------------------------------------------------------------
# contextual_discount — Ω-contextual discounting
# ---------------------------------------------------------------------------

def test_contextual_discount_case1_mercier2008():
    # Case 1 of Mercier (2008): β_a=0.6, β_h=1.0, β_r=1.0
    # βm({a})=0.5, βm({r})=0.3, βm({a,r})=0.2
    betas = {
        frozenset({"a"}): 0.6,
        frozenset({"h"}): 1.0,
        frozenset({"r"}): 1.0,
    }
    mcd = contextual_discount(M, betas)
    assert np.isclose(mcd[frozenset({"a"})],     0.5)
    assert np.isclose(mcd[frozenset({"r"})],     0.3)
    assert np.isclose(mcd[frozenset({"a","r"})], 0.2)
    assert np.isclose(sum(mcd.sparse.values()),  1.0)


def test_contextual_discount_all_reliable_unchanged():
    # All β=1 → no correction
    betas = {frozenset({x}): 1.0 for x in FRAME_AHR}
    assert np.allclose(contextual_discount(M, betas).dense, M.dense, atol=1e-10)


def test_contextual_discount_non_singleton_raises():
    with pytest.raises(ValueError, match="singletons"):
        contextual_discount(M, {
            frozenset({"a","h"}): 0.5,
            frozenset({"r"}):     1.0,
        })


def test_contextual_discount_kind_m():
    betas = {frozenset({x}): 0.8 for x in FRAME_AHR}
    assert contextual_discount(M, betas).kind == Kind.M


# ---------------------------------------------------------------------------
# theta_contextual_discount — general Θ-discounting
# ---------------------------------------------------------------------------

def test_theta_discount_reduces_to_classical():
    # Θ = {Ω}, single β → same as discount(m, beta)
    beta = 0.6
    omega = frozenset(FRAME_AHR)
    mt = theta_contextual_discount(M, {omega: beta})
    assert np.allclose(mt.dense, discount(M, beta).dense, atol=1e-10)


def test_theta_discount_reduces_to_contextual():
    # Θ = singletons → same as contextual_discount
    betas = {frozenset({x}): 0.7 for x in FRAME_AHR}
    mt = theta_contextual_discount(M, betas)
    mcd = contextual_discount(M, betas)
    assert np.allclose(mt.dense, mcd.dense, atol=1e-10)


def test_theta_discount_coarsening():
    # Θ = {{a}, {h,r}}
    betas = {
        frozenset({"a"}):      0.4,
        frozenset({"h","r"}):  0.9,
    }
    mt = theta_contextual_discount(M, betas)
    assert np.isclose(sum(mt.sparse.values()), 1.0)
    assert mt.kind == Kind.M


def test_theta_discount_all_reliable_unchanged():
    omega = frozenset(FRAME_AHR)
    mt = theta_contextual_discount(M, {omega: 1.0})
    assert np.allclose(mt.dense, M.dense, atol=1e-10)


# ---------------------------------------------------------------------------
# Partition validation
# ---------------------------------------------------------------------------

def test_partition_missing_atom_raises():
    with pytest.raises(ValueError, match="covered"):
        theta_contextual_discount(M, {frozenset({"a"}): 0.5})


def test_partition_overlap_raises():
    with pytest.raises(ValueError, match="disjoint"):
        theta_contextual_discount(M, {
            frozenset({"a"}):      0.5,
            frozenset({"a","h"}):  0.5,
            frozenset({"r"}):      1.0,
        })


def test_partition_unknown_atom_raises():
    with pytest.raises(ValueError, match="not in the frame"):
        theta_contextual_discount(M, {
            frozenset({"a","h","r","z"}): 0.5,
        })


def test_partition_empty_set_raises():
    with pytest.raises(ValueError, match="empty set"):
        theta_contextual_discount(M, {
            frozenset():           0.5,
            frozenset({"a","h","r"}): 0.5,
        })


def test_partition_beta_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        theta_contextual_discount(M, {
            frozenset({"a"}):      1.5,
            frozenset({"h","r"}):  0.8,
        })


# ===========================================================================
# is_valid
# ===========================================================================

def test_is_valid_valid_bba():
    assert M.is_valid

def test_is_valid_negative_mass():
    m_inv = DSVector.from_sparse(FRAME_AHR, {
        frozenset({"a"}): -0.1,
        frozenset({"a","h","r"}): 1.1,
    }, kind=Kind.M)
    assert not m_inv.is_valid

def test_is_valid_sum_not_one():
    m_inv = DSVector.from_sparse(FRAME_AHR, {frozenset({"a"}): 0.3}, kind=Kind.M)
    assert not m_inv.is_valid

def test_is_valid_other_kinds_always_true():
    assert M.to_bel().is_valid
    assert M.to_pl().is_valid


# ===========================================================================
# contextual_reinforce (CR)
# ===========================================================================

from evtools.corrections import contextual_reinforce

BETAS_SINGLETONS = {frozenset({x}): 0.7 for x in FRAME_AHR}

def test_cr_valid():
    assert contextual_reinforce(M, BETAS_SINGLETONS).is_valid

def test_cr_kind_m():
    assert contextual_reinforce(M, BETAS_SINGLETONS).kind == Kind.M

def test_cr_sums_to_one():
    assert np.isclose(sum(contextual_reinforce(M, BETAS_SINGLETONS).sparse.values()), 1.0)

def test_cr_all_reliable_unchanged():
    betas = {frozenset({x}): 1.0 for x in FRAME_AHR}
    assert np.allclose(contextual_reinforce(M, betas).dense, M.dense, atol=1e-10)

def test_cr_beta_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        contextual_reinforce(M, {frozenset({"a"}): 1.5})


# ===========================================================================
# contextual_dediscount (CdD) — inverse of CD
# ===========================================================================

from evtools.corrections import contextual_dediscount

BETAS_CD = {
    frozenset({"a"}): 0.6,
    frozenset({"h"}): 1.0,
    frozenset({"r"}): 1.0,
}

def test_cdd_inverts_cd():
    mcd = contextual_discount(M, BETAS_CD)
    mdd = contextual_dediscount(mcd, BETAS_CD)
    assert mdd.is_valid
    assert np.allclose(mdd.dense, M.dense, atol=1e-6)

def test_cdd_beta_zero_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        contextual_dediscount(M, {frozenset({"a"}): 0.0})


# ===========================================================================
# contextual_dereinforce (CdR) — inverse of CR
# ===========================================================================

from evtools.corrections import contextual_dereinforce

def test_cdr_inverts_cr():
    mcr = contextual_reinforce(M, BETAS_SINGLETONS)
    mdr = contextual_dereinforce(mcr, BETAS_SINGLETONS)
    assert mdr.is_valid
    assert np.allclose(mdr.dense, M.dense, atol=1e-6)

def test_cdr_beta_zero_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        contextual_dereinforce(M, {frozenset({"a"}): 0.0})


# ===========================================================================
# contextual_negate (CN)
# ===========================================================================

from evtools.corrections import contextual_negate

def test_cn_all_reliable_unchanged():
    betas = {frozenset({x}): 1.0 for x in FRAME_AHR}
    assert np.allclose(contextual_negate(M, betas).dense, M.dense, atol=1e-10)

def test_cn_pure_negation():
    # beta=0, A=∅ → full negation: m(B) → m(B̄)
    mcn = contextual_negate(M, {frozenset(): 0.0})
    assert np.isclose(mcn[frozenset({"h","r"})], 0.5)
    assert np.isclose(mcn[frozenset({"a","h"})], 0.5)

def test_cn_valid():
    assert contextual_negate(M, {frozenset({"a"}): 0.7}).is_valid

def test_cn_sums_to_one():
    assert np.isclose(
        sum(contextual_negate(M, {frozenset({"a"}): 0.7}).sparse.values()), 1.0
    )

def test_cn_beta_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        contextual_negate(M, {frozenset({"a"}): 1.5})
