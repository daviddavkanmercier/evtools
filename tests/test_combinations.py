"""
Tests for evtools.combinations — CRC.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind
from evtools.combinations import crc


FRAME2 = ["a", "b"]
FRAME3 = ["a", "b", "c"]

# m1: m({a})=0.3, m({a,b})=0.7
M1 = DSVector.from_focal(FRAME2, {"a": 0.3, "a,b": 0.7})

# m2: m({a})=0.5, m({b})=0.3, m({a,b})=0.2
M2 = DSVector.from_focal(FRAME2, {"a": 0.5, "b": 0.3, "a,b": 0.2})

# Vacuous BBA
VAC = DSVector.from_focal(FRAME2, {})  # all mass on Ω

# Categorical BBA: m({a}) = 1
CAT_A = DSVector.from_focal(FRAME2, {"a": 1.0})
CAT_B = DSVector.from_focal(FRAME2, {"b": 1.0})


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_crc_sparse_known_result():
    # m12(A) = Σ_{B∩C=A} m1(B)·m2(C)
    # B={a}, C={a}  → A={a}:   0.3*0.5 = 0.15
    # B={a}, C={b}  → A=∅:     0.3*0.3 = 0.09
    # B={a}, C={a,b}→ A={a}:   0.3*0.2 = 0.06   → {a} total = 0.21
    # B={a,b},C={a} → A={a}:   0.7*0.5 = 0.35
    # B={a,b},C={b} → A={b}:   0.7*0.3 = 0.21
    # B={a,b},C={a,b}→A={a,b}: 0.7*0.2 = 0.14
    m12 = crc(M1, M2)
    assert np.isclose(m12[frozenset()],        0.09)
    assert np.isclose(m12[frozenset({"a"})],   0.15 + 0.06 + 0.35)
    assert np.isclose(m12[frozenset({"b"})],   0.21)
    assert np.isclose(m12[frozenset({"a","b"})], 0.14)
    assert np.isclose(sum(m12.sparse.values()), 1.0)


def test_crc_dense_matches_sparse():
    m12_sparse = crc(M1, M2, method="sparse")
    m12_dense  = crc(M1, M2, method="dense")
    assert np.allclose(m12_sparse.dense, m12_dense.dense, atol=1e-10)


def test_crc_sparse_matches_dense_3atom():
    m1 = DSVector.from_focal(FRAME3, {"a": 0.3, "b,c": 0.4, "a,b,c": 0.3})
    m2 = DSVector.from_focal(FRAME3, {"b": 0.2, "a,b": 0.5, "a,b,c": 0.3})
    m12_s = crc(m1, m2, method="sparse")
    m12_d = crc(m1, m2, method="dense")
    assert np.allclose(m12_s.dense, m12_d.dense, atol=1e-10)


# ---------------------------------------------------------------------------
# Algebraic properties
# ---------------------------------------------------------------------------

def test_crc_commutativity():
    assert np.allclose(crc(M1, M2).dense, crc(M2, M1).dense, atol=1e-10)


def test_crc_associativity():
    m3 = DSVector.from_focal(FRAME2, {"b": 0.4, "a,b": 0.6})
    lhs = crc(crc(M1, M2), m3)
    rhs = crc(M1, crc(M2, m3))
    assert np.allclose(lhs.dense, rhs.dense, atol=1e-10)


def test_crc_neutral_element_vacuous():
    # vacuous BBA is the neutral element
    assert np.allclose(crc(M1, VAC).dense, M1.dense, atol=1e-10)
    assert np.allclose(crc(VAC, M1).dense, M1.dense, atol=1e-10)


def test_crc_categorical_gives_empty_set():
    # m({a}) ∩ m({b}) → all mass on ∅ (full conflict)
    m12 = crc(CAT_A, CAT_B)
    assert np.isclose(m12[frozenset()], 1.0)


def test_crc_result_sums_to_one():
    m12 = crc(M1, M2)
    assert np.isclose(sum(m12.sparse.values()), 1.0)


def test_crc_result_kind_is_m():
    assert crc(M1, M2).kind == Kind.M


def test_crc_result_is_sparse():
    m12 = crc(M1, M2)
    assert isinstance(m12.sparse, dict)


# ---------------------------------------------------------------------------
# Operator &
# ---------------------------------------------------------------------------

def test_operator_and_matches_crc():
    assert np.allclose((M1 & M2).dense, crc(M1, M2).dense, atol=1e-10)


def test_operator_and_commutativity():
    assert np.allclose((M1 & M2).dense, (M2 & M1).dense, atol=1e-10)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_crc_wrong_kind_raises():
    bel = M1.to_bel()
    with pytest.raises(ValueError, match="kind"):
        crc(bel, M2)


def test_crc_mismatched_frames_raises():
    m_other = DSVector.from_focal(FRAME3, {"a": 0.5, "a,b,c": 0.5})
    with pytest.raises(ValueError, match="frames"):
        crc(M1, m_other)


def test_crc_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        crc(M1, M2, method="fmt")


# ===========================================================================
# Dempster's rule
# ===========================================================================

from evtools.combinations import dempster

def test_dempster_normalizes():
    m12 = dempster(M1, M2)
    assert np.isclose(m12[frozenset()], 0.0)
    assert np.isclose(sum(m12.sparse.values()), 1.0)


def test_dempster_matches_crc_normalized():
    m12_crc = crc(M1, M2)
    conflict = m12_crc[frozenset()]
    k = 1.0 / (1.0 - conflict)
    m12_d = dempster(M1, M2)
    for subset, value in m12_crc.sparse.items():
        if subset != frozenset():
            assert np.isclose(m12_d[subset], value * k, atol=1e-10)


def test_dempster_total_conflict_raises():
    with pytest.raises(ValueError, match="fully contradictory"):
        dempster(CAT_A, CAT_B)


def test_dempster_operator():
    assert np.allclose((M1 @ M2).dense, dempster(M1, M2).dense, atol=1e-10)


def test_dempster_commutativity():
    assert np.allclose(dempster(M1, M2).dense, dempster(M2, M1).dense, atol=1e-10)


def test_dempster_sparse_dense_match():
    assert np.allclose(
        dempster(M1, M2, method="sparse").dense,
        dempster(M1, M2, method="dense").dense,
        atol=1e-10,
    )


# ===========================================================================
# DRC
# ===========================================================================

from evtools.combinations import drc

def test_drc_sparse_known_result():
    # m12(A) = Σ_{B∪C=A} m1(B)·m2(C)
    # B={a},   C={a}   → A={a}:   0.3*0.5=0.15
    # B={a},   C={b}   → A={a,b}: 0.3*0.3=0.09
    # B={a},   C={a,b} → A={a,b}: 0.3*0.2=0.06
    # B={a,b}, C={a}   → A={a,b}: 0.7*0.5=0.35
    # B={a,b}, C={b}   → A={a,b}: 0.7*0.3=0.21
    # B={a,b}, C={a,b} → A={a,b}: 0.7*0.2=0.14
    m12 = drc(M1, M2)
    assert np.isclose(m12[frozenset({"a"})],     0.15)
    assert np.isclose(m12[frozenset({"a", "b"})], 0.09 + 0.06 + 0.35 + 0.21 + 0.14)
    assert np.isclose(sum(m12.sparse.values()), 1.0)


def test_drc_sparse_dense_match():
    assert np.allclose(
        drc(M1, M2, method="sparse").dense,
        drc(M1, M2, method="dense").dense,
        atol=1e-10,
    )


def test_drc_commutativity():
    assert np.allclose(drc(M1, M2).dense, drc(M2, M1).dense, atol=1e-10)


def test_drc_associativity():
    m3 = DSVector.from_focal(FRAME2, {"b": 0.4, "a,b": 0.6})
    assert np.allclose(
        drc(drc(M1, M2), m3).dense,
        drc(M1, drc(M2, m3)).dense,
        atol=1e-10,
    )


def test_drc_operator():
    assert np.allclose((M1 | M2).dense, drc(M1, M2).dense, atol=1e-10)


def test_drc_result_sums_to_one():
    assert np.isclose(sum(drc(M1, M2).sparse.values()), 1.0)


# ===========================================================================
# Cautious rule
# ===========================================================================

from evtools.combinations import cautious

# Nondogmatic BBAs (m(Ω) > 0)
M1_ND = DSVector.from_focal(FRAME2, {"a": 0.3, "b": 0.3, "a,b": 0.4})
M2_ND = DSVector.from_focal(FRAME2, {"a": 0.4, "b": 0.2, "a,b": 0.4})


def test_cautious_commutativity():
    assert np.allclose(cautious(M1_ND, M2_ND).dense, cautious(M2_ND, M1_ND).dense, atol=1e-10)


def test_cautious_idempotent():
    assert np.allclose(cautious(M1_ND, M1_ND).dense, M1_ND.dense, atol=1e-10)


def test_cautious_associativity():
    M3_ND = DSVector.from_focal(FRAME2, {"a": 0.2, "b": 0.5, "a,b": 0.3})
    assert np.allclose(
        cautious(cautious(M1_ND, M2_ND), M3_ND).dense,
        cautious(M1_ND, cautious(M2_ND, M3_ND)).dense,
        atol=1e-10,
    )


def test_cautious_dogmatic_raises():
    m_dog = DSVector.from_focal(FRAME2, {"a": 0.6, "b": 0.4}, complete=False)
    with pytest.raises(ValueError, match="dogmatic"):
        cautious(m_dog, M2_ND)


def test_cautious_result_kind_m():
    assert cautious(M1_ND, M2_ND).kind == Kind.M


# ===========================================================================
# Bold rule
# ===========================================================================

from evtools.combinations import bold

# Subnormal BBAs (m(∅) > 0)
M1_SUB = DSVector.from_focal(FRAME2, {"": 0.1, "a": 0.4, "a,b": 0.5}, complete=False)
M2_SUB = DSVector.from_focal(FRAME2, {"": 0.2, "b": 0.3, "a,b": 0.5}, complete=False)


def test_bold_commutativity():
    assert np.allclose(bold(M1_SUB, M2_SUB).dense, bold(M2_SUB, M1_SUB).dense, atol=1e-10)


def test_bold_idempotent():
    assert np.allclose(bold(M1_SUB, M1_SUB).dense, M1_SUB.dense, atol=1e-10)


def test_bold_normal_raises():
    with pytest.raises(ValueError, match="normal"):
        bold(M1, M2_SUB)


def test_bold_result_kind_m():
    assert bold(M1_SUB, M2_SUB).kind == Kind.M


def test_bold_result_sums_to_one():
    assert np.isclose(sum(bold(M1_SUB, M2_SUB).sparse.values()), 1.0)
