"""
Tests for evtools.dsvector — DSVector and Kind.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FRAME2 = ["a", "b"]
FRAME3 = ["a", "b", "c"]

# m over {a,b}: m({a})=0.3, m({b})=0.5, m({a,b})=0.2
DENSE2 = np.array([0.0, 0.3, 0.5, 0.2])

# m over {a,b,c}: sparse, only a few focal elements
SPARSE3 = {
    frozenset({"a"}):       0.1,
    frozenset({"b", "c"}):  0.5,
    frozenset({"a","b","c"}): 0.4,
}


# ---------------------------------------------------------------------------
# Kind enum
# ---------------------------------------------------------------------------

def test_kind_values():
    assert Kind.M.value   == "m"
    assert Kind.BEL.value == "bel"
    assert Kind.PL.value  == "pl"
    assert Kind.B.value   == "b"
    assert Kind.Q.value   == "q"
    assert Kind.V.value   == "v"
    assert Kind.W.value   == "w"


# ---------------------------------------------------------------------------
# from_focal
# ---------------------------------------------------------------------------

def test_from_focal_basic():
    m = DSVector.from_focal(FRAME2, {"a": 0.3, "b": 0.5, "a,b": 0.2})
    assert np.isclose(m[frozenset({"a"})], 0.3)
    assert np.isclose(m[frozenset({"b"})], 0.5)
    assert np.isclose(m[frozenset({"a", "b"})], 0.2)
    assert m.kind == Kind.M


def test_from_focal_complete():
    # Only partial spec — remainder goes to Omega
    m = DSVector.from_focal(FRAME2, {"a": 0.3})
    assert np.isclose(m[frozenset({"a", "b"})], 0.7)
    assert np.isclose(sum(m.sparse.values()), 1.0)


def test_from_focal_no_complete():
    m = DSVector.from_focal(FRAME2, {"a": 0.3}, complete=False)
    assert np.isclose(m[frozenset({"a"})], 0.3)
    assert np.isclose(sum(m.sparse.values()), 0.3)


def test_from_focal_empty_set():
    m = DSVector.from_focal(FRAME2, {"": 0.1, "a": 0.3, "a,b": 0.6}, complete=False)
    assert np.isclose(m[frozenset()], 0.1)
    assert np.isclose(sum(m.sparse.values()), 1.0)


def test_from_focal_unknown_atom_raises():
    with pytest.raises(ValueError, match="not in the frame"):
        DSVector.from_focal(FRAME2, {"c": 0.5, "a,b": 0.5})


def test_from_focal_negative_mass_raises():
    with pytest.raises(ValueError, match="non-negative"):
        DSVector.from_focal(FRAME2, {"a": -0.1, "a,b": 1.1})


def test_from_focal_total_exceeds_one_raises():
    with pytest.raises(ValueError, match="exceeds 1"):
        DSVector.from_focal(FRAME2, {"a": 0.7, "b": 0.8}, complete=False)


# ---------------------------------------------------------------------------
# from_dense
# ---------------------------------------------------------------------------

def test_from_dense_basic():
    m = DSVector.from_dense(FRAME2, DENSE2)
    assert np.allclose(m.dense, DENSE2)
    assert m.kind == Kind.M


def test_from_dense_wrong_size_raises():
    with pytest.raises(ValueError, match="Array length"):
        DSVector.from_dense(FRAME2, np.array([0.3, 0.7]))


def test_from_dense_tol():
    array = np.array([1e-15, 0.3, 0.5, 0.2])
    m = DSVector.from_dense(FRAME2, array, tol=1e-12)
    assert frozenset() not in m.sparse   # dropped
    assert frozenset({"a"}) in m.sparse


def test_from_dense_caches_dense():
    m = DSVector.from_dense(FRAME2, DENSE2)
    d1 = m.dense
    d2 = m.dense
    assert np.allclose(d1, d2)


# ---------------------------------------------------------------------------
# from_sparse
# ---------------------------------------------------------------------------

def test_from_sparse_basic():
    m = DSVector.from_sparse(FRAME3, SPARSE3)
    assert np.isclose(m[frozenset({"a"})], 0.1)
    assert np.isclose(m[frozenset({"b", "c"})], 0.5)
    assert m.kind == Kind.M


def test_from_sparse_unknown_atom_raises():
    with pytest.raises(ValueError, match="not in the frame"):
        DSVector.from_sparse(FRAME2, {frozenset({"z"}): 0.5})


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_properties():
    m = DSVector.from_dense(FRAME2, DENSE2)
    assert m.frame == FRAME2
    assert m.n_atoms == 2
    assert m.n_focal == 3   # three non-zero entries


def test_sparse_dense_roundtrip():
    m = DSVector.from_sparse(FRAME3, SPARSE3)
    d = m.dense
    m2 = DSVector.from_dense(FRAME3, d)
    assert np.allclose(m2.dense, d)


# ---------------------------------------------------------------------------
# Access and iteration
# ---------------------------------------------------------------------------

def test_getitem_missing_returns_zero():
    m = DSVector.from_dense(FRAME2, DENSE2)
    assert m[frozenset()] == 0.0


def test_iter():
    m = DSVector.from_sparse(FRAME3, SPARSE3)
    items = dict(m)
    assert frozenset({"a"}) in items
    assert frozenset({"b", "c"}) in items


def test_len():
    m = DSVector.from_sparse(FRAME3, SPARSE3)
    assert len(m) == 3


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [Kind.BEL, Kind.PL, Kind.B, Kind.Q])
def test_to_and_back(target):
    m = DSVector.from_dense(FRAME2, DENSE2)
    converted = m.to(target)
    assert converted.kind == target
    back = converted.to(Kind.M)
    assert np.allclose(back.dense, DENSE2, atol=1e-10)


def test_to_same_kind_returns_copy():
    m = DSVector.from_dense(FRAME2, DENSE2)
    m2 = m.to(Kind.M)
    assert m2.kind == Kind.M
    assert np.allclose(m2.dense, m.dense)


def test_convenience_shortcuts():
    m = DSVector.from_dense(FRAME2, DENSE2)
    assert m.to_bel().kind == Kind.BEL
    assert m.to_pl().kind  == Kind.PL
    assert m.to_b().kind   == Kind.B
    assert m.to_q().kind   == Kind.Q


def test_subnormal_to_v_and_w():
    # v and w require b > 0 everywhere (subnormal BBA)
    m_sub = np.array([0.1, 0.3, 0.4, 0.2])
    m = DSVector.from_dense(FRAME2, m_sub)
    assert np.allclose(m.to_v().to_m().dense, m_sub, atol=1e-10)
    assert np.allclose(m.to_w().to_m().dense, m_sub, atol=1e-10)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_kind_and_frame():
    m = DSVector.from_dense(FRAME2, DENSE2)
    r = repr(m)
    assert "m" in r
    assert "a" in r
    assert "b" in r
