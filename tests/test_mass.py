"""Tests for evtools.mass"""

import numpy as np
import pytest
from evtools.mass import mass, frame_labels


def test_basic_2atom():
    m = mass(["a", "b"], {"a": 0.1, "a,b": 0.9})
    expected = np.array([0.0, 0.1, 0.0, 0.9])
    assert np.allclose(m, expected)


def test_basic_3atom():
    m = mass(["a", "b", "c"], {"b": 0.5, "a,c": 0.3, "a,b,c": 0.2})
    assert np.isclose(m.sum(), 1.0)
    assert np.isclose(m[2], 0.5)   # {b} → index 2
    assert np.isclose(m[5], 0.3)   # {a,c} → index 1+4=5
    assert np.isclose(m[7], 0.2)   # {a,b,c} → index 7


def test_empty_set():
    m = mass(["a", "b"], {"": 0.1, "a": 0.3, "a,b": 0.6})
    assert np.isclose(m[0], 0.1)
    assert np.isclose(m.sum(), 1.0)


def test_complete():
    # Partial spec: remainder goes to Omega automatically
    m = mass(["a", "b"], {"a": 0.3})
    assert np.isclose(m[1], 0.3)       # m({a}) = 0.3
    assert np.isclose(m[3], 0.7)       # m({a,b}) = 0.7
    assert np.isclose(m.sum(), 1.0)

    # complete=False: subnormal BBA, no auto-completion
    m = mass(["a", "b"], {"a": 0.3}, complete=False)
    assert np.isclose(m[1], 0.3)
    assert np.isclose(m[3], 0.0)       # Omega untouched
    assert np.isclose(m.sum(), 0.3)


def test_unknown_atom_raises():
    with pytest.raises(ValueError, match="not in the frame"):
        mass(["a", "b"], {"c": 0.5, "a,b": 0.5})


def test_negative_mass_raises():
    with pytest.raises(ValueError, match="non-negative"):
        mass(["a", "b"], {"a": -0.1, "a,b": 1.1})


def test_frame_labels_2atom():
    assert frame_labels(["a", "b"]) == ["∅", "a", "b", "a,b"]


def test_frame_labels_3atom():
    labels = frame_labels(["a", "b", "c"])
    assert labels[0] == "∅"
    assert labels[7] == "a,b,c"
    assert len(labels) == 8