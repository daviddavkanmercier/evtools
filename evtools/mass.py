"""
Utilities for constructing mass functions (basic belief assignments).

The main entry point is :func:`mass`, which lets you define a BBA by naming
focal elements in plain text rather than by their binary index.

Example
-------
>>> from evtools.mass import mass
>>> m = mass(["a", "b"], {"a": 0.1})
>>> m   # remaining 0.9 goes to {a,b} automatically
array([0. , 0.1, 0. , 0.9])
"""

from __future__ import annotations
import numpy as np


def _focal_index(atoms: list[str], frame: list[str]) -> int:
    idx = 0
    for atom in atoms:
        atom = atom.strip()
        if atom not in frame:
            raise ValueError(
                f"Atom '{atom}' is not in the frame {frame}. "
                "Check spelling or add it to the frame."
            )
        idx += 2 ** frame.index(atom)
    return idx


def mass(
    frame: list[str],
    focal: dict[str, float],
    *,
    sep: str = ",",
    complete: bool = True,
) -> np.ndarray:
    """Build a mass function vector from a human-readable focal element dict.

    Parameters
    ----------
    frame:
        Ordered list of atoms of the frame of discernment, e.g. ["a", "b", "c"].
    focal:
        Mapping from focal element description to mass value.
        Each key is a sep-separated string of atom names, e.g. {"a": 0.3, "b,c": 0.5}.
        Use "" to assign mass to the empty set.
    sep:
        Separator used in focal element keys (default ",").
    complete:
        If True (default), any missing mass (1 - sum) is automatically assigned
        to Omega (the full frame), representing pure ignorance for the remainder.
        If False, masses are used as-is (useful for subnormal BBAs).

    Returns
    -------
    np.ndarray
        Mass vector of length 2 ** len(frame).

    Raises
    ------
    ValueError
        If an atom is not in frame, any mass is negative, or total exceeds 1.

    Examples
    --------
    Partial — remainder goes to Omega automatically:

    >>> m = mass(["a", "b"], {"a": 0.1})
    >>> m  # m({a,b}) = 0.9
    array([0. , 0.1, 0. , 0.9])

    Fully specified:

    >>> m = mass(["a", "b"], {"a": 0.1, "a,b": 0.9})
    >>> m
    array([0. , 0.1, 0. , 0.9])

    Subnormal BBA — disable auto-completion:

    >>> m = mass(["a", "b"], {"": 0.1, "a": 0.3, "a,b": 0.6}, complete=False)
    >>> m[0]
    0.1
    """
    n = len(frame)
    m = np.zeros(2 ** n)
    omega_idx = 2 ** n - 1

    for key, value in focal.items():
        if value < 0:
            raise ValueError(f"Mass values must be non-negative, got {value} for '{key}'.")
        if key.strip() == "":
            idx = 0
        else:
            atoms = [a.strip() for a in key.split(sep)]
            idx = _focal_index(atoms, frame)
        m[idx] += value

    total = m.sum()
    if total > 1.0 + 1e-12:
        raise ValueError(f"Total mass {total:.6g} exceeds 1. Check your focal element values.")

    if complete:
        remainder = 1.0 - total
        if remainder > 1e-12:
            m[omega_idx] += remainder

    return m


def frame_labels(frame: list[str], sep: str = ",") -> list[str]:
    """Return human-readable labels for all 2^n subsets of frame.

    Example
    -------
    >>> frame_labels(["a", "b"])
    ['∅', 'a', 'b', 'a,b']
    """
    n = len(frame)
    labels = []
    for i in range(2 ** n):
        members = [frame[bit] for bit in range(n) if i & (1 << bit)]
        labels.append(sep.join(members) if members else "∅")
    return labels
