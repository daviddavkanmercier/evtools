"""
Combination rules for belief functions in the Dempster-Shafer theory.

Available rules
---------------
crc(m1, m2, method="sparse")      — Conjunctive Rule of Combination (TBM)
dempster(m1, m2, method="sparse") — Dempster's normalized conjunctive rule
drc(m1, m2, method="sparse")      — Disjunctive Rule of Combination (TBM)
cautious(m1, m2)                  — Cautious conjunctive rule (nondistinct sources)
bold(m1, m2)                      — Bold disjunctive rule (nondistinct sources)

Operator shortcuts on DSVector
-------------------------------
m1 & m2   →  crc(m1, m2)
m1 @ m2   →  dempster(m1, m2)
m1 | m2   →  drc(m1, m2)

Choice of rule
--------------
                    All sources reliable    At least one reliable
Distinct sources         CRC / Dempster           DRC
Nondistinct sources      Cautious                 Bold

References
----------
- Smets, P. (1990). The combination of evidence in the transferable belief model.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 12(5), 447-458.
- Denoeux, T. (2008). Conjunctive and disjunctive combination of belief functions
  induced by nondistinct bodies of evidence. Artificial Intelligence, 172, 234-264.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

from .dsvector import DSVector, Kind
from .conversions import (
    mtoq, qtom,   # commonality  — used by CRC (dense)
    mtob, btom,   # implicability — used by DRC (dense)
    mtow, wtom,   # conjunctive weights — used by Cautious
    mtov, vtom,   # disjunctive weights — used by Bold
)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _check_compatible(m1: DSVector, m2: DSVector, rule: str) -> None:
    """Raise ValueError if m1 and m2 cannot be combined."""
    if m1.kind != Kind.M:
        raise ValueError(
            f"{rule}: first argument has kind '{m1.kind.value}', expected 'm'."
        )
    if m2.kind != Kind.M:
        raise ValueError(
            f"{rule}: second argument has kind '{m2.kind.value}', expected 'm'."
        )
    if m1.frame != m2.frame:
        raise ValueError(
            f"{rule}: frames do not match — {m1.frame} vs {m2.frame}."
        )


# ---------------------------------------------------------------------------
# Shared sparse helper
# ---------------------------------------------------------------------------

def _combine_sparse(
    m1: DSVector,
    m2: DSVector,
    op,          # frozenset operator: & for CRC, | for DRC
) -> DSVector:
    """
    Generic sparse combination: m12(A) = Σ_{op(B,C)=A} m1(B)·m2(C).

    op is the frozenset operation applied to each pair of focal elements:
    - frozenset.__and__  for CRC  (intersection)
    - frozenset.__or__   for DRC  (union)
    """
    result: dict[frozenset, float] = {}
    for b, mb in m1.sparse.items():
        for c, mc in m2.sparse.items():
            a = op(b, c)
            value = mb * mc
            result[a] = result.get(a, 0.0) + value
    result = {k: v for k, v in result.items() if abs(v) > 1e-15}
    return DSVector.from_sparse(m1.frame, result, kind=Kind.M)


# ---------------------------------------------------------------------------
# CRC — Conjunctive Rule of Combination
# ---------------------------------------------------------------------------

def crc(
    m1: DSVector,
    m2: DSVector,
    *,
    method: Literal["sparse", "dense"] = "sparse",
) -> DSVector:
    """
    Conjunctive Rule of Combination (CRC) — TBM unnormalized conjunctive rule.

    Combines two BBAs under the assumption that both sources are reliable
    and their bodies of evidence are distinct.

        m12(A) = Σ_{B∩C=A} m1(B)·m2(C),  ∀A ⊆ Ω.

    Equivalently via commonality functions:  q12 = q1·q2.

    Parameters
    ----------
    m1, m2 : DSVector
        Two BBAs (kind=Kind.M) on the same frame.
    method : {"sparse", "dense"}
        "sparse" (default): focal element enumeration, O(k1·k2).
        "dense": via Fast Möbius Transform on commonality, O(n·2^n).

    Returns
    -------
    DSVector
        Combined BBA in sparse representation.
    """
    _check_compatible(m1, m2, "crc")
    if method == "sparse":
        return _combine_sparse(m1, m2, frozenset.__and__)
    elif method == "dense":
        q12 = mtoq(m1.dense) * mtoq(m2.dense)
        return DSVector.from_dense(m1.frame, qtom(q12), kind=Kind.M)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sparse' or 'dense'.")


# ---------------------------------------------------------------------------
# Dempster's rule — normalized CRC
# ---------------------------------------------------------------------------

def dempster(
    m1: DSVector,
    m2: DSVector,
    *,
    method: Literal["sparse", "dense"] = "sparse",
) -> DSVector:
    """
    Dempster's rule of combination — normalized conjunctive rule.

    Equivalent to CRC followed by normalization: mass on ∅ is redistributed
    proportionally to all other focal elements.

    Parameters
    ----------
    m1, m2 : DSVector
        Two BBAs (kind=Kind.M) on the same frame.
    method : {"sparse", "dense"}
        Passed to the underlying CRC computation.

    Returns
    -------
    DSVector
        Normalized combined BBA (m(∅) = 0) in sparse representation.

    Raises
    ------
    ValueError
        If the two BBAs are fully contradictory (conflict = 1).
    """
    _check_compatible(m1, m2, "dempster")
    m12 = crc(m1, m2, method=method)
    conflict = m12[frozenset()]
    if np.isclose(conflict, 1.0):
        raise ValueError(
            "dempster: total conflict — BBAs are fully contradictory "
            "(m12(∅) = 1). Dempster's rule is undefined in this case."
        )
    k = 1.0 / (1.0 - conflict)
    normalized = {
        s: v * k for s, v in m12.sparse.items() if s != frozenset()
    }
    return DSVector.from_sparse(m1.frame, normalized, kind=Kind.M)


# ---------------------------------------------------------------------------
# DRC — Disjunctive Rule of Combination
# ---------------------------------------------------------------------------

def drc(
    m1: DSVector,
    m2: DSVector,
    *,
    method: Literal["sparse", "dense"] = "sparse",
) -> DSVector:
    """
    Disjunctive Rule of Combination (DRC) — TBM disjunctive rule.

    Combines two BBAs under the assumption that at least one source is
    reliable, but we do not know which one.

        m12(A) = Σ_{B∪C=A} m1(B)·m2(C),  ∀A ⊆ Ω.

    Equivalently via implicability functions:  b12 = b1·b2.

    Parameters
    ----------
    m1, m2 : DSVector
        Two BBAs (kind=Kind.M) on the same frame.
    method : {"sparse", "dense"}
        "sparse" (default): focal element enumeration, O(k1·k2).
        "dense": via Fast Möbius Transform on implicability, O(n·2^n).

    Returns
    -------
    DSVector
        Combined BBA in sparse representation.
    """
    _check_compatible(m1, m2, "drc")
    if method == "sparse":
        return _combine_sparse(m1, m2, frozenset.__or__)
    elif method == "dense":
        b12 = mtob(m1.dense) * mtob(m2.dense)
        return DSVector.from_dense(m1.frame, btom(b12), kind=Kind.M)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sparse' or 'dense'.")


# ---------------------------------------------------------------------------
# Cautious rule — nondistinct reliable sources
# ---------------------------------------------------------------------------

def cautious(m1: DSVector, m2: DSVector) -> DSVector:
    """
    Cautious conjunctive rule (Denoeux 2008).

    For nondogmatic BBAs from reliable but possibly overlapping sources.
    Defined via conjunctive weights (Kind.W):

        w12(A) = min(w1(A), w2(A)),  ∀A ⊂ Ω.

    Properties: commutative, associative, idempotent.

    Parameters
    ----------
    m1, m2 : DSVector
        Two nondogmatic BBAs (m(Ω) > 0) on the same frame.

    Returns
    -------
    DSVector
        Combined BBA in sparse representation.

    Raises
    ------
    ValueError
        If m1 or m2 is dogmatic (m(Ω) = 0).
    """
    _check_compatible(m1, m2, "cautious")
    omega = frozenset(m1.frame)
    if np.isclose(m1[omega], 0.0):
        raise ValueError("cautious: m1 is dogmatic (m(Ω) = 0).")
    if np.isclose(m2[omega], 0.0):
        raise ValueError("cautious: m2 is dogmatic (m(Ω) = 0).")
    w12 = np.minimum(mtow(m1.dense), mtow(m2.dense))
    return DSVector.from_dense(m1.frame, wtom(w12), kind=Kind.M)


# ---------------------------------------------------------------------------
# Bold rule — nondistinct possibly unreliable sources
# ---------------------------------------------------------------------------

def bold(m1: DSVector, m2: DSVector) -> DSVector:
    """
    Bold disjunctive rule (Denoeux 2008).

    For subnormal BBAs from possibly overlapping sources where at least
    one is reliable. Defined via disjunctive weights (Kind.V):

        v12(A) = min(v1(A), v2(A)),  ∀A ≠ ∅.

    Properties: commutative, associative, idempotent.

    Parameters
    ----------
    m1, m2 : DSVector
        Two subnormal BBAs (m(∅) > 0) on the same frame.

    Returns
    -------
    DSVector
        Combined BBA in sparse representation.

    Raises
    ------
    ValueError
        If m1 or m2 is normal (m(∅) = 0).
    """
    _check_compatible(m1, m2, "bold")
    if np.isclose(m1[frozenset()], 0.0):
        raise ValueError("bold: m1 is normal (m(∅) = 0). Bold rule requires subnormal BBAs.")
    if np.isclose(m2[frozenset()], 0.0):
        raise ValueError("bold: m2 is normal (m(∅) = 0). Bold rule requires subnormal BBAs.")
    v12 = np.minimum(mtov(m1.dense), mtov(m2.dense))
    return DSVector.from_dense(m1.frame, vtom(v12), kind=Kind.M)
