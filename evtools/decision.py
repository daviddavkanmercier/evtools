"""
Decision-making functions for belief functions in the Dempster-Shafer theory.

Two families of decision criteria are provided:

Complete preference relations (precise assignment)
--------------------------------------------------
These criteria define a total order among acts and return the optimal act
as a tuple (index, atom_name).

    maximin(m, U)              — pessimistic: maximize lower expected utility
    maximax(m, U)              — optimistic:  maximize upper expected utility
    pignistic_decision(m, U)   — MEU with BetP (Smets pignistic)
    plp_decision(m, U)         — MEU with PlP  (Cobb & Shenoy plausibility-prob.)
    probability_decision(m, U, transform=...)
                               — generic MEU with an arbitrary m → probability transform
    hurwicz(m, U, alpha)       — convex combination of maximin and maximax

Partial preference relations (partial decision)
-----------------------------------------------
These criteria define a partial order and return the set of non-dominated
atoms as a frozenset of strings.

    strong_dominance(m)      — ω ≻_sd ω' ⟺ Bel({ω}) ≥ Pl({ω'})
    weak_dominance(m)        — ω ≻_wd ω' ⟺ Bel({ω}) ≥ Bel({ω'}) and Pl({ω}) ≥ Pl({ω'})

Note
----
Performance metrics for evaluating decisions (u65, u80, ...) live in
:mod:`evtools.metrics`, not here.

Utility matrix
--------------
U is a numpy array of shape (n, n), where U[i, j] = u(a_i, ω_j) is the
utility of taking act a_i when the true state is ω_j. Acts are indexed in
the same order as the frame atoms. The default (0-1 utility, identity matrix)
corresponds to the standard classification setting.

Return values
-------------
- Complete criteria return (index: int, atom: str) — the optimal act.
- Partial criteria return frozenset[str] — the non-dominated atoms.

References
----------
- Smets, P., Kennes, R. (1994). The transferable belief model.
  Artificial Intelligence, 66(2), 191-234.
- Strat, T.M. (1990). Decision analysis using belief functions.
  IJAR, 4(5-6), 391-417.
- Troffaes, M.C.M. (2007). Decision making under uncertainty using imprecise
  probabilities. IJAR, 45(1), 17-29.
- Ma, L., Denœux, T. (2021). Partial classification in the belief function
  framework. Knowledge-Based Systems, 214, 106742.
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2019). On learning
  evidential contextual corrections from soft labels using a measure of
  discrepancy between contour functions. SUM 2019.
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2021). Improving an
  evidential source of information using contextual corrections depending
  on partial decisions. BELIEF 2021, pp. 247-256.
- Mutmainah, S. (2021). Learning to adjust an evidential source of information
  using partially labeled data and partial decisions. PhD thesis, Université
  d'Artois. Section 1.4.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dsvector import DSVector, Kind
from .constants import VALID_TOL


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_bba(m: DSVector, fn: str) -> None:
    """Raise ValueError if m is not a BBA."""
    if m.kind != Kind.M:
        raise ValueError(
            f"{fn}: argument has kind '{m.kind.value}', expected 'm'."
        )


def _check_utility(U: np.ndarray, n: int, fn: str) -> None:
    """Raise ValueError if U is not a valid n×n utility matrix."""
    if U.shape != (n, n):
        raise ValueError(
            f"{fn}: utility matrix must have shape ({n}, {n}), "
            f"got {U.shape}."
        )


def _default_utility(n: int) -> np.ndarray:
    """Return the n×n identity utility matrix (0-1 losses)."""
    return np.eye(n)


# ---------------------------------------------------------------------------
# Lower and upper expected utilities
# ---------------------------------------------------------------------------

def _lower_expected(m: DSVector, U: np.ndarray) -> np.ndarray:
    """
    Compute lower expected utilities for all acts.

    E_-(a_i) = Σ_{B⊆Ω} m(B) · min_{ω_j ∈ B} U[i, j]

    Returns a vector of length n.
    """
    n = len(m.frame)
    result = np.zeros(n)
    for subset, mass in m.sparse.items():
        if not subset:
            continue  # ∅ contributes 0 (min over empty set undefined)
        indices = [m.frame.index(atom) for atom in subset]
        for i in range(n):
            result[i] += mass * U[i, indices].min()
    return result


def _upper_expected(m: DSVector, U: np.ndarray) -> np.ndarray:
    """
    Compute upper expected utilities for all acts.

    E^+(a_i) = Σ_{B⊆Ω} m(B) · max_{ω_j ∈ B} U[i, j]

    Returns a vector of length n.
    """
    n = len(m.frame)
    result = np.zeros(n)
    for subset, mass in m.sparse.items():
        if not subset:
            continue
        indices = [m.frame.index(atom) for atom in subset]
        for i in range(n):
            result[i] += mass * U[i, indices].max()
    return result


# ---------------------------------------------------------------------------
# Complete preference relations
# ---------------------------------------------------------------------------

def maximin(
    m: DSVector,
    U: np.ndarray | None = None,
) -> tuple[int, str]:
    """
    Maximin decision criterion — pessimistic attitude.

    Selects the act that maximizes the lower expected utility:

        E_-(a_i) = Σ_{B⊆Ω} m(B) · min_{ω_j ∈ B} U[i, j]

    This criterion considers the worst possible consequence within each
    focal set, reflecting a pessimistic (cautious) decision-maker attitude.

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
    U : np.ndarray, optional
        Utility matrix of shape (n, n), where U[i, j] = u(a_i, ω_j).
        Defaults to the identity matrix (0-1 utilities).

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act, where index is the position in
        the frame and atom is the atom name.

    References
    ----------
    Strat, T.M. (1990). IJAR, 4(5-6), 391-417.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Eq. (13).
    Mutmainah, S. (thesis). Eq. (1.21) and (1.23).
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    _check_bba(m, "maximin")
    n = len(m.frame)
    if U is None:
        U = _default_utility(n)
    _check_utility(U, n, "maximin")

    scores = _lower_expected(m, U)
    idx = int(np.argmax(scores))
    return idx, m.frame[idx]


def maximax(
    m: DSVector,
    U: np.ndarray | None = None,
) -> tuple[int, str]:
    """
    Maximax decision criterion — optimistic attitude.

    Selects the act that maximizes the upper expected utility:

        E^+(a_i) = Σ_{B⊆Ω} m(B) · max_{ω_j ∈ B} U[i, j]

    This criterion considers the best possible consequence within each
    focal set, reflecting an optimistic (ambiguity-seeking) attitude.

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
    U : np.ndarray, optional
        Utility matrix of shape (n, n). Defaults to identity.

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act.

    References
    ----------
    Strat, T.M. (1990). IJAR, 4(5-6), 391-417.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Eq. (14).
    Mutmainah, S. (thesis). Eq. (1.22) and (1.24).
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    _check_bba(m, "maximax")
    n = len(m.frame)
    if U is None:
        U = _default_utility(n)
    _check_utility(U, n, "maximax")

    scores = _upper_expected(m, U)
    idx = int(np.argmax(scores))
    return idx, m.frame[idx]


def probability_decision(
    m: DSVector,
    U: np.ndarray | None = None,
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[int, str]:
    """
    Generic MEU decision under a probability transformation of *m*.

    Selects the act that maximizes the expected utility weighted by a
    probability vector p obtained from m:

        E_p(a_i) = Σ_j p_j · U[i, j],   p = transform(m.dense)

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
    U : np.ndarray, optional
        Utility matrix of shape (n, n). Defaults to identity.
    transform : callable, optional
        A function np.ndarray → np.ndarray mapping the dense BBA (length 2^n)
        to a probability vector of length n. Defaults to PlP
        (Cobb & Shenoy 2006). Common choices: ``evtools.conversions.plp``,
        ``evtools.conversions.betp``.

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act.

    Raises
    ------
    ValueError
        If the transform raises (e.g. BetP on a fully contradictory BBA),
        or if the transform's output length does not match the frame size.

    References
    ----------
    Smets, P., Kennes, R. (1994). AI, 66(2), 191-234.
    Cobb, B.R., Shenoy, P.P. (2006). IJAR, 41(3), 314-330.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Eq. (15).
    """
    _check_bba(m, "probability_decision")
    n = len(m.frame)
    if U is None:
        U = _default_utility(n)
    _check_utility(U, n, "probability_decision")

    if transform is None:
        from .conversions import plp
        transform = plp

    p = transform(m.dense)
    if p.shape != (n,):
        raise ValueError(
            f"probability_decision: transform must return a vector of length "
            f"{n} (got shape {p.shape})."
        )
    scores = U @ p
    idx = int(np.argmax(scores))
    return idx, m.frame[idx]


def pignistic_decision(
    m: DSVector,
    U: np.ndarray | None = None,
) -> tuple[int, str]:
    """
    Pignistic decision criterion — MEU with BetP.

    Convenience wrapper around :func:`probability_decision` with
    ``transform=betp``. With 0-1 utilities (identity U), reduces to
    selecting the atom of maximum pignistic probability.

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
        Must not be fully contradictory (m(∅) < 1).
    U : np.ndarray, optional
        Utility matrix of shape (n, n). Defaults to identity.

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act.

    Raises
    ------
    ValueError
        If m(∅) = 1 (BetP undefined).

    References
    ----------
    Smets, P., Kennes, R. (1994). AI, 66(2), 191-234.
    Smets, P. (2005). IJAR, 38(2), 133-147.
    """
    from .conversions import betp
    return probability_decision(m, U, transform=betp)


def plp_decision(
    m: DSVector,
    U: np.ndarray | None = None,
) -> tuple[int, str]:
    """
    Plausibility-probability decision criterion — MEU with PlP.

    Convenience wrapper around :func:`probability_decision` with
    ``transform=plp`` (Cobb & Shenoy 2006). With 0-1 utilities (identity U),
    reduces to selecting the atom of maximum plausibility-probability.

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
        Must have non-zero singleton plausibilities.
    U : np.ndarray, optional
        Utility matrix of shape (n, n). Defaults to identity.

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act.

    Raises
    ------
    ValueError
        If all singleton plausibilities are zero (PlP undefined).

    References
    ----------
    Cobb, B.R., Shenoy, P.P. (2006). On the plausibility transformation
    method for translating belief function models to probability models.
    IJAR, 41(3), 314-330.
    """
    from .conversions import plp
    return probability_decision(m, U, transform=plp)


def hurwicz(
    m: DSVector,
    U: np.ndarray | None = None,
    *,
    alpha: float = 0.5,
) -> tuple[int, str]:
    """
    Hurwicz decision criterion — convex combination of maximin and maximax.

    Selects the act that maximizes:

        E_α(a_i) = α · E_-(a_i) + (1-α) · E^+(a_i)

    Special cases:
    - α = 1 → maximin (fully pessimistic)
    - α = 0 → maximax (fully optimistic)
    - α = 0.5 → equal weight to lower and upper (default)

    Parameters
    ----------
    m : DSVector
        The BBA representing uncertainty (kind=Kind.M).
    U : np.ndarray, optional
        Utility matrix of shape (n, n). Defaults to identity.
    alpha : float
        Pessimism index α ∈ [0, 1]. Default: 0.5.

    Returns
    -------
    tuple[int, str]
        (index, atom) of the optimal act.

    Raises
    ------
    ValueError
        If alpha is not in [0, 1].

    References
    ----------
    Strat, T.M. (1990). IJAR, 4(5-6), 391-417.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Eq. (16).
    """
    _check_bba(m, "hurwicz")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"hurwicz: alpha must be in [0, 1], got {alpha}.")
    n = len(m.frame)
    if U is None:
        U = _default_utility(n)
    _check_utility(U, n, "hurwicz")

    scores = alpha * _lower_expected(m, U) + (1.0 - alpha) * _upper_expected(m, U)
    idx = int(np.argmax(scores))
    return idx, m.frame[idx]


# ---------------------------------------------------------------------------
# Partial preference relations
# ---------------------------------------------------------------------------

def strong_dominance(m: DSVector) -> frozenset:
    """
    Strong dominance — partial decision via interval dominance.

    Returns the set of non-dominated atoms under the strong dominance
    relation (Troffaes 2007):

        ω ≻_sd ω' ⟺ Bel({ω}) ≥ Pl({ω'})

    An atom ω is non-dominated if no other atom ω' satisfies
    Bel({ω'}) ≥ Pl({ω}).

    With 0-1 utilities: E_-(ω) = Bel({ω}) and E^+(ω) = Pl({ω}),
    so strong dominance compares Bel of the challenger to Pl of ω.

    Note: strong dominance implies weak dominance. The set of non-dominated
    elements under weak dominance is a subset of those under strong dominance.

    Parameters
    ----------
    m : DSVector
        The BBA (kind=Kind.M).

    Returns
    -------
    frozenset[str]
        The set of non-dominated atoms. May contain a single atom
        (precise decision) or multiple atoms (partial decision).

    References
    ----------
    Troffaes, M.C.M. (2007). IJAR, 45(1), 17-29.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Section 3.4.
    Mutmainah, S. (thesis). Eqs. (1.25) and (1.29).
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    _check_bba(m, "strong_dominance")

    n = len(m.frame)
    # Bel({ω_k}) = m({ω_k})  and  Pl({ω_k}) = contour(ω_k)
    m_dense = m.dense
    bel_s = np.array([m_dense[1 << k] for k in range(n)])
    pl_s  = m.contour()

    # ω is non-dominated if no ω' satisfies Bel({ω'}) ≥ Pl({ω})
    non_dominated = []
    for k in range(n):
        dominated = any(
            bel_s[j] >= pl_s[k] - VALID_TOL
            for j in range(n) if j != k
        )
        if not dominated:
            non_dominated.append(m.frame[k])

    return frozenset(non_dominated)


def weak_dominance(m: DSVector) -> frozenset:
    """
    Weak dominance — partial decision via componentwise comparison.

    Returns the set of non-dominated atoms under the weak dominance
    relation (Walley 1991):

        ω ≻_wd ω' ⟺ Bel({ω}) ≥ Bel({ω'}) and Pl({ω}) ≥ Pl({ω'})

    An atom ω is non-dominated if no other atom ω' dominates it in
    both Bel and Pl simultaneously.

    Weak dominance is less selective than strong dominance: every set
    of non-dominated elements under strong dominance is a subset of
    those under weak dominance.

    With 0-1 utilities this corresponds to the weak dominance in
    the sense of Troffaes (2007): E_-(ω) = Bel({ω}), E^+(ω) = Pl({ω}).

    Parameters
    ----------
    m : DSVector
        The BBA (kind=Kind.M).

    Returns
    -------
    frozenset[str]
        The set of non-dominated atoms.

    References
    ----------
    Walley, P. (1991). Statistical reasoning with imprecise probabilities.
    Ma, L., Denœux, T. (2021). KBS, 214, 106742. Section 3.4.
    Mutmainah, S. (thesis). Eqs. (1.26) and (1.30).
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    _check_bba(m, "weak_dominance")

    n = len(m.frame)
    m_dense = m.dense
    bel_s = np.array([m_dense[1 << k] for k in range(n)])
    pl_s  = m.contour()

    # ω is non-dominated if no ω' satisfies Bel({ω'}) ≥ Bel({ω})
    # and Pl({ω'}) ≥ Pl({ω}) simultaneously
    non_dominated = []
    for k in range(n):
        dominated = any(
            bel_s[j] >= bel_s[k] - VALID_TOL and pl_s[j] >= pl_s[k] - VALID_TOL
            for j in range(n) if j != k
        )
        if not dominated:
            non_dominated.append(m.frame[k])

    return frozenset(non_dominated)
