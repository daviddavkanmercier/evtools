"""
Correction mechanisms for belief functions in the Dempster-Shafer theory.

A correction mechanism modifies a BBA mS provided by a source, to account
for knowledge about the quality (reliability, truthfulness) of that source.

Available functions
-------------------
theta_contextual_discount(m, betas)  — Θ-contextual discounting (general)
contextual_discount(m, betas)        — Ω-contextual discounting (singletons)
discount(m, alpha)                   — Classical discounting (single rate)
contextual_reinforce(m, betas)       — Contextual reinforcement (dual of CD)
contextual_dediscount(m, betas)      — Contextual de-discounting (inverse of CD)
contextual_dereinforce(m, betas)     — Contextual de-reinforcement (inverse of CR)
contextual_negate(m, betas)          — Contextual negating

Hierarchy of discounting
------------------------
discount(m, α)
    Special case of theta_contextual_discount with Θ = {Ω} and β = 1−α.

contextual_discount(m, betas)
    Special case of theta_contextual_discount with Θ = singletons of Ω.

theta_contextual_discount(m, betas)
    General form: Θ is any partition of Ω.

Summary table (Pichon et al. 2016, Figure 2)
--------------------------------------------
                          Dual         Inverse
  CD  (discounting)  ←→  CR           CdD
  CR  (reinforcement) ←→ CD           CdR
  CdD (de-discounting) ←→ CdR        CD
  CdR (de-reinforce)   ←→ CdD        CR
  CN  (negating)       —  generalization of negating

Notation for simple MFs
------------------------
  θ^β   : negative simple MF — focal sets ∅ (mass β) and θ (mass 1−β)
           used in CD, CdD
  A^β   : positive simple MF — focal sets Ω (mass β) and A (mass 1−β)
           used in CR, CdR

References
----------
- Mercier, D., Quost, B., Denoeux, T. (2008). Refined modeling of sensor
  reliability in the belief function framework using contextual discounting.
  Information Fusion, 9(2), 246-258.
- Pichon, F., Mercier, D., Lefevre, E., Delmotte, F. (2016). Proposition and
  learning of some belief function contextual correction mechanisms.
  International Journal of Approximate Reasoning, 72, 4-42.
"""

from __future__ import annotations

import numpy as np

from .dsvector import DSVector, Kind
from .combinations import drc, decombine_crc, decombine_drc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_bba(m: DSVector, fn: str) -> None:
    """Raise ValueError if m is not a BBA (kind != Kind.M)."""
    if m.kind != Kind.M:
        raise ValueError(
            f"{fn}: argument has kind '{m.kind.value}', expected 'm'."
        )


def _check_betas(betas: dict[frozenset, float], fn: str,
                 open_left: bool = False) -> None:
    """
    Validate beta values.

    Parameters
    ----------
    open_left : bool
        If True, require β ∈ (0, 1] (strict lower bound, for de-corrections).
        If False, require β ∈ [0, 1] (closed interval, default).
    """
    for subset, beta in betas.items():
        lo_ok = beta > 1e-15 if open_left else beta >= 0.0
        hi_ok = beta <= 1.0
        if not (lo_ok and hi_ok):
            interval = "(0, 1]" if open_left else "[0, 1]"
            raise ValueError(
                f"{fn}: beta={beta} for {set(subset)} "
                f"must be in {interval}."
            )


def _check_partition(frame: list[str], betas: dict[frozenset, float],
                     fn: str) -> None:
    """
    Raise ValueError if the keys of betas do not form a partition of frame.

    Requires:
    - No empty set as context.
    - All atoms belong to frame.
    - Contexts are pairwise disjoint.
    - Union of contexts = Ω.
    """
    omega = set(frame)
    covered: set[str] = set()

    for subset, beta in betas.items():
        if len(subset) == 0:
            raise ValueError(
                f"{fn}: the empty set cannot be a context in the partition."
            )
        unknown = set(subset) - omega
        if unknown:
            raise ValueError(
                f"{fn}: atoms {unknown} in subset {set(subset)} "
                f"are not in the frame {frame}."
            )
        overlap = covered & set(subset)
        if overlap:
            raise ValueError(
                f"{fn}: atom(s) {overlap} appear in more than one subset "
                f"— the contexts must be disjoint (partition of Ω)."
            )
        covered |= set(subset)

    missing = omega - covered
    if missing:
        raise ValueError(
            f"{fn}: atom(s) {missing} are not covered by any context "
            f"— the contexts must cover all of Ω (partition)."
        )


# ---------------------------------------------------------------------------
# Θ-contextual discounting (CD based on a coarsening)
# ---------------------------------------------------------------------------

def theta_contextual_discount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Θ-contextual discounting of a BBA (most general discounting form).

    Given a coarsening Θ = {θ₁, ..., θL} (a partition of Ω) and reliability
    degrees β = (β₁, ..., βL) with βℓ ∈ [0,1], the corrected BBA is:

        αm = mS ∪ (∪_{θℓ ∈ Θ} θℓ^{βℓ})

    where θℓ^{βℓ} is the negative simple MF with focal sets ∅ (mass βℓ)
    and θℓ (mass 1−βℓ), and ∪ is the TBM disjunctive rule.

    Interpretation: βℓ is the agent's degree of belief that the source is
    reliable when the true value of the variable of interest lies in context
    θℓ. The source is a negative liar in the complement of θℓ with mass
    1−βℓ (Pichon et al. 2016, Remark 5 and Proposition 7).

    Special cases
    -------------
    - Classical discounting (Shafer 1976):
        Θ = {Ω},  betas = {frozenset(frame): 1−α}
    - Ω-contextual discounting (Mercier et al. 2008, Section 3):
        Θ = singletons {{x1}, ..., {xK}}

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context θℓ (non-empty subset of Ω) to reliability
        degree βℓ ∈ [0,1]. The keys must form a partition of Ω.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation. Always valid.

    Raises
    ------
    ValueError
        If m is not a BBA, if any beta is outside [0,1], or if the
        contexts do not form a partition of Ω.

    Examples
    --------
    Classical discounting with α = 0.4:

    >>> frame = ["a", "h", "r"]
    >>> m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
    >>> m_disc = theta_contextual_discount(m, {frozenset(frame): 0.6})

    Contextual discounting, sensor unreliable only when target is an airplane
    (Example 1 of Mercier et al. 2008, Case 1 with α_a=0.4):

    >>> betas = {
    ...     frozenset({"a"}): 0.6,
    ...     frozenset({"h"}): 1.0,
    ...     frozenset({"r"}): 1.0,
    ... }
    >>> m_cd = theta_contextual_discount(m, betas)
    >>> # m_cd({a})=0.5, m_cd({r})=0.3, m_cd({a,r})=0.2

    References
    ----------
    Mercier, D., Quost, B., Denoeux, T. (2008). Information Fusion, 9(2),
    246-258. Sections 3 and 4 (Propositions 6–11).
    """
    _check_bba(m, "theta_contextual_discount")
    _check_betas(betas, "theta_contextual_discount")
    _check_partition(m.frame, betas, "theta_contextual_discount")

    # Build the corrective MF: ∪_{θℓ} θℓ^{βℓ}
    # Neutral element of DRC: the inconsistent BBA m(∅)=1
    corrective = DSVector.from_sparse(m.frame, {frozenset(): 1.0}, kind=Kind.M)
    for subset, beta in betas.items():
        corrective = drc(corrective, DSVector.negative_simple(m.frame, subset, beta))

    return drc(m, corrective)


# ---------------------------------------------------------------------------
# Ω-contextual discounting (special case: Θ = singletons)
# ---------------------------------------------------------------------------

def contextual_discount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Ω-contextual discounting of a BBA.

    Special case of Θ-contextual discounting where the partition Θ is the
    set of singletons: Θ = {{x1}, {x2}, ..., {xK}}.

    Reliability degrees βk ∈ [0,1] are given per atom xk ∈ Ω. The source
    is assumed to be reliable in context {xk} with degree βk, independently
    for each k.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from singleton {xk} to reliability degree βk ∈ [0,1].
        All atoms of Ω must appear exactly once as singleton keys.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation. Always valid.

    Raises
    ------
    ValueError
        If any key is not a singleton, or if other partition conditions fail.

    References
    ----------
    Mercier, D., Quost, B., Denoeux, T. (2008). Information Fusion, 9(2),
    246-258. Section 3 (Propositions 1–5).
    """
    for subset in betas:
        if len(subset) != 1:
            raise ValueError(
                f"contextual_discount: all contexts must be singletons, "
                f"got {set(subset)} (size {len(subset)}). "
                f"Use theta_contextual_discount for non-singleton contexts."
            )
    return theta_contextual_discount(m, betas)


# ---------------------------------------------------------------------------
# Classical discounting (special case: Θ = {Ω})
# ---------------------------------------------------------------------------

def discount(m: DSVector, alpha: float) -> DSVector:
    """
    Classical discounting of a BBA with a single discount rate α.

    Special case of Θ-contextual discounting with Θ = {Ω} and β = 1−α.

    The corrected BBA satisfies (Shafer 1976, p. 252):
        αm(A) = (1−α) · m(A)   for all A ⊊ Ω
        αm(Ω) = (1−α) · m(Ω) + α

    Interpretation: the source is reliable with degree of belief 1−α. With
    mass α, the source is irrelevant and is replaced by the vacuous BBA.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    alpha : float
        Discount rate α ∈ [0, 1].
        α = 0: m is unchanged (fully reliable source).
        α = 1: m is replaced by the vacuous BBA (fully unreliable source).

    Returns
    -------
    DSVector
        The discounted BBA, in sparse representation. Always valid.

    Raises
    ------
    ValueError
        If m is not a BBA, or if alpha is not in [0, 1].

    References
    ----------
    Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton, p. 252.
    Mercier, D., Quost, B., Denoeux, T. (2008). Section 2.5.
    """
    _check_bba(m, "discount")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"discount: alpha must be in [0, 1], got {alpha}.")
    return theta_contextual_discount(m, {frozenset(m.frame): 1.0 - alpha})


# ---------------------------------------------------------------------------
# Contextual Reinforcement (CR)
# ---------------------------------------------------------------------------

def contextual_reinforce(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual Reinforcement (CR) of a BBA.

    Dual of contextual discounting: uses the CRC (∩) instead of the DRC (∪).
    For each context A ∈ A with degree βA ∈ [0,1]:

        CR(mS) = mS ∩ (∩_{A ∈ A} A^{βA})

    where A^{βA} is the simple MF with focal sets Ω (mass βA) and A
    (mass 1−βA).

    Unlike CD, the set A of contexts need not form a partition of Ω.

    Interpretation (Pichon et al. 2016, Proposition 5): for each context A,
    the source is truthful with mass βA and is a positive liar in A with
    mass 1−βA (it lies only when it declares a value in A as possible).

    When A = {∅} and β = β₀, CR reduces to reinforcement (the dual of
    classical discounting), which transfers a fraction 1−β₀ of the mass
    of non-empty focal sets to the empty set.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context A (subset of Ω) to degree βA ∈ [0,1].
        Contexts need not form a partition of Ω.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation. Always valid if m
        is valid and βA ∈ [0,1].

    Raises
    ------
    ValueError
        If m is not a BBA, or if any beta is outside [0,1].

    References
    ----------
    Pichon, F., Mercier, D., Lefevre, E., Delmotte, F. (2016). IJAR, 72,
    4-42. Section 5.2, Eq. (18) and Proposition 5.
    """
    _check_bba(m, "contextual_reinforce")
    _check_betas(betas, "contextual_reinforce")

    from .combinations import crc
    result = m
    for subset, beta in betas.items():
        result = crc(result, DSVector.simple(m.frame, subset, beta))
    return result


# ---------------------------------------------------------------------------
# Contextual De-Discounting (CdD) — inverse of CD
# ---------------------------------------------------------------------------

def contextual_dediscount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual De-Discounting (CdD) of a BBA.

    Inverse of contextual discounting: removes a previously applied CD
    correction. Defined via the disjunctive decombination rule (6∪):

        CdD(mS) = mS 6∪ (∪_{A ∈ A} A^{βA})

    where A^{βA} is the negative simple MF with focal sets ∅ (mass βA)
    and A (mass 1−βA).

    Use case: an agent Ag received a source output mS, applied CD with
    degrees β, and transmitted the result mCD. A second agent Ag2 who
    thinks this CD was wrong can use CdD to recover mS, then apply
    their own CD (Pichon et al. 2016, Example 15).

    Warning
    -------
    The result may not be a valid BBA (negative masses or sum ≠ 1).
    Always check `.is_valid` after calling this function.
    Validity requires that the original CD was actually applied
    (1 − mS(X) ≤ βA ≤ 1 for classical discounting).

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M), typically the output of a
        prior contextual_discount call.
    betas : dict[frozenset, float]
        Mapping from context A to degree βA ∈ (0, 1].
        Must match the betas used in the original CD.
        Note: β = 0 is excluded (would require dividing by zero).

    Returns
    -------
    DSVector
        The corrected BBA. Check .is_valid before use.

    Raises
    ------
    ValueError
        If m is not a BBA, or if any beta is not in (0, 1].

    References
    ----------
    Pichon, F., Mercier, D., Lefevre, E., Delmotte, F. (2016). IJAR, 72,
    4-42. Section 6.1, Definition 7, Eq. (30).
    """
    _check_bba(m, "contextual_dediscount")
    _check_betas(betas, "contextual_dediscount", open_left=True)

    corrective = DSVector.from_sparse(m.frame, {frozenset(): 1.0}, kind=Kind.M)
    for subset, beta in betas.items():
        corrective = drc(corrective, DSVector.negative_simple(m.frame, subset, beta))

    return decombine_drc(m, corrective)


# ---------------------------------------------------------------------------
# Contextual De-Reinforcement (CdR) — inverse of CR
# ---------------------------------------------------------------------------

def contextual_dereinforce(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual De-Reinforcement (CdR) of a BBA.

    Inverse of contextual reinforcement: removes a previously applied CR
    correction. Defined via the conjunctive decombination rule (6∩):

        CdR(mS) = mS 6∩ (∩_{A ∈ A} A^{βA})

    where A^{βA} is the simple MF with focal sets Ω (mass βA) and A
    (mass 1−βA).

    Dual of CdD: CdD is the inverse of CD, CdR is the inverse of CR.

    Warning
    -------
    The result may not be a valid BBA (negative masses or sum ≠ 1).
    Always check `.is_valid` after calling this function.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M), typically the output of a
        prior contextual_reinforce call.
    betas : dict[frozenset, float]
        Mapping from context A to degree βA ∈ (0, 1].
        Must match the betas used in the original CR.
        Note: β = 0 is excluded (would require dividing by zero).

    Returns
    -------
    DSVector
        The corrected BBA. Check .is_valid before use.

    Raises
    ------
    ValueError
        If m is not a BBA, or if any beta is not in (0, 1].

    References
    ----------
    Pichon, F., Mercier, D., Lefevre, E., Delmotte, F. (2016). IJAR, 72,
    4-42. Section 6.1, Definition 8, Eq. (34).
    """
    _check_bba(m, "contextual_dereinforce")
    _check_betas(betas, "contextual_dereinforce", open_left=True)

    from .combinations import crc
    corrective = DSVector.from_sparse(
        m.frame, {frozenset(m.frame): 1.0}, kind=Kind.M
    )
    for subset, beta in betas.items():
        corrective = crc(corrective, DSVector.simple(m.frame, subset, beta))

    return decombine_crc(m, corrective)


# ---------------------------------------------------------------------------
# Contextual Negating (CN)
# ---------------------------------------------------------------------------

def contextual_negate(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual Negating (CN) of a BBA.

    Extension of the negating operation (Pichon et al. 2016, Section 7).
    For each context A ∈ A with degree βA ∈ [0,1]:

        CN(mS) = mS ∩̂ (∩̂_{A ∈ A} A^{βA})

    where ∩̂ is the equivalence combination rule and A^{βA} is the simple
    MF with focal sets Ω (mass βA) and A (mass 1−βA).

    Implemented directly from the BBC procedure (Proposition 9 of Pichon
    et al. 2016): for each context A and focal set B of the current BBA,
        - mass βA stays on B (truthful part)
        - mass 1−βA goes to B∩̂A = (B∩A) ∪ (B̄∩Ā) (non-truthful part)
    applied iteratively for each context.

    Interpretation: for each context A, the source is truthful with mass
    βA and non-truthful in A with mass 1−βA — it tells the contrary of
    what it knows, but only for the values in A.

    Special cases
    -------------
    - βA = 1 for all A: no correction (fully truthful source).
    - A = {∅}, βA = 0: pure negation — m(B) → m(B̄) for all B ⊆ Ω.
    - A = {∅}, β ∈ (0,1): classical negating — m = β·mS + (1−β)·m̄S.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context A (subset of Ω) to degree βA ∈ [0,1].
        Contexts need not form a partition of Ω.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation. Always valid if
        m is valid and βA ∈ [0,1].

    Raises
    ------
    ValueError
        If m is not a BBA, or if any beta is outside [0,1].

    References
    ----------
    Pichon, F., Mercier, D., Lefevre, E., Delmotte, F. (2016). IJAR, 72,
    4-42. Section 7, Definition 11, Eq. (39) and Proposition 9.
    """
    _check_bba(m, "contextual_negate")
    _check_betas(betas, "contextual_negate")

    result = m
    for subset, beta in betas.items():
        a_bar = frozenset(m.frame) - subset
        new_sparse: dict[frozenset, float] = {}
        for focal, value in result.sparse.items():
            # Truthful part: mass β stays on focal set B
            if beta > 1e-15:
                new_sparse[focal] = new_sparse.get(focal, 0.0) + beta * value
            # Non-truthful part: mass (1−β) goes to B∩̂A = (B∩A) ∪ (B̄∩Ā)
            if 1.0 - beta > 1e-15:
                b_inter_a = focal & subset
                bbar_inter_abar = (frozenset(m.frame) - focal) & a_bar
                equiv_set = b_inter_a | bbar_inter_abar
                new_sparse[equiv_set] = (
                    new_sparse.get(equiv_set, 0.0) + (1.0 - beta) * value
                )
        new_sparse = {k: v for k, v in new_sparse.items() if abs(v) > 1e-15}
        result = DSVector.from_sparse(m.frame, new_sparse, kind=Kind.M)

    return result
