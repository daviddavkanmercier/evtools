"""
Correction mechanisms for belief functions in the Dempster-Shafer theory.

A correction mechanism modifies a BBA mS provided by a source, to account
for knowledge about the quality (reliability, truthfulness) of that source.

Available corrections
---------------------
theta_contextual_discount(m, betas) — Θ-contextual discounting (general form)
contextual_discount(m, betas)       — Contextual discounting (partition of singletons)
discount(m, alpha)                  — Classical discounting (single rate)

Hierarchy
---------
discount          is a special case of contextual_discount  (Θ = {Ω})
contextual_discount is a special case of theta_contextual_discount (Θ = singletons)

Θ-contextual discounting (Mercier, Quost, Denoeux 2008)
--------------------------------------------------------
Given a coarsening Θ = {θ₁, ..., θL} (a partition of Ω) and a vector of
reliability degrees β = (β₁, ..., βL) with βℓ ∈ [0,1], the Θ-contextually
discounted BBA is:

    αm(A) = mS ∪ (∪_{θℓ ∈ Θ} θℓ^{βℓ})

where θℓ^{βℓ} denotes the negative simple MF with focal sets ∅ and θℓ,
with respective masses βℓ and 1 − βℓ. The operation ∪ is the TBM
disjunctive rule of combination.

Special cases
-------------
- Classical discounting (Shafer):  Θ = {Ω}, single rate α → β = {Ω: 1-α}
- Contextual discounting (Ω-contextual): Θ = {{x1}, {x2}, ..., {xK}}

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
from .combinations import drc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_bba(m: DSVector, fn: str) -> None:
    """Raise ValueError if m is not a BBA."""
    if m.kind != Kind.M:
        raise ValueError(
            f"{fn}: argument has kind '{m.kind.value}', expected 'm'."
        )


def _check_partition(frame: list[str], betas: dict[frozenset, float], fn: str) -> None:
    """
    Raise ValueError if the keys of betas do not form a partition of frame.

    A partition requires:
    - Every atom appears in exactly one subset (key).
    - Every value is in [0, 1].
    - No empty sets.
    """
    omega = set(frame)

    # Check beta values
    for subset, beta in betas.items():
        if not (0.0 <= beta <= 1.0):
            raise ValueError(
                f"{fn}: beta value {beta} for subset {set(subset)} "
                f"is not in [0, 1]."
            )
        if len(subset) == 0:
            raise ValueError(
                f"{fn}: the empty set cannot be a context in the partition."
            )

    # Check partition: union must equal Ω, subsets must be disjoint
    covered: set[str] = set()
    for subset in betas:
        atoms = set(subset)
        # Check all atoms belong to frame
        unknown = atoms - omega
        if unknown:
            raise ValueError(
                f"{fn}: atoms {unknown} in subset {set(subset)} "
                f"are not in the frame {frame}."
            )
        # Check disjointness
        overlap = covered & atoms
        if overlap:
            raise ValueError(
                f"{fn}: atom(s) {overlap} appear in more than one subset "
                f"— the contexts must be disjoint (partition of Ω)."
            )
        covered |= atoms

    # Check completeness
    missing = omega - covered
    if missing:
        raise ValueError(
            f"{fn}: atom(s) {missing} are not covered by any context "
            f"— the contexts must cover all of Ω (partition)."
        )


# ---------------------------------------------------------------------------
# Negative simple MF helper
# ---------------------------------------------------------------------------

def _negative_simple_mf(frame: list[str], subset: frozenset, beta: float) -> DSVector:
    """
    Build the negative simple MF θ^β with focal sets ∅ and θ=subset.

    This MF assigns mass β to ∅ and mass (1-β) to subset.
    It is a subnormal BBA used as the corrective term in Θ-discounting.

    Reference: Mercier et al. (2008), Eq. (13) and surrounding.
    """
    sparse: dict[frozenset, float] = {}
    if beta > 1e-15:
        sparse[frozenset()] = beta
    if 1.0 - beta > 1e-15:
        sparse[subset] = 1.0 - beta
    return DSVector.from_sparse(frame, sparse, kind=Kind.M)


# ---------------------------------------------------------------------------
# Θ-contextual discounting
# ---------------------------------------------------------------------------

def theta_contextual_discount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Θ-contextual discounting of a BBA.

    Given a coarsening Θ = {θ₁, ..., θL} of Ω (a partition) and reliability
    degrees βℓ ∈ [0,1] for each θℓ ∈ Θ, the corrected BBA is:

        αm = mS ∪ (∪_{θℓ ∈ Θ} θℓ^{βℓ})

    where θℓ^{βℓ} is the negative simple MF assigning mass βℓ to ∅ and
    mass 1−βℓ to θℓ.

    Interpretation: βℓ is the degree of belief that the source is reliable
    in context θℓ. When βℓ = 1 the source is fully reliable in that context
    (no correction). When βℓ = 0 the source is fully unreliable (vacuous
    belief in that context).

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context (subset of Ω) to reliability degree β ∈ [0,1].
        The keys must form a partition of Ω.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation.

    Raises
    ------
    ValueError
        If m is not a BBA, if beta values are outside [0,1], or if the
        contexts do not form a partition of Ω.

    Examples
    --------
    Classical discounting with α = 0.4 (β = 0.6):

    >>> frame = ["a", "h", "r"]
    >>> m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
    >>> alpha = 0.4
    >>> m_disc = theta_contextual_discount(
    ...     m, {frozenset(frame): 1 - alpha}
    ... )

    Contextual discounting — sensor unreliable when target is an airplane:

    >>> betas = {
    ...     frozenset({"a"}):      0.6,   # 60% reliable when airplane
    ...     frozenset({"h"}):      1.0,   # fully reliable when helicopter
    ...     frozenset({"r"}):      1.0,   # fully reliable when rocket
    ... }
    >>> m_cd = theta_contextual_discount(m, betas)

    References
    ----------
    Mercier, D., Quost, B., Denoeux, T. (2008). Refined modeling of sensor
    reliability in the belief function framework using contextual discounting.
    Information Fusion, 9(2), 246-258. Section 4.
    """
    _check_bba(m, "theta_contextual_discount")
    _check_partition(m.frame, betas, "theta_contextual_discount")

    # Build the corrective MF: ∪_{θℓ} θℓ^{βℓ}
    # Start from the neutral element of DRC: the inconsistent MF m∅(∅)=1
    corrective = DSVector.from_sparse(
        m.frame, {frozenset(): 1.0}, kind=Kind.M
    )

    for subset, beta in betas.items():
        # θℓ^{βℓ}: negative simple MF with mass β on ∅, mass (1-β) on θℓ
        neg_simple = _negative_simple_mf(m.frame, subset, beta)
        corrective = drc(corrective, neg_simple)

    # Apply correction: mS ∪ corrective
    return drc(m, corrective)


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------

def contextual_discount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual discounting (Ω-contextual discounting).

    Special case of Θ-contextual discounting where the partition is the
    set of singletons Θ = {{x1}, {x2}, ..., {xK}}.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from singleton {xk} to reliability degree βk ∈ [0,1].
        Each key must be a singleton subset of Ω, and all atoms of Ω
        must appear exactly once.

    Returns
    -------
    DSVector
        The corrected BBA, in sparse representation.

    References
    ----------
    Mercier, D., Quost, B., Denoeux, T. (2008). Section 3.
    """
    # Validate that all keys are singletons
    for subset in betas:
        if len(subset) != 1:
            raise ValueError(
                f"contextual_discount: all contexts must be singletons, "
                f"got {set(subset)} (size {len(subset)}). "
                f"Use theta_contextual_discount for non-singleton contexts."
            )
    return theta_contextual_discount(m, betas)


def discount(m: DSVector, alpha: float) -> DSVector:
    """
    Classical discounting with a single discount rate α.

    Special case of Θ-contextual discounting with Θ = {Ω} and β = 1 − α.

    The corrected BBA satisfies:
        αm(A) = (1−α) · m(A)   for all A ≠ Ω
        αm(Ω) = (1−α) · m(Ω) + α

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    alpha : float
        Discount rate α ∈ [0, 1]. When α=0, m is unchanged. When α=1,
        m is replaced by the vacuous BBA.

    Returns
    -------
    DSVector
        The discounted BBA, in sparse representation.

    References
    ----------
    Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton.
    Mercier, D., Quost, B., Denoeux, T. (2008). Section 2.5.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"discount: alpha must be in [0, 1], got {alpha}."
        )
    omega = frozenset(m.frame)
    return theta_contextual_discount(m, {omega: 1.0 - alpha})


# ---------------------------------------------------------------------------
# Inverse combination helpers (decombination)
# ---------------------------------------------------------------------------

def _conjunctive_decombine(m1: DSVector, m2: DSVector) -> DSVector:
    """
    Conjunctive decombination: m1 6∩ m2.

    Defined via commonality functions: q_result = q1 / q2.
    Requires m2 to be non-dogmatic (q2(A) > 0 for all A).

    The result may not be a valid BBA — check .is_valid.
    """
    from .conversions import mtoq, qtom
    q1 = mtoq(m1.dense)
    q2 = mtoq(m2.dense)
    if np.any(np.isclose(q2, 0.0)):
        raise ValueError(
            "conjunctive_decombine: m2 is dogmatic (q2(A)=0 for some A). "
            "Decombination is undefined."
        )
    q_result = q1 / q2
    return DSVector.from_dense(m1.frame, qtom(q_result), kind=Kind.M)


def _disjunctive_decombine(m1: DSVector, m2: DSVector) -> DSVector:
    """
    Disjunctive decombination: m1 6∪ m2.

    Defined via implicability functions: b_result = b1 / b2.
    Requires m2 to be non-normal (b2(A) > 0 for all A).

    The result may not be a valid BBA — check .is_valid.
    """
    from .conversions import mtob, btom
    b1 = mtob(m1.dense)
    b2 = mtob(m2.dense)
    if np.any(np.isclose(b2, 0.0)):
        raise ValueError(
            "disjunctive_decombine: m2 is normal (b2(A)=0 for some A). "
            "Decombination is undefined."
        )
    b_result = b1 / b2
    return DSVector.from_dense(m1.frame, btom(b_result), kind=Kind.M)


# ---------------------------------------------------------------------------
# Simple MF helpers
# ---------------------------------------------------------------------------

def _simple_mf(frame: list[str], subset: frozenset, beta: float) -> DSVector:
    """
    Build the simple MF A^β with focal sets Ω and A=subset.

    Assigns mass β to Ω and mass (1-β) to subset.
    Used as corrective term in CR and CdR.
    """
    omega = frozenset(frame)
    sparse: dict[frozenset, float] = {}
    if beta > 1e-15:
        sparse[omega] = beta
    if 1.0 - beta > 1e-15:
        sparse[subset] = 1.0 - beta
    return DSVector.from_sparse(frame, sparse, kind=Kind.M)


# ---------------------------------------------------------------------------
# Contextual Reinforcement (CR)
# ---------------------------------------------------------------------------

def contextual_reinforce(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual Reinforcement (CR) of a BBA.

    Dual of contextual discounting: uses the CRC instead of the DRC.
    Defined as:

        CR(mS) = mS ∩ (∩_{A ∈ A} A^{βA})

    where A^{βA} is the simple MF assigning mass βA to Ω and mass (1-βA)
    to A. The set A of contexts can be arbitrary (need not form a partition).

    Interpretation: the source is assumed to be truthful with mass βA and
    to be a positive liar in A with mass (1-βA), for each context A.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context (subset of Ω) to degree βA ∈ [0,1].
        Contexts need not form a partition.

    Returns
    -------
    DSVector
        The corrected BBA. Always valid if inputs are valid.

    References
    ----------
    Pichon et al. (2016). IJAR, 72, 4-42. Section 5.2, Eq. (18).
    """
    _check_bba(m, "contextual_reinforce")
    for subset, beta in betas.items():
        if not (0.0 <= beta <= 1.0):
            raise ValueError(
                f"contextual_reinforce: beta={beta} for {set(subset)} "
                f"is not in [0, 1]."
            )

    from .combinations import crc
    result = m
    for subset, beta in betas.items():
        simple = _simple_mf(m.frame, subset, beta)
        result = crc(result, simple)
    return result


# ---------------------------------------------------------------------------
# Contextual De-Discounting (CdD)
# ---------------------------------------------------------------------------

def contextual_dediscount(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual De-Discounting (CdD) of a BBA.

    Inverse of contextual discounting: removes a CD correction.
    Defined as:

        CdD(mS) = mS 6∪ (∪_{A ∈ A} A^{βA})

    where A^{βA} is the negative simple MF assigning mass βA to ∅ and
    mass (1-βA) to A.

    Warning: the result may not be a valid BBA. Check .is_valid.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context to degree βA ∈ (0, 1].

    Returns
    -------
    DSVector
        The corrected BBA (may not be valid — check .is_valid).

    References
    ----------
    Pichon et al. (2016). IJAR, 72, 4-42. Section 6.1, Eq. (30).
    """
    _check_bba(m, "contextual_dediscount")
    for subset, beta in betas.items():
        if not (0.0 < beta <= 1.0):
            raise ValueError(
                f"contextual_dediscount: beta={beta} for {set(subset)} "
                f"must be in (0, 1]."
            )

    # Build ∪_{A} A^{βA} — same as in CD
    corrective = DSVector.from_sparse(
        m.frame, {frozenset(): 1.0}, kind=Kind.M
    )
    for subset, beta in betas.items():
        neg_simple = _negative_simple_mf(m.frame, subset, beta)
        corrective = drc(corrective, neg_simple)

    return _disjunctive_decombine(m, corrective)


# ---------------------------------------------------------------------------
# Contextual De-Reinforcement (CdR)
# ---------------------------------------------------------------------------

def contextual_dereinforce(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual De-Reinforcement (CdR) of a BBA.

    Inverse of contextual reinforcement: removes a CR correction.
    Defined as:

        CdR(mS) = mS 6∩ (∩_{A ∈ A} A^{βA})

    where A^{βA} is the simple MF assigning mass βA to Ω and mass (1-βA)
    to A.

    Warning: the result may not be a valid BBA. Check .is_valid.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context to degree βA ∈ (0, 1].

    Returns
    -------
    DSVector
        The corrected BBA (may not be valid — check .is_valid).

    References
    ----------
    Pichon et al. (2016). IJAR, 72, 4-42. Section 6.1, Eq. (34).
    """
    _check_bba(m, "contextual_dereinforce")
    for subset, beta in betas.items():
        if not (0.0 < beta <= 1.0):
            raise ValueError(
                f"contextual_dereinforce: beta={beta} for {set(subset)} "
                f"must be in (0, 1]."
            )

    from .combinations import crc
    corrective = DSVector.from_sparse(
        m.frame, {frozenset(m.frame): 1.0}, kind=Kind.M
    )
    for subset, beta in betas.items():
        simple = _simple_mf(m.frame, subset, beta)
        corrective = crc(corrective, simple)

    return _conjunctive_decombine(m, corrective)


# ---------------------------------------------------------------------------
# Contextual Negating (CN)
# ---------------------------------------------------------------------------

def contextual_negate(
    m: DSVector,
    betas: dict[frozenset, float],
) -> DSVector:
    """
    Contextual Negating (CN) of a BBA.

    Extension of the negating operation to contextual knowledge.
    Defined as:

        CN(mS) = mS ∩̂ (∩̂_{A ∈ A} A^{βA})

    where ∩̂ is the equivalence rule and A^{βA} is the simple MF assigning
    mass βA to Ω and mass (1-βA) to A.

    For each context A, the source is assumed to be truthful with mass βA
    and non-truthful in A (tells the contrary of what it knows for values
    in A) with mass (1-βA).

    Implemented via: m(A) = β·mS(A) + (1-β)·mS(Ā) applied iteratively
    for each context, using the equivalence combination rule.

    Parameters
    ----------
    m : DSVector
        The BBA to correct (kind=Kind.M).
    betas : dict[frozenset, float]
        Mapping from context (subset of Ω) to degree βA ∈ [0,1].

    Returns
    -------
    DSVector
        The corrected BBA.

    References
    ----------
    Pichon et al. (2016). IJAR, 72, 4-42. Section 7, Eq. (39).
    """
    _check_bba(m, "contextual_negate")
    for subset, beta in betas.items():
        if not (0.0 <= beta <= 1.0):
            raise ValueError(
                f"contextual_negate: beta={beta} for {set(subset)} "
                f"is not in [0, 1]."
            )

    result = m
    n = len(m.frame)

    for subset, beta in betas.items():
        # For each context A: m_new(B) = β·m(B) + (1-β)·m(B∩A)
        # where B∩A is the logical equality (symmetric difference complement)
        # B∩̂A = (B∩A) ∪ (B̄∩Ā)
        new_sparse: dict[frozenset, float] = {}
        a_bar = frozenset(m.frame) - subset  # complement of A

        for focal, value in result.sparse.items():
            # Truthful part: mass β stays on focal
            if beta > 1e-15:
                new_sparse[focal] = new_sparse.get(focal, 0.0) + beta * value
            # Non-truthful part: mass (1-β) goes to logical equality B∩̂A
            if 1.0 - beta > 1e-15:
                # B∩̂A = (B∩A)∪(B̄∩Ā) = symmetric complement
                b_inter_a     = focal & subset
                b_bar_inter_abar = (frozenset(m.frame) - focal) & a_bar
                equiv_set = b_inter_a | b_bar_inter_abar
                new_sparse[equiv_set] = (
                    new_sparse.get(equiv_set, 0.0) + (1.0 - beta) * value
                )

        new_sparse = {k: v for k, v in new_sparse.items() if abs(v) > 1e-15}
        result = DSVector.from_sparse(m.frame, new_sparse, kind=Kind.M)

    return result
