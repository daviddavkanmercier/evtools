"""
Learning of contextual correction parameters from labeled data.

For each contextual correction mechanism (CD, CR, CN), the parameter
vector β ∈ [0, 1]^K minimizing the pl-based discrepancy E_pl is the
solution of a constrained linear least-squares problem (Pichon et al.
2016, Propositions 12, 14, 16). The K parameters decouple per atom
(Mutmainah 2021, Section 2.5), so each β_k has a closed-form
expression which is then clipped to [0, 1].

These functions accept hard labels (str) or soft labels (DSVector),
exactly like :func:`evtools.metrics.pl_loss` — hard labels recover the
classical E_pl problem, soft labels solve the Ẽ_pl extension
(Mutmainah et al. 2019; thesis Section 5.2).

API
---
    fit_cd(predictions, labels) -> {frozenset({ω_k}): β_k, ...}
    fit_cr(predictions, labels) -> {frozenset({ω_k}): β_k, ...}
    fit_cn(predictions, labels) -> {frozenset({ω_k}): β_k, ...}

The returned dict is in the format expected by the corresponding
correction in :mod:`evtools.corrections`:

    >>> betas = fit_cd(predictions, labels)
    >>> m_corrected = contextual_discount(m, betas)

References
----------
- Pichon, F., Mercier, D., Lefèvre, É., Delmotte, F. (2016). Proposition
  and learning of some belief function contextual correction mechanisms.
  IJAR, 72, 4-42. Propositions 12 (CD, Eq. 47), 14 (CR, Eq. 52),
  16 (CN, Eq. 54).
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2019). On learning
  evidential contextual corrections from soft labels using a measure of
  discrepancy between contour functions. SUM 2019.
- Mutmainah, S. (2021). Learning to adjust an evidential source of information
  using partially labeled data and partial decisions. PhD thesis, Université
  d'Artois. Section 2.5.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .dsvector import DSVector
from .constants import ZERO_MASS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_contours_and_labels(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build the (n, K) matrices PL and DELTA from a list of predictions
    and labels.

    PL[i, k] = plS_i(ω_k)   (contour function of prediction i on atom k)
    DELTA[i, k] = δ_{i,k}   (indicator if hard label, contour if soft)

    All predictions must share the same frame; the function returns the
    frame in addition to PL and DELTA.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"learning: predictions and labels have different lengths "
            f"({len(predictions)} vs {len(labels)})."
        )
    if len(predictions) == 0:
        raise ValueError("learning: predictions list is empty.")

    frame = list(predictions[0].frame)
    K = len(frame)
    n = len(predictions)

    PL    = np.zeros((n, K))
    DELTA = np.zeros((n, K))

    for i, (m_pred, label) in enumerate(zip(predictions, labels)):
        if list(m_pred.frame) != frame:
            raise ValueError(
                f"learning: prediction {i} has a different frame from prediction 0."
            )
        PL[i] = m_pred.contour()
        if isinstance(label, str):
            try:
                DELTA[i, frame.index(label)] = 1.0
            except ValueError:
                raise ValueError(
                    f"learning: label {label!r} not in frame {frame}."
                )
        elif isinstance(label, DSVector):
            if list(label.frame) != frame:
                raise ValueError(
                    f"learning: label {i} has a different frame from prediction {i}."
                )
            DELTA[i] = label.contour()
        else:
            raise TypeError(
                f"learning: label {i} must be str or DSVector, got "
                f"{type(label).__name__}."
            )

    return PL, DELTA, frame


def _ratio_clipped(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """
    Compute num/den element-wise with safety:
    - If |den| < ZERO_MASS, β_k defaults to 1.0 (no correction).
    - The result is clipped to [0, 1].
    """
    n = len(num)
    out = np.ones(n)  # default: no correction
    safe = np.abs(den) > ZERO_MASS
    out[safe] = num[safe] / den[safe]
    return np.clip(out, 0.0, 1.0)


def _betas_dict_singletons(frame: list[str], betas: np.ndarray) -> dict:
    """Convert β_k vector into a dict keyed by singletons (used for CD)."""
    return {frozenset({frame[k]}): float(betas[k]) for k in range(len(frame))}


def _betas_dict_complements(frame: list[str], betas: np.ndarray) -> dict:
    """Convert β_k vector into a dict keyed by complements of singletons
    (used for CR and CN — see Pichon 2016 Section 8.3, 8.4)."""
    omega = frozenset(frame)
    return {omega - {frame[k]}: float(betas[k]) for k in range(len(frame))}


# ---------------------------------------------------------------------------
# fit_cd, fit_cr, fit_cn
# ---------------------------------------------------------------------------

def fit_cd(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> dict:
    """
    Closed-form least-squares fit of contextual discounting (CD) parameters.

    Minimizes E_pl over β ∈ [0, 1]^K using the linearization of CD
    (Pichon et al. 2016, Proposition 12, Eq. 47):

        plᵢ(ωₖ) = 1 − βₖ · (1 − plSᵢ(ωₖ))

    Closed form (per atom, before clipping to [0, 1]):

        βₖ = Σᵢ (1 − plSᵢ(ωₖ))(1 − δᵢ,ₖ) / Σᵢ (1 − plSᵢ(ωₖ))²

    Parameters
    ----------
    predictions : sequence of DSVector
        Source BBAs, one per training instance. All share the same frame.
    labels : sequence of str or DSVector
        Hard labels (atom names) → minimizes E_pl.
        Soft labels (BBAs) → minimizes Ẽ_pl. Mixable in the same call.

    Returns
    -------
    dict[frozenset, float]
        Mapping {frozenset({ωₖ}): βₖ}, ready to pass to
        :func:`evtools.corrections.contextual_discount`.

    References
    ----------
    Pichon, F., Mercier, D., Lefèvre, É., Delmotte, F. (2016). IJAR, 72,
    4-42. Proposition 12, Eq. (47).
    """
    PL, DELTA, frame = _stack_contours_and_labels(predictions, labels)
    # Per-atom 1D least squares: β_k = Σ (1-pl)(1-δ) / Σ (1-pl)²
    one_minus_pl    = 1.0 - PL                      # (n, K)
    one_minus_delta = 1.0 - DELTA                   # (n, K)
    num = np.sum(one_minus_pl * one_minus_delta, axis=0)
    den = np.sum(one_minus_pl ** 2,              axis=0)
    # CD contexts are singletons {ω_k}.
    return _betas_dict_singletons(frame, _ratio_clipped(num, den))


def fit_cr(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> dict:
    """
    Closed-form least-squares fit of contextual reinforcement (CR) parameters.

    Minimizes E_pl over β ∈ [0, 1]^K using the linearization of CR
    (Pichon et al. 2016, Proposition 14, Eq. 52):

        plᵢ(ωₖ) = βₖ · plSᵢ(ωₖ)

    Closed form (per atom, before clipping):

        βₖ = Σᵢ plSᵢ(ωₖ) · δᵢ,ₖ / Σᵢ plSᵢ(ωₖ)²

    Returns
    -------
    dict[frozenset, float]
        Mapping ready to pass to
        :func:`evtools.corrections.contextual_reinforce`.

    References
    ----------
    Pichon, F., Mercier, D., Lefèvre, É., Delmotte, F. (2016). IJAR, 72,
    4-42. Proposition 14, Eq. (52).
    """
    PL, DELTA, frame = _stack_contours_and_labels(predictions, labels)
    num = np.sum(PL * DELTA, axis=0)
    den = np.sum(PL ** 2,    axis=0)
    # CR contexts are the complements of singletons (Pichon 2016, Section 8.3).
    return _betas_dict_complements(frame, _ratio_clipped(num, den))


def fit_cn(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> dict:
    """
    Closed-form least-squares fit of contextual negating (CN) parameters.

    Minimizes E_pl over β ∈ [0, 1]^K using the linearization of CN
    (Pichon et al. 2016, Proposition 16, Eq. 54):

        plᵢ(ωₖ) = 0.5 + (plSᵢ(ωₖ) − 0.5) · (2βₖ − 1)

    Closed form (per atom, before clipping):

        βₖ = Σᵢ (2·plSᵢ − 1)(plSᵢ + δᵢ − 1) / Σᵢ (2·plSᵢ − 1)²

    Returns
    -------
    dict[frozenset, float]
        Mapping ready to pass to
        :func:`evtools.corrections.contextual_negate`.

    References
    ----------
    Pichon, F., Mercier, D., Lefèvre, É., Delmotte, F. (2016). IJAR, 72,
    4-42. Proposition 16, Eq. (54).
    """
    PL, DELTA, frame = _stack_contours_and_labels(predictions, labels)
    two_pl_minus_1 = 2.0 * PL - 1.0
    rhs = PL + DELTA - 1.0
    num = np.sum(two_pl_minus_1 * rhs,        axis=0)
    den = np.sum(two_pl_minus_1 ** 2,         axis=0)
    # CN contexts are the complements of singletons (Pichon 2016, Section 8.4).
    return _betas_dict_complements(frame, _ratio_clipped(num, den))
