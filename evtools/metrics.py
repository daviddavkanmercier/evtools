"""
Performance metrics for evidential predictions.

Set-valued predictions (partial decisions)
------------------------------------------
Score a partial decision d ⊆ Ω against a true class ω. Typical inputs are
the output of :func:`evtools.decision.strong_dominance` or :func:`weak_dominance`.

    discounted_accuracy(d, ω)              — x = I(ω ∈ d) / |d|
    u65(d, ω)                              — 1.6·x − 0.6·x²
    u80(d, ω)                              — 2.2·x − 1.2·x²
    utility_score(d, ω, a=…, b=…)          — generic a·x − b·x²

BBA-valued predictions (evidential classifiers)
-----------------------------------------------
Discrepancy between the predicted contour function and a (hard or soft) label.

    pl_loss(predictions, labels)           — Σᵢ Σₖ (plᵢ(ωₖ) − δᵢ,ₖ)²
    mean_pl_loss(predictions, labels)      — pl_loss / n

Aggregators iterate over a paired list of predictions/labels and return
the mean:

    mean_discounted_accuracy(predictions, labels)
    mean_u65(predictions, labels)
    mean_u80(predictions, labels)
    mean_utility_score(predictions, labels, a=…, b=…)

References
----------
- Zaffalon, M., Corani, G., Mauá, D. (2012). Evaluating credal classifiers
  by utility-discounted predictive accuracy. IJAR, 53(8), 1282-1301.
- Mercier, D., Quost, B., Denœux, T. (2008). Refined modeling of sensor
  reliability in the belief function framework using contextual discounting.
  Information Fusion, 9(2), 246-258.
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2019). On learning
  evidential contextual corrections from soft labels using a measure of
  discrepancy between contour functions. SUM 2019.
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2021). Improving an
  evidential source of information using contextual corrections depending
  on partial decisions. BELIEF 2021, pp. 247-256.
- Mutmainah, S. (2021). Learning to adjust an evidential source of information
  using partially labeled data and partial decisions. PhD thesis, Université
  d'Artois. Sections 2.5, 3.4, 5.2.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

from .dsvector import DSVector

# ---------------------------------------------------------------------------
# Set-valued predictions (partial decisions)
# ---------------------------------------------------------------------------

def discounted_accuracy(d: frozenset, omega: str) -> float:
    """
    Discounted accuracy of a partial decision *d* against true class *omega*.

        x = I(ω ∈ d) / |d|

    where I is the indicator function. Returns 0 when d is empty or ω ∉ d.
    Returns 1/|d| ∈ (0, 1] when ω ∈ d, with the precise correct decision
    (|d| = 1) scoring 1.0 and larger correct partial decisions scoring less.

    Parameters
    ----------
    d : frozenset[str]
        Partial decision: a (possibly empty) set of atom names. Typically
        the output of :func:`evtools.decision.strong_dominance` or
        :func:`evtools.decision.weak_dominance`.
    omega : str
        The true class.

    Returns
    -------
    float
        The discounted accuracy x ∈ [0, 1].

    References
    ----------
    Zaffalon, M., Corani, G., Mauá, D. (2012). IJAR, 53(8), 1282-1301.
    Mutmainah, S. (2021). PhD thesis. Section 3.4.
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    if not d or omega not in d:
        return 0.0
    return 1.0 / len(d)


def utility_score(
    d: frozenset,
    omega: str,
    *,
    a: float,
    b: float,
) -> float:
    """
    Generic utility-discounted accuracy: u(x) = a·x − b·x², with x the
    discounted accuracy of *d* against true class *omega*.

    Special cases (Zaffalon et al. 2012):
    - u65 corresponds to (a, b) = (1.6, 0.6) — yields 0.65 when |d|=2 correct
    - u80 corresponds to (a, b) = (2.2, 1.2) — yields 0.80 when |d|=2 correct
      (more risk-averse decision maker)

    References
    ----------
    Zaffalon, M., Corani, G., Mauá, D. (2012). IJAR, 53(8), 1282-1301.
    Mutmainah, S. (2021). PhD thesis. Section 3.4, Eqs. (3.4)-(3.5).
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    x = discounted_accuracy(d, omega)
    return a * x - b * x * x


def u65(d: frozenset, omega: str) -> float:
    """
    Utility-discounted accuracy u65 of Zaffalon et al. (2012):

        u65(x) = 1.6·x − 0.6·x²

    where x = I(ω ∈ d)/|d|. Yields 1.0 for a precise correct decision,
    0.65 for a 2-element correct partial decision, and 0 otherwise.
    """
    return utility_score(d, omega, a=1.6, b=0.6)


def u80(d: frozenset, omega: str) -> float:
    """
    Utility-discounted accuracy u80 of Zaffalon et al. (2012):

        u80(x) = 2.2·x − 1.2·x²

    where x = I(ω ∈ d)/|d|. Yields 1.0 for a precise correct decision,
    0.80 for a 2-element correct partial decision, and 0 otherwise.
    Corresponds to a more risk-averse decision maker than :func:`u65`.
    """
    return utility_score(d, omega, a=2.2, b=1.2)


# ---------------------------------------------------------------------------
# Batch aggregators (mean over a dataset)
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float]) -> float:
    """Empirical mean. Returns 0.0 for an empty sequence (degenerate case)."""
    total = 0.0
    n = 0
    for v in values:
        total += v
        n += 1
    if n == 0:
        return 0.0
    return total / n


def mean_discounted_accuracy(
    predictions: Iterable[frozenset],
    labels: Iterable[str],
) -> float:
    """Mean discounted accuracy over a paired iterable of (prediction, label)."""
    return _mean(discounted_accuracy(d, y) for d, y in zip(predictions, labels))


def mean_u65(
    predictions: Iterable[frozenset],
    labels: Iterable[str],
) -> float:
    """Mean u65 over a paired iterable of (prediction, label)."""
    return _mean(u65(d, y) for d, y in zip(predictions, labels))


def mean_u80(
    predictions: Iterable[frozenset],
    labels: Iterable[str],
) -> float:
    """Mean u80 over a paired iterable of (prediction, label)."""
    return _mean(u80(d, y) for d, y in zip(predictions, labels))


def mean_utility_score(
    predictions: Iterable[frozenset],
    labels: Iterable[str],
    *,
    a: float,
    b: float,
) -> float:
    """Mean generic utility a·x − b·x² over a paired iterable of (prediction, label)."""
    return _mean(utility_score(d, y, a=a, b=b) for d, y in zip(predictions, labels))


# ---------------------------------------------------------------------------
# BBA-valued predictions: pl-based discrepancy (E_pl / Ẽ_pl)
# ---------------------------------------------------------------------------

def _label_to_indicator(label: Union[str, DSVector], frame: Sequence[str]) -> np.ndarray:
    """
    Convert a single label to a length-K indicator/contour vector.

    - str            → indicator vector δ_k = 1 if frame[k] == label else 0
    - DSVector       → contour function (length-K vector of singleton plausibilities)
    """
    if isinstance(label, str):
        delta = np.zeros(len(frame))
        delta[frame.index(label)] = 1.0
        return delta
    if isinstance(label, DSVector):
        if list(label.frame) != list(frame):
            raise ValueError(
                "pl_loss: label frame does not match prediction frame."
            )
        return label.contour()
    raise TypeError(
        f"pl_loss: label must be str (hard) or DSVector (soft), got {type(label).__name__}."
    )


def pl_loss(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> float:
    """
    Pl-based discrepancy between predictions and (hard or soft) labels.

        L(predictions, labels) = Σᵢ Σₖ (plᵢ(ωₖ) − δᵢ,ₖ)²

    where plᵢ is the contour function of prediction i and δᵢ,ₖ is the truth
    indicator (hard label, str) or the soft label's contour value
    (soft label, DSVector). Hard and soft labels can be mixed in the same
    call: a hard label is dispatched to the indicator form, a DSVector
    label is dispatched to the contour form.

    Lower is better. This is the criterion minimized when learning
    contextual correction parameters β.

    Parameters
    ----------
    predictions : sequence of DSVector
        BBA outputs, one per instance. All must share the same frame.
    labels : sequence of str or DSVector
        Ground truth: hard labels (atom names) or soft labels (BBA over
        the same frame). Can be mixed within the same call.

    Returns
    -------
    float
        The total squared discrepancy, summed over all instances.

    Raises
    ------
    ValueError
        If predictions and labels have different lengths, or if a soft
        label has a frame different from its prediction.
    TypeError
        If a label is neither str nor DSVector.

    References
    ----------
    Hard labels — Mercier et al. (2008), Eq. (2.24) of Mutmainah (2021) thesis.
    Soft labels — Mutmainah et al. (2019), SUM, Eq. (8); thesis Eq. (5.6).
    Used as a performance measure in Mutmainah et al. (2021), BELIEF.
    """
    preds = list(predictions)
    lbls  = list(labels)
    if len(preds) != len(lbls):
        raise ValueError(
            f"pl_loss: predictions and labels have different lengths "
            f"({len(preds)} vs {len(lbls)})."
        )

    total = 0.0
    for m_pred, label in zip(preds, lbls):
        pl_pred = m_pred.contour()
        delta   = _label_to_indicator(label, m_pred.frame)
        diff = pl_pred - delta
        total += float(np.dot(diff, diff))
    return total


def mean_pl_loss(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> float:
    """Mean pl-based discrepancy: :func:`pl_loss` divided by the number of instances.

    Returns 0.0 for an empty sequence (degenerate case)."""
    n = len(list(predictions))
    if n == 0:
        return 0.0
    # Re-iterate (sequences only): pl_loss again accepts iterables, but we
    # passed sequences here so this is fine.
    return pl_loss(predictions, labels) / n
