"""
Performance metrics for evidential predictions.

Three families of metrics are provided:

Set-valued predictions (partial decisions)
------------------------------------------
Score a partial decision d ⊆ Ω against a true class ω. Typical inputs are
the output of :func:`evtools.decision.strong_dominance` or :func:`weak_dominance`.

    discounted_accuracy(d, ω)              — x = I(ω ∈ d) / |d|
    u65(d, ω)                              — 1.6·x − 0.6·x²
    u80(d, ω)                              — 2.2·x − 1.2·x²
    utility_score(d, ω, a=…, b=…)          — generic a·x − b·x²

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
- Mutmainah, S. (2021). PhD thesis, Université d'Artois. Section 3.4.
"""

from __future__ import annotations

from typing import Iterable

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
