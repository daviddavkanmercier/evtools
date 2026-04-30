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
- Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2021). Improving an
  evidential source of information using contextual corrections depending
  on partial decisions. BELIEF 2021, pp. 247-256.
- Mutmainah, S. (2021). Learning to adjust an evidential source of information
  using partially labeled data and partial decisions. PhD thesis, Université
  d'Artois. Section 2.5.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple, Sequence, Union

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

        plᵢ(ωₖ) = 1 - βₖ · (1 - plSᵢ(ωₖ))

    Closed form (per atom, before clipping to [0, 1]):

        βₖ = Σᵢ (1 - plSᵢ(ωₖ))(1 - δᵢ,ₖ) / Σᵢ (1 - plSᵢ(ωₖ))²

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

        plᵢ(ωₖ) = 0.5 + (plSᵢ(ωₖ) - 0.5) · (2βₖ - 1)

    Closed form (per atom, before clipping):

        βₖ = Σᵢ (2·plSᵢ - 1)(plSᵢ + δᵢ - 1) / Σᵢ (2·plSᵢ - 1)²

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


# ---------------------------------------------------------------------------
# Soft-label generation (Algorithm 2 of Mutmainah 2021)
# ---------------------------------------------------------------------------

def hard_to_soft_labels(
    hard_labels: Sequence[str],
    frame: Sequence[str],
    *,
    mu: float = 0.5,
    var: float = 0.04,
    rng: "np.random.Generator | None" = None,
) -> list[DSVector]:
    """
    Generate soft labels from hard labels (Mutmainah 2021, Algorithm 2).

    For each instance:

    1. Draw p_i ~ Beta with mean *mu* and variance *var*.
    2. Draw b_i ~ Bernoulli(p_i).
    3. If b_i = 1, pick a uniformly random class k_i ∈ Ω and emit a soft
       label whose contour function is::

           plᵢ(ω_{k_i}) = 1,    plᵢ(ω_k) = p_i  for k ≠ k_i

       which corresponds to the simple MF
       :math:`m(\\{\\omega_{k_i}\\}) = 1 - p_i,\\ m(\\Omega) = p_i`.
    4. Otherwise (b_i = 0), keep the hard label as a categorical BBA.

    The procedure tends to produce labels that are all the more imprecise
    as the most plausible class differs from the true one, which is the
    intended behaviour described by the algorithm.

    Parameters
    ----------
    hard_labels : sequence of str
        True classes — one atom name per instance.
    frame : sequence of str
        The frame Ω. Each ``hard_labels[i]`` must be in ``frame``.
    mu : float, default 0.5
        Mean of the Beta distribution from which p_i is drawn.
    var : float, default 0.04
        Variance of the Beta distribution. Must satisfy
        ``var < mu * (1 - mu)`` (otherwise the Beta is undefined).
    rng : numpy.random.Generator, optional
        Random number generator. Defaults to ``numpy.random.default_rng()``,
        which means non-reproducible draws unless one is provided.

    Returns
    -------
    list[DSVector]
        One DSVector per instance, ready to be used as soft labels for
        :func:`evtools.metrics.pl_loss`, :func:`fit_cd`, :func:`fit_cr`,
        or :func:`fit_cn`.

    Raises
    ------
    ValueError
        If a hard label is not in the frame, or if ``var ≥ mu·(1-mu)``.

    References
    ----------
    Côme, E., Oukhellou, L., Denœux, T., Aknin, P. (2009). Learning from
    partially supervised data using mixture models and belief functions.
    Pattern Recognition, 42(3), 334-348.
    Quost, B., Denœux, T., Li, S. (2017). Parametric classification with
    soft labels using the evidential EM algorithm: linear discriminant
    analysis versus logistic regression. Advances in Data Analysis and
    Classification, 11(4), 659-690.
    Mutmainah, S. (2021). Learning to adjust an evidential source of
    information using partially labeled data and partial decisions. PhD
    thesis, Université d'Artois. Algorithm 2.
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert (mu, var) into the (α, β) shape parameters of Beta(α, β).
    common = mu * (1.0 - mu) / var - 1.0
    if common <= 0.0:
        raise ValueError(
            f"hard_to_soft_labels: variance {var} too large for mean {mu}; "
            f"requires var < mu·(1-mu) = {mu * (1 - mu)}."
        )
    alpha_shape = mu * common
    beta_shape  = (1.0 - mu) * common

    frame_list = list(frame)
    K = len(frame_list)
    soft_labels: list[DSVector] = []

    for label in hard_labels:
        if label not in frame_list:
            raise ValueError(
                f"hard_to_soft_labels: label {label!r} not in frame {frame_list}."
            )

        p_i = float(rng.beta(alpha_shape, beta_shape))
        b_i = bool(rng.binomial(1, p_i))

        if b_i:
            # Random "most plausible" class — may or may not be the true one.
            k_i = frame_list[int(rng.integers(K))]
            soft_labels.append(
                DSVector.simple(frame_list, frozenset({k_i}), beta=p_i)
            )
        else:
            # Keep the hard label as a categorical BBA.
            soft_labels.append(
                DSVector.from_focal(frame_list, {label: 1.0})
            )

    return soft_labels


# ---------------------------------------------------------------------------
# Per-group learning of contextual corrections
# (Algorithm 1 of Mutmainah 2021 — Chapter 4 with hard labels, Section 5.3
#  with soft labels; both share the same code path thanks to the polymorphic
#  pl_loss / fit_cd / fit_cr / fit_cn used internally.)
# ---------------------------------------------------------------------------

CorrectionKind = Literal["cd", "cr", "cn"]


class GroupCorrection(NamedTuple):
    """The correction selected for one partial-decision group.

    Attributes
    ----------
    kind : {"cd", "cr", "cn"}
        Which correction (contextual discount / reinforce / negate) attained
        the lowest pl_loss on the instances of this group.
    betas : dict[frozenset, float]
        The β parameters of the chosen correction, in the format expected by
        the corresponding ``evtools.corrections`` function.
    loss : float
        The pl_loss attained on the instances of this group after applying
        the chosen correction.
    """
    kind: CorrectionKind
    betas: dict
    loss: float


class GroupedCorrectionModel(NamedTuple):
    """A learned per-group correction model.

    Attributes
    ----------
    groups : dict[frozenset, GroupCorrection]
        Mapping from a partial decision (subset of Ω) to the chosen correction.
    fallback : GroupCorrection
        Correction fitted on the whole training set, used at predict time
        when an instance's partial decision was not seen during training.
    dominance : callable
        The partial-decision function used at training and prediction time
        (typically ``evtools.decision.strong_dominance`` or
        ``evtools.decision.weak_dominance``).
    """
    groups: dict
    fallback: GroupCorrection
    dominance: Callable[[DSVector], frozenset]


def _fit_best_correction(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
) -> GroupCorrection:
    """Fit CD, CR, CN; return the one with lowest pl_loss."""
    from .corrections import contextual_discount, contextual_reinforce, contextual_negate
    from .metrics import pl_loss

    candidates = (
        ("cd", fit_cd, contextual_discount),
        ("cr", fit_cr, contextual_reinforce),
        ("cn", fit_cn, contextual_negate),
    )
    best: GroupCorrection | None = None
    for kind, fit_fn, apply_fn in candidates:
        betas = fit_fn(predictions, labels)
        corrected = [apply_fn(p, betas) for p in predictions]
        loss = pl_loss(corrected, labels)
        if best is None or loss < best.loss:
            best = GroupCorrection(kind=kind, betas=betas, loss=loss)
    assert best is not None
    return best


def fit_per_group(
    predictions: Sequence[DSVector],
    labels: Sequence[Union[str, DSVector]],
    *,
    dominance: Callable[[DSVector], frozenset],
) -> GroupedCorrectionModel:
    """
    Per-group learning of contextual corrections.

    Implements Algorithm 1 of Mutmainah (2021), Chapter 4. The same algorithm
    extends naturally to soft labels (Section 5.3): pass DSVector labels and
    the underlying optimization minimizes Ẽ_pl instead of E_pl.

    Procedure:

    1. Group training instances by their partial decision (computed via
       *dominance*).
    2. For each group, fit CD, CR, CN parameters and keep the one attaining
       the lowest pl_loss on that group.
    3. Also fit a fallback correction on the full training set, used when
       an unseen partial decision occurs at predict time.

    Parameters
    ----------
    predictions : sequence of DSVector
        Source BBA outputs on the training set (must all share the same frame).
    labels : sequence of str or DSVector
        Ground truth — hard labels (str) or soft labels (DSVector). Hard
        and soft can be mixed.
    dominance : callable
        A function ``DSVector → frozenset`` returning the partial decision
        used to define groups. Typically :func:`evtools.decision.strong_dominance`
        or :func:`evtools.decision.weak_dominance`.

    Returns
    -------
    GroupedCorrectionModel
        The fitted model. Pass it to :func:`apply_per_group` to correct new
        instances.

    Raises
    ------
    ValueError
        If ``predictions`` and ``labels`` have different lengths.

    References
    ----------
    Mutmainah, S. (2021). Learning to adjust an evidential source of
    information using partially labeled data and partial decisions. PhD
    thesis, Université d'Artois. Algorithm 1 (Section 4.2) and Section 5.3.
    Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2021). Improving an
    evidential source of information using contextual corrections depending
    on partial decisions. BELIEF 2021, pp. 247-256 (hard-label version).
    Mutmainah, S., Hachour, S., Pichon, F., Mercier, D. (2019). On learning
    evidential contextual corrections from soft labels using a measure of
    discrepancy between contour functions. SUM 2019 (soft-label extension).
    """
    preds = list(predictions)
    lbls  = list(labels)
    if len(preds) != len(lbls):
        raise ValueError(
            f"fit_per_group: predictions and labels have different lengths "
            f"({len(preds)} vs {len(lbls)})."
        )
    if not preds:
        raise ValueError("fit_per_group: empty training set.")

    # Step 1: partition by partial decision.
    grouped_preds: dict[frozenset, list[DSVector]] = {}
    grouped_lbls:  dict[frozenset, list] = {}
    for p, lbl in zip(preds, lbls):
        d = dominance(p)
        grouped_preds.setdefault(d, []).append(p)
        grouped_lbls.setdefault(d, []).append(lbl)

    # Step 2: fit best correction for each group.
    groups: dict[frozenset, GroupCorrection] = {
        d: _fit_best_correction(grouped_preds[d], grouped_lbls[d])
        for d in grouped_preds
    }

    # Step 3: fallback (best correction on the whole training set).
    fallback = _fit_best_correction(preds, lbls)

    return GroupedCorrectionModel(
        groups=groups, fallback=fallback, dominance=dominance,
    )


def apply_per_group(
    model: GroupedCorrectionModel,
    predictions: Sequence[DSVector],
) -> list[DSVector]:
    """
    Apply a per-group correction model to new BBA outputs.

    For each prediction, the partial decision is computed using the model's
    ``dominance`` function and looked up in ``model.groups``. If the partial
    decision is in the model, the corresponding correction is applied;
    otherwise ``model.fallback`` is used.

    Parameters
    ----------
    model : GroupedCorrectionModel
        Result of :func:`fit_per_group`.
    predictions : sequence of DSVector
        New BBAs to correct.

    Returns
    -------
    list[DSVector]
        Corrected BBAs, one per input.

    References
    ----------
    Mutmainah, S. (2021). Algorithm 1 (predict step), Section 4.2.
    See also: Mutmainah et al. SUM (2019), BELIEF (2021).
    """
    from .corrections import contextual_discount, contextual_reinforce, contextual_negate

    apply_map = {
        "cd": contextual_discount,
        "cr": contextual_reinforce,
        "cn": contextual_negate,
    }
    out: list[DSVector] = []
    for p in predictions:
        d = model.dominance(p)
        gc = model.groups.get(d, model.fallback)
        out.append(apply_map[gc.kind](p, gc.betas))
    return out
