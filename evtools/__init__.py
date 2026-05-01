"""
evtools — Evidence Theory Tools
================================

A growing collection of utilities for working with belief functions
in the Dempster-Shafer theory of evidence.

Submodules
----------
evtools.dsvector
    DSVector — unified container for any belief function representation,
    with sparse and dense modes.

evtools.conversions
    Low-level conversion functions between all standard representations
    (m, bel, pl, b, q, v, w) via the Fast Möbius Transform.

evtools.combinations
    Combination rules: CRC, Dempster, DRC, Cautious, Bold.

evtools.corrections
    Correction mechanisms: discounting, reinforcement, negating and
    their contextual and inverse variants.

evtools.decision
    Decision criteria: maximin, maximax, pignistic_decision, plp_decision,
    probability_decision, hurwicz, strong_dominance, weak_dominance.

evtools.metrics
    Performance metrics for evaluating decisions and predictions:
    discounted_accuracy, u65, u80, utility_score, pl_loss (E_pl/Ẽ_pl),
    plus mean_* aggregators.

evtools.learning
    Learning of contextual correction parameters from labeled data.
    Per-correction closed-form fits: fit_cd, fit_cr, fit_cn (Pichon
    et al. 2016, Propositions 12, 14, 16). Per-group learning:
    fit_per_group / apply_per_group (Mutmainah 2021, Algorithm 1 —
    Chapter 4 hard labels and Section 5.3 soft labels). Soft-label
    synthesis: hard_to_soft_labels (Mutmainah 2021, Algorithm 2).

evtools.display
    Display functions: to_string, to_ansi, to_html, to_latex (each
    accepts ``all_kinds=False``). DSVector.__repr__ delegates to
    to_ansi, _repr_html_ to to_html.

evtools.classifiers
    Evidential classifiers based on belief functions. Currently provides
    EkNN (Denoeux 1995, Zouhal & Denoeux 1998). ENN (Denoeux 2000)
    planned.

evtools.constants
    Numerical tolerance constants used throughout the library.

Usage
-----
    from evtools.dsvector import DSVector, Kind
    from evtools.combinations import crc, dempster
    from evtools.corrections import discount, contextual_discount

    m = DSVector.from_focal(["a", "b"], {"a": 0.3, "b": 0.5})
    m_disc = discount(m, beta=0.6)
"""

from . import conversions
from . import combinations
from . import corrections
from . import decision
from . import metrics
from . import learning
from . import display
from . import classifiers
from .dsvector import DSVector, Kind

__version__ = "0.24.0"
__all__ = ["conversions", "combinations", "corrections", "decision", "metrics", "learning", "display", "classifiers", "DSVector", "Kind"]
