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
    Learning of contextual correction parameters from labeled data:
    fit_cd, fit_cr, fit_cn — closed-form least-squares minimizers of
    pl_loss (Pichon et al. 2016, Propositions 12, 14, 16). Hard or
    soft labels.

evtools.display
    Display functions: repr_ansi, repr_plain, repr_html, repr_latex.
    DSVector.__repr__ and _repr_html_ delegate here.

evtools.constants
    Numerical tolerance constants used throughout the library.

Usage
-----
    from evtools.dsvector import DSVector, Kind
    from evtools.combinations import crc, dempster
    from evtools.corrections import discount, contextual_discount

    m = DSVector.from_focal(["a", "b"], {"a": 0.3, "b": 0.5})
    m_disc = discount(m, alpha=0.4)
"""

from . import conversions
from . import combinations
from . import corrections
from . import decision
from . import metrics
from . import learning
from . import display
from .dsvector import DSVector, Kind

__version__ = "0.20.2"
__all__ = ["conversions", "combinations", "corrections", "decision", "metrics", "learning", "display", "DSVector", "Kind"]
