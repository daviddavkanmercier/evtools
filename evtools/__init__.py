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
from .dsvector import DSVector, Kind

__version__ = "0.5.0"
__all__ = ["conversions", "combinations", "corrections", "DSVector", "Kind"]
