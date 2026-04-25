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

Usage
-----
    from evtools.dsvector import DSVector, Kind

    m = DSVector.from_focal(["a", "b"], {"a": 0.3, "b": 0.5})
    pl = m.to_pl()
"""

from . import conversions
from .dsvector import DSVector, Kind

__version__ = "0.2.0"
__all__ = ["conversions", "DSVector", "Kind"]