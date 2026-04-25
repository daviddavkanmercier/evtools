"""
evtools — Evidence Theory Tools
================================

A growing collection of utilities for working with belief functions
in the Dempster-Shafer theory of evidence.

Submodules
----------
evtools.conversions
    Conversion functions between all standard belief function
    representations (m, bel, pl, b, q, v, w).

Usage
-----
Import the submodule::

    from evtools import conversions
    conversions.mtob(m)

Or import functions directly::

    from evtools.conversions import mtob, mtopl
"""

from . import conversions
from . import mass as mass_module
from .mass import mass, frame_labels

__version__ = "0.1.0"
__all__ = ["conversions", "mass", "frame_labels"]
