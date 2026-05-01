"""
Evidential classifiers — pattern classification based on Dempster-Shafer
theory of evidence.

Available classifiers
---------------------
EkNN
    Evidential k-nearest neighbor classifier (Denoeux 1995, Zouhal & Denoeux
    1998). Each of the K nearest neighbors of a test instance produces a
    simple BBA, and the K BBAs are combined with Dempster's rule. The
    per-class parameters γ_q can be optimized to minimize a pl-based
    discrepancy on the training set.

Future
------
ENN — neural network classifier with learned prototypes (Denoeux 2000).
"""

from .eknn import EkNN

__all__ = ["EkNN"]
