"""
Numerical constants used throughout evtools.

All tolerance thresholds are defined here to ensure consistency
across modules. Import from this module rather than using
numeric literals directly in the code.
"""

# Values strictly below this are treated as zero in sparse representation.
# Used when dropping near-zero entries from dicts and when testing whether
# a mass or beta value is strictly positive.
ZERO_MASS: float = 1e-15

# Tolerance for validating mass constraints.
# Used when checking that the total mass does not exceed 1.0, and when
# computing the remainder to assign to Ω in from_focal (complete=True).
MASS_TOL: float = 1e-12

# Tolerance for floating-point comparisons of belief-function values.
# Used in is_valid (verifying all masses are non-negative and sum to 1)
# and in dominance comparisons (strong/weak dominance in evtools.decision).
VALID_TOL: float = 1e-10

# Tolerance for display.
# Used in display.py to decide whether the total is shown as valid (green)
# or potentially invalid (orange).
DISPLAY_TOL: float = 1e-9
