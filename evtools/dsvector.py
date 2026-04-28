"""
DSVector — a vector on 2^Ω representing any belief function representation.

All standard representations in the Dempster-Shafer theory of evidence
(mass, belief, plausibility, commonality, implicability, conjunctive weights,
disjunctive weights) are vectors indexed by the subsets of a frame Ω.
DSVector provides a unified container for all of them.

Internally, the sparse representation (dict) is the master. The dense
numpy array is computed on demand and cached.

Supported kinds
---------------
Kind.M   : Basic Belief Assignment (mass function)
Kind.BEL : Belief function
Kind.PL  : Plausibility function
Kind.B   : Commonality function
Kind.Q   : Implicability function
Kind.V   : Disjunctive weight function
Kind.W   : Conjunctive weight function

Constructors
------------
DSVector.from_focal(frame, focal, kind=Kind.M)  — human-friendly dict of focal elements
DSVector.from_dense(frame, array, kind=Kind.M)  — numpy array
DSVector.from_sparse(frame, sparse, kind=Kind.M) — dict[frozenset, float]

Conversions
-----------
v.to(kind)  — returns a new DSVector of the requested kind
v.to_m(), v.to_bel(), v.to_pl(), ...  — convenience shortcuts
"""

from __future__ import annotations

from .constants import ZERO_MASS, MASS_TOL, VALID_TOL, DISPLAY_TOL

from enum import Enum
from typing import Iterator
import numpy as np

from .conversions import (
    mtob, mtobel, mtopl, mtoq, mtov, mtow,
    btom, beltom, pltom, qtom, vtom, wtom,
    btobel, btopl, btoq, btov, btow,
    beltob, beltopl, beltoq, beltov, beltow,
    pltob, pltobel, pltom, pltoq, pltov, pltow,
    qtob, qtobel, qtom, qtopl, qtov, qtow,
    vtob, vtobel, vtom, vtopl, vtoq, vtow,
    wtob, wtobel, wtom, wtopl, wtoq, wtov,
)


# ---------------------------------------------------------------------------
# Kind enum
# ---------------------------------------------------------------------------

class Kind(Enum):
    """
    Enumeration of the supported belief function representations.

    Each kind corresponds to a standard mathematical function defined
    on the power set 2^Ω.
    """
    M   = "m"    # Basic Belief Assignment
    BEL = "bel"  # Belief function
    PL  = "pl"   # Plausibility function
    B   = "b"    # Commonality function
    Q   = "q"    # Implicability function
    V   = "v"    # Disjunctive weight function
    W   = "w"    # Conjunctive weight function


# ---------------------------------------------------------------------------
# Conversion dispatch table:  (source Kind, target Kind) → function
# ---------------------------------------------------------------------------

_CONVERT = {
    (Kind.M,   Kind.BEL): mtobel,
    (Kind.M,   Kind.PL):  mtopl,
    (Kind.M,   Kind.B):   mtob,
    (Kind.M,   Kind.Q):   mtoq,
    (Kind.M,   Kind.V):   mtov,
    (Kind.M,   Kind.W):   mtow,

    (Kind.BEL, Kind.M):   beltom,
    (Kind.BEL, Kind.PL):  beltopl,
    (Kind.BEL, Kind.B):   beltob,
    (Kind.BEL, Kind.Q):   beltoq,
    (Kind.BEL, Kind.V):   beltov,
    (Kind.BEL, Kind.W):   beltow,

    (Kind.PL,  Kind.M):   pltom,
    (Kind.PL,  Kind.BEL): pltobel,
    (Kind.PL,  Kind.B):   pltob,
    (Kind.PL,  Kind.Q):   pltoq,
    (Kind.PL,  Kind.V):   pltov,
    (Kind.PL,  Kind.W):   pltow,

    (Kind.B,   Kind.M):   btom,
    (Kind.B,   Kind.BEL): btobel,
    (Kind.B,   Kind.PL):  btopl,
    (Kind.B,   Kind.Q):   btoq,
    (Kind.B,   Kind.V):   btov,
    (Kind.B,   Kind.W):   btow,

    (Kind.Q,   Kind.M):   qtom,
    (Kind.Q,   Kind.BEL): qtobel,
    (Kind.Q,   Kind.PL):  qtopl,
    (Kind.Q,   Kind.B):   qtob,
    (Kind.Q,   Kind.V):   qtov,
    (Kind.Q,   Kind.W):   qtow,

    (Kind.V,   Kind.M):   vtom,
    (Kind.V,   Kind.BEL): vtobel,
    (Kind.V,   Kind.PL):  vtopl,
    (Kind.V,   Kind.B):   vtob,
    (Kind.V,   Kind.Q):   vtoq,
    (Kind.V,   Kind.W):   vtow,

    (Kind.W,   Kind.M):   wtom,
    (Kind.W,   Kind.BEL): wtobel,
    (Kind.W,   Kind.PL):  wtopl,
    (Kind.W,   Kind.B):   wtob,
    (Kind.W,   Kind.Q):   wtoq,
    (Kind.W,   Kind.V):   wtov,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subset_index(subset: frozenset, frame: list[str]) -> int:
    """Return the integer index of a subset in the power-set ordering."""
    idx = 0
    for atom in subset:
        idx += 2 ** frame.index(atom)
    return idx


def _index_to_subset(idx: int, frame: list[str]) -> frozenset:
    """Return the frozenset corresponding to a power-set index."""
    return frozenset(frame[bit] for bit in range(len(frame)) if idx & (1 << bit))


def _dense_to_sparse(
    array: np.ndarray,
    frame: list[str],
    *,
    tol: float = 0.0,
) -> dict[frozenset, float]:
    """Convert a dense array to a sparse dict, dropping values ≤ tol."""
    return {
        _index_to_subset(i, frame): float(array[i])
        for i in range(len(array))
        if abs(array[i]) > tol
    }


def _sparse_to_dense(
    sparse: dict[frozenset, float],
    frame: list[str],
) -> np.ndarray:
    """Convert a sparse dict to a dense numpy array."""
    n = len(frame)
    array = np.zeros(2 ** n)
    for subset, value in sparse.items():
        array[_subset_index(subset, frame)] = value
    return array


def _focal_index_from_str(key: str, frame: list[str], sep: str) -> frozenset:
    """Parse a string focal element key into a frozenset."""
    if key.strip() == "":
        return frozenset()
    atoms = [a.strip() for a in key.split(sep)]
    for atom in atoms:
        if atom not in frame:
            raise ValueError(
                f"Atom '{atom}' is not in the frame {frame}."
            )
    return frozenset(atoms)


# ---------------------------------------------------------------------------
# DSVector
# ---------------------------------------------------------------------------

class DSVector:
    """
    A vector on 2^Ω representing a belief function in any standard form.

    Parameters
    ----------
    frame : list[str]
        Ordered list of atoms of the frame of discernment.
    sparse : dict[frozenset, float]
        Sparse representation (master). Zero entries may be omitted.
    kind : Kind
        Which belief function representation this vector holds.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        frame: list[str],
        sparse: dict[frozenset, float],
        kind: Kind = Kind.M,
    ) -> None:
        self._frame: list[str] = list(frame)
        self._sparse: dict[frozenset, float] = dict(sparse)
        self._kind: Kind = kind
        self._dense_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_focal(
        cls,
        frame: list[str],
        focal: dict[str, float],
        *,
        kind: Kind = Kind.M,
        sep: str = ",",
        complete: bool = True,
    ) -> "DSVector":
        """
        Build from a human-readable focal element dict.

        Parameters
        ----------
        frame : list[str]
            Ordered list of atoms.
        focal : dict[str, float]
            Mapping from focal element string to value.
            Use "" for the empty set, e.g. {"": 0.1, "a": 0.3, "a,b": 0.6}.
        kind : Kind
            Interpretation of the values (default: Kind.M).
        sep : str
            Separator used in focal element keys (default ",").
        complete : bool
            For Kind.M only: if True (default), missing mass is assigned to Ω.
        """
        sparse: dict[frozenset, float] = {}

        for key, value in focal.items():
            if kind == Kind.M and value < 0:
                raise ValueError(
                    f"Mass values must be non-negative, got {value} for '{key}'."
                )
            subset = _focal_index_from_str(key, frame, sep)
            sparse[subset] = sparse.get(subset, 0.0) + value

        if kind == Kind.M:
            total = sum(sparse.values())
            if total > 1.0 + MASS_TOL:
                raise ValueError(
                    f"Total mass {total:.6g} exceeds 1."
                )
            if complete:
                remainder = 1.0 - total
                if remainder > MASS_TOL:
                    omega = frozenset(frame)
                    sparse[omega] = sparse.get(omega, 0.0) + remainder

        return cls(frame, sparse, kind)

    @classmethod
    def from_dense(
        cls,
        frame: list[str],
        array: np.ndarray,
        *,
        kind: Kind = Kind.M,
        tol: float = 0.0,
    ) -> "DSVector":
        """
        Build from a dense numpy array.

        The array follows the binary index ordering of Smets (2002): index i
        corresponds to the subset whose members are the atoms at positions
        indicated by the 1-bits of i.

        For frame = ["a", "b", "c"] the 8 indices map to:
            0 (000) → ∅
            1 (001) → {a}
            2 (010) → {b}
            3 (011) → {a, b}
            4 (100) → {c}
            5 (101) → {a, c}
            6 (110) → {b, c}
            7 (111) → {a, b, c}

        Parameters
        ----------
        frame : list[str]
            Ordered list of atoms — their order defines the bit positions.
        array : np.ndarray
            Dense vector of length 2 ** len(frame), in binary index order.
        kind : Kind
            Interpretation of the values (default: Kind.M).
        tol : float
            Values whose absolute value is ≤ tol are dropped from sparse.

        References
        ----------
        Smets, P. (2002). The application of the matrix calculus to belief
        functions. International Journal of Approximate Reasoning, 31, 1-30.
        """
        n = len(frame)
        expected = 2 ** n
        if len(array) != expected:
            raise ValueError(
                f"Array length {len(array)} does not match 2^{n} = {expected}."
            )
        sparse = _dense_to_sparse(array, frame, tol=tol)
        obj = cls(frame, sparse, kind)
        obj._dense_cache = np.array(array, dtype=float)
        return obj

    @classmethod
    def from_sparse(
        cls,
        frame: list[str],
        sparse: dict[frozenset, float],
        *,
        kind: Kind = Kind.M,
    ) -> "DSVector":
        """
        Build from a dict mapping frozensets to values.

        Parameters
        ----------
        frame : list[str]
            Ordered list of atoms.
        sparse : dict[frozenset, float]
            Mapping from frozenset subsets of frame to values.
        kind : Kind
            Interpretation of the values (default: Kind.M).
        """
        for subset in sparse:
            for atom in subset:
                if atom not in frame:
                    raise ValueError(
                        f"Atom '{atom}' in subset {set(subset)} "
                        f"is not in the frame {frame}."
                    )
        return cls(frame, dict(sparse), kind)

    @classmethod
    def simple(
        cls,
        frame: list[str],
        subset: frozenset,
        beta: float,
    ) -> "DSVector":
        """
        Build a simple MF A^β (positive simple mass function).

        Focal sets: Ω with mass β, A=subset with mass 1−β.

        Used in the Contextual Reinforcement (CR) and Contextual
        De-Reinforcement (CdR) correction mechanisms.

        Parameters
        ----------
        frame : list[str]
            Ordered list of atoms.
        subset : frozenset
            The focal set A ⊂ Ω (must be a proper subset of Ω).
        beta : float
            Mass assigned to Ω ∈ [0, 1]. Mass 1−β is assigned to A.

        Returns
        -------
        DSVector
            A BBA with at most two focal elements: A and Ω.

        References
        ----------
        Denoeux, T. (2008). Artificial Intelligence, 172, 234-264.
        Pichon et al. (2016). IJAR, 72, 4-42.
        """
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"simple: beta must be in [0, 1], got {beta}.")
        omega = frozenset(frame)
        sparse: dict[frozenset, float] = {}
        if beta > ZERO_MASS:
            sparse[omega] = beta
        if 1.0 - beta > ZERO_MASS:
            sparse[subset] = 1.0 - beta
        return cls(frame, sparse, Kind.M)

    @classmethod
    def negative_simple(
        cls,
        frame: list[str],
        subset: frozenset,
        beta: float,
    ) -> "DSVector":
        """
        Build a negative simple MF A_β (negative simple mass function).

        Focal sets: ∅ with mass β, θ=subset with mass 1−β.
        This is a subnormal BBA (m(∅) > 0 when β > 0).

        Used in the Contextual Discounting (CD) and Contextual
        De-Discounting (CdD) correction mechanisms.

        Parameters
        ----------
        frame : list[str]
            Ordered list of atoms.
        subset : frozenset
            The focal set θ ⊆ Ω.
        beta : float
            Mass assigned to ∅ ∈ [0, 1]. Mass 1−β is assigned to θ.

        Returns
        -------
        DSVector
            A subnormal BBA with at most two focal elements: ∅ and θ.

        References
        ----------
        Denoeux, T. (2008). Artificial Intelligence, 172, 234-264.
        Pichon et al. (2016). IJAR, 72, 4-42.
        """
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"negative_simple: beta must be in [0, 1], got {beta}.")
        sparse: dict[frozenset, float] = {}
        if beta > ZERO_MASS:
            sparse[frozenset()] = beta
        if 1.0 - beta > ZERO_MASS:
            sparse[subset] = 1.0 - beta
        return cls(frame, sparse, Kind.M)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame(self) -> list[str]:
        """Ordered list of atoms of the frame of discernment."""
        return list(self._frame)

    @property
    def kind(self) -> Kind:
        """Which belief function representation this vector holds."""
        return self._kind

    @property
    def sparse(self) -> dict[frozenset, float]:
        """Sparse representation: dict mapping frozensets to values."""
        return dict(self._sparse)

    @property
    def dense(self) -> np.ndarray:
        """Dense numpy array of length 2^n (computed and cached on first access)."""
        if self._dense_cache is None:
            self._dense_cache = _sparse_to_dense(self._sparse, self._frame)
        return self._dense_cache.copy()

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the frame."""
        return len(self._frame)

    @property
    def n_focal(self) -> int:
        """Number of non-zero focal elements."""
        return len(self._sparse)

    @property
    def is_valid(self) -> bool:
        """
        Whether this DSVector is a valid BBA (for Kind.M only).

        A BBA is valid if:
        - all masses are non-negative
        - masses sum to 1

        For other kinds (bel, pl, b, q, v, w), always returns True since
        validity constraints differ and are not checked here.

        This property is useful after inverse operations (CdD, CdR) which
        may produce functions that are not valid BBAs.
        """
        if self._kind != Kind.M:
            return True
        if any(v < -VALID_TOL for v in self._sparse.values()):
            return False
        total = sum(self._sparse.values())
        return abs(total - 1.0) < VALID_TOL

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to(self, kind: Kind) -> "DSVector":
        """
        Convert to another belief function representation.

        Parameters
        ----------
        kind : Kind
            Target representation.

        Returns
        -------
        DSVector
            A new DSVector of the requested kind.
        """
        if kind == self._kind:
            return DSVector(self._frame, self._sparse, self._kind)

        key = (self._kind, kind)
        if key not in _CONVERT:
            raise ValueError(
                f"No direct conversion from {self._kind} to {kind}."
            )
        result_dense = _CONVERT[key](self.dense)
        return DSVector.from_dense(self._frame, result_dense, kind=kind)

    def to_betp(self) -> np.ndarray:
        """
        Pignistic probability transformation (BetP).

        Returns a probability vector of length n (one value per atom),
        not a DSVector. The BBA must not be fully contradictory (m(∅) < 1).

        Returns
        -------
        np.ndarray
            BetP vector of length n = len(self.frame).

        Raises
        ------
        ValueError
            If the BBA is of wrong kind, or if m(∅) = 1.

        References
        ----------
        Smets, P., Kennes, R. (1994). Artificial Intelligence, 66(2), 191-234.
        """
        if self._kind != Kind.M:
            raise ValueError(
                f"to_betp: kind is '{self._kind.value}', expected 'm'. "
                "Convert with .to_m() first."
            )
        from .conversions import betp
        return betp(self.dense)

    def to_plp(self) -> np.ndarray:
        """
        Plausibility probability transformation (PlP).

        Returns a probability vector of length n (one value per atom),
        not a DSVector, by normalizing the plausibility of singletons.

        Returns
        -------
        np.ndarray
            PlP vector of length n = len(self.frame).

        Raises
        ------
        ValueError
            If the BBA is of wrong kind, or if all singleton plausibilities
            are zero.

        References
        ----------
        Cobb, B.R., Shenoy, P.P. (2006). IJAR, 41(3), 314-330.
        """
        if self._kind != Kind.M:
            raise ValueError(
                f"to_plp: kind is '{self._kind.value}', expected 'm'. "
                "Convert with .to_m() first."
            )
        from .conversions import plp
        return plp(self.dense)

    def to_m(self)   -> "DSVector":
        """Convert to mass function (Kind.M)."""
        return self.to(Kind.M)

    def to_bel(self) -> "DSVector":
        """Convert to belief function (Kind.BEL)."""
        return self.to(Kind.BEL)

    def to_pl(self)  -> "DSVector":
        """Convert to plausibility function (Kind.PL)."""
        return self.to(Kind.PL)

    def to_b(self)   -> "DSVector":
        """Convert to commonality function (Kind.B)."""
        return self.to(Kind.B)

    def to_q(self)   -> "DSVector":
        """Convert to implicability function (Kind.Q)."""
        return self.to(Kind.Q)

    def to_v(self)   -> "DSVector":
        """Convert to disjunctive weight function (Kind.V). Requires subnormal BBA (m(∅) > 0)."""
        return self.to(Kind.V)

    def to_w(self)   -> "DSVector":
        """Convert to conjunctive weight function (Kind.W). Requires non-dogmatic BBA (m(Ω) > 0)."""
        return self.to(Kind.W)

    # ------------------------------------------------------------------
    # Combination operators
    # ------------------------------------------------------------------

    def __and__(self, other: "DSVector") -> "DSVector":
        """CRC: m1 & m2  (sparse method by default)."""
        from .combinations import crc
        return crc(self, other)

    def __matmul__(self, other: "DSVector") -> "DSVector":
        """Dempster: m1 @ m2  (sparse method by default)."""
        from .combinations import dempster
        return dempster(self, other)

    def __or__(self, other: "DSVector") -> "DSVector":
        """DRC: m1 | m2  (sparse method by default)."""
        from .combinations import drc
        return drc(self, other)

    # ------------------------------------------------------------------
    # Iteration and access
    # ------------------------------------------------------------------

    def __getitem__(self, subset: frozenset) -> float:
        """Return the value for a given subset (0.0 if not a focal element)."""
        return self._sparse.get(subset, 0.0)

    def __iter__(self) -> Iterator[tuple[frozenset, float]]:
        """Iterate over (subset, value) pairs of non-zero focal elements."""
        return iter(self._sparse.items())

    def __len__(self) -> int:
        """Number of non-zero focal elements."""
        return len(self._sparse)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, fmt: str = "ansi") -> str:
        """
        Render this DSVector in the requested format.

        Parameters
        ----------
        fmt : str
            Output format. One of:
            - "ansi"  : colored terminal output (default)
            - "plain" : plain text, no colors
            - "html"  : HTML table for Jupyter notebooks
            - "latex" : LaTeX tabular for academic papers

        Returns
        -------
        str
            Formatted string in the requested format.
        """
        from .display import repr_ansi, repr_plain, repr_html, repr_latex
        formats = {
            "ansi":  repr_ansi,
            "plain": repr_plain,
            "html":  repr_html,
            "latex": repr_latex,
        }
        if fmt not in formats:
            raise ValueError(
                f"Unknown format '{fmt}'. Choose from: {list(formats.keys())}"
            )
        return formats[fmt](self)

    def __repr__(self) -> str:
        """Return colored ANSI terminal representation."""
        from .display import repr_ansi
        return repr_ansi(self)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        from .display import repr_html
        return repr_html(self)

    def display_all(self, fmt: str = "ansi") -> str:
        """
        Render all representations in a single table.

        Columns: m, bel, pl, b, q — plus v and w if the BBA is subnormal.
        Rows: all subsets with at least one non-zero value.

        Parameters
        ----------
        fmt : str
            Output format: "ansi" (default), "plain", "html", or "latex".

        Returns
        -------
        str
            Formatted string in the requested format.
        """
        from .display import display_all
        return display_all(self, fmt)

