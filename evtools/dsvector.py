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
            if total > 1.0 + 1e-12:
                raise ValueError(
                    f"Total mass {total:.6g} exceeds 1."
                )
            if complete:
                remainder = 1.0 - total
                if remainder > 1e-12:
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

    def to_m(self)   -> "DSVector": return self.to(Kind.M)
    def to_bel(self) -> "DSVector": return self.to(Kind.BEL)
    def to_pl(self)  -> "DSVector": return self.to(Kind.PL)
    def to_b(self)   -> "DSVector": return self.to(Kind.B)
    def to_q(self)   -> "DSVector": return self.to(Kind.Q)
    def to_v(self)   -> "DSVector": return self.to(Kind.V)
    def to_w(self)   -> "DSVector": return self.to(Kind.W)

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

    # ANSI codes
    _RESET  = "\033[0m"
    _BOLD   = "\033[1m"
    _DIM    = "\033[2m"
    _CYAN   = "\033[36m"
    _BLUE   = "\033[34m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"

    # Bar characters (filled → empty)
    _BAR_FULL  = "█"
    _BAR_HALF  = "▌"
    _BAR_EMPTY = "░"

    _KIND_COLOR = {
        Kind.M:   "\033[36m",   # cyan
        Kind.BEL: "\033[34m",   # blue
        Kind.PL:  "\033[32m",   # green
        Kind.B:   "\033[33m",   # yellow
        Kind.Q:   "\033[35m",   # magenta
        Kind.V:   "\033[31m",   # red
        Kind.W:   "\033[91m",   # bright red
    }

    _KIND_LABEL = {
        Kind.M:   "Basic Belief Assignment",
        Kind.BEL: "Belief function",
        Kind.PL:  "Plausibility function",
        Kind.B:   "Commonality function",
        Kind.Q:   "Implicability function",
        Kind.V:   "Disjunctive weights",
        Kind.W:   "Conjunctive weights",
    }

    def _subset_label(self, subset: frozenset) -> str:
        if not subset:
            return "∅"
        return "{" + ", ".join(sorted(subset, key=self._frame.index)) + "}"

    def _bar(self, value: float, max_val: float, width: int = 16) -> str:
        """Return a colored progress bar proportional to value/max_val."""
        if max_val == 0:
            ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, abs(value) / max_val))
        filled = int(ratio * width * 2)  # half-block precision
        full   = filled // 2
        half   = filled % 2
        empty  = width - full - half
        filled_part = f"{self._GREEN}{self._BAR_FULL * full}{self._BAR_HALF * half}{self._RESET}"
        empty_part  = f"{self._DIM}{self._BAR_EMPTY * empty}{self._RESET}"
        return filled_part + empty_part

    def __repr__(self) -> str:
        B, R, D = self._BOLD, self._RESET, self._DIM
        kind_color = self._KIND_COLOR.get(self._kind, "")
        kind_label = self._KIND_LABEL.get(self._kind, self._kind.value)
        frame_str  = "{" + ", ".join(self._frame) + "}"
        n_focal    = len(self._sparse)

        # Header
        lines = [
            f"{B}DSVector{R}  "
            f"kind={kind_color}{B}{self._kind.value}{R}  "
            f"{D}({kind_label}){R}  "
            f"frame={B}{frame_str}{R}  "
            f"{D}{n_focal} focal element{'s' if n_focal != 1 else ''}{R}"
        ]

        if not self._sparse:
            lines.append(f"  {D}(empty){R}")
            return "\n".join(lines)

        # Column widths
        col_subset = max(len(self._subset_label(s)) for s in self._sparse)
        col_subset = max(col_subset, 6)

        # Separator
        sep = f"  {D}{'─' * (col_subset + 2)}{'─' * 10}{'─' * 19}{R}"
        header_row = (
            f"  {B}{'Subset':<{col_subset}}  {'Value':>8}  {'':19}{R}"
        )
        lines += ["", header_row, sep]

        # Values sorted by index
        sorted_items = sorted(
            self._sparse.items(),
            key=lambda kv: _subset_index(kv[0], self._frame),
        )

        # Max absolute value for bar scaling
        max_val = max(abs(v) for _, v in sorted_items) if sorted_items else 1.0

        for subset, value in sorted_items:
            label = self._subset_label(subset)
            bar   = self._bar(value, max_val)
            lines.append(
                f"  {self._BLUE}{label:<{col_subset}}{R}"
                f"  {B}{value:>8.4f}{R}"
                f"  {bar}"
            )

        lines.append(sep)

        # Footer: total (only meaningful for Kind.M)
        total = sum(self._sparse.values())
        if self._kind == Kind.M:
            ok = abs(total - 1.0) < 1e-9
            total_color = self._GREEN if ok else self._YELLOW
            lines.append(
                f"  {D}{'Total':<{col_subset}}  "
                f"{total_color}{B}{total:>8.4f}{R}"
            )

        return "\n".join(lines)
