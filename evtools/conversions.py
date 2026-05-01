"""
Conversion functions between the different representations of belief functions
in the Dempster-Shafer theory of evidence.

All functions operate on plain numpy arrays of length 2^n, where n is the
number of atoms in the frame of discernment.

Binary index ordering (Smets 2002)
-----------------------------------
Index i in the array corresponds to the subset whose members are the atoms
at the bit positions set in i. For frame = [x1, x2, ..., xn]:

    index 0  (00...0) → ∅
    index 1  (00...1) → {x1}
    index 2  (00..10) → {x2}
    index 3  (00..11) → {x1, x2}
    ...
    index 2^n - 1    → {x1, x2, ..., xn} = Ω

This ordering is essential for the Fast Möbius Transform (FMT) algorithm
that underlies all conversions in this module. See Smets (2002, Section 3)
for the full derivation and MatLab code.

Conditioning matrices
---------------------
conditioning_matrix(frame, a)    — C_A specialization matrix for conditioning on A
deconditioning_matrix(frame, a)  — D_A generalization matrix for deconditioning on A

Probability transformations
---------------------------
betp(m)   — Pignistic probability BetP (vector of length n)
plp(m)    — Plausibility probability PlP (vector of length n)

Supported representations:
    - m   : Basic Belief Assignment (mass function)
    - b   : implicability function,  b(A) = Σ_{B⊆A} m(B)
    - bel : belief function,         bel(A) = b(A) - m(∅)
    - pl  : plausibility function,   pl(A) = 1 - b(Ā)
    - q   : commonality function,    q(A) = Σ_{B⊇A} m(B)
    - v   : disjunctive weight function  (Denoeux 2008)
    - w   : conjunctive weight function  (Denoeux 2008)

Each function is named ``<source>to<target>``, e.g. ``mtob`` converts a mass
function to its implicability function.

References:
    Smets, P. (2002). The application of the matrix calculus to belief
    functions. International Journal of Approximate Reasoning, 31, 1-30.

    Denoeux, T. (2008). Conjunctive and disjunctive combination of belief
    functions induced by non-distinct bodies of evidence. Artificial
    Intelligence, 172, 234-264.
"""
import numpy as np
from .constants import ZERO_MASS


# ---------------------------------------------------------------------------
# bto* — from implicability function
# ---------------------------------------------------------------------------

def btobel(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to belief function *bel*.

    bel(A) = b(A) - m(∅)
    """
    return b - b[0]


def btom(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to mass function *m* using the FMT (Smets)."""
    b = np.copy(b)
    lb = len(b)
    natoms = int(np.round(np.log2(lb)))
    for step in range(1, natoms + 1):
        i124 = 2 ** (step - 1)
        i842 = 2 ** (natoms + 1 - step)
        i421 = 2 ** (natoms - step)
        b = b.reshape(i124, i842, order="F")
        b[:, np.arange(1, i421 + 1) * 2 - 1] -= b[:, np.arange(1, i421 + 1) * 2 - 2]
    return b.reshape(1, lb, order="F").ravel()


def btopl(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to plausibility function *pl*.

    pl(A) = 1 - b(Ā)
    """
    b = np.copy(b)
    lb = len(b)
    b = b.reshape(1, lb, order="F")
    b = b[0][-1] - np.fliplr(b).ravel()
    b[0] = 0
    return b


def btoq(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to commonality function *q*."""
    return pltoq(btopl(b))


def btov(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to disjunctive weight function *v* (Denoeux 2008)."""
    v = np.exp(-btom(np.log(b)))
    v[0] = 1
    return v


def btow(b: np.ndarray) -> np.ndarray:
    """Convert implicability *b* to conjunctive weight function *w*."""
    return qtow(btoq(b))


# ---------------------------------------------------------------------------
# belto* — from belief function
# ---------------------------------------------------------------------------

def beltob(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to implicability *b*.

    b(A) = bel(A) + m(∅)
    """
    m_emptyset = 1 - bel[-1]
    return bel + m_emptyset


def beltom(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to mass function *m*."""
    return btom(beltob(bel))


def beltopl(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to plausibility function *pl*."""
    return btopl(bel)


def beltoq(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to commonality function *q*."""
    return btoq(bel)


def beltov(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to disjunctive weight function *v*."""
    return btov(beltob(bel))


def beltow(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to conjunctive weight function *w*."""
    return qtow(beltoq(bel))


# ---------------------------------------------------------------------------
# mto* — from mass function
# ---------------------------------------------------------------------------

def mtob(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to implicability *b* using the FMT (Smets)."""
    m = np.copy(m)
    lm = len(m)
    natoms = int(np.round(np.log2(lm)))
    for step in range(1, natoms + 1):
        i124 = 2 ** (step - 1)
        i842 = 2 ** (natoms + 1 - step)
        i421 = 2 ** (natoms - step)
        m = m.reshape(i124, i842, order="F")
        m[:, np.arange(1, i421 + 1) * 2 - 1] += m[:, np.arange(1, i421 + 1) * 2 - 2]
    return m.reshape(1, lm, order="F").ravel()


def mtobel(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to belief function *bel*."""
    return btobel(mtob(m))


def mtopl(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to plausibility function *pl*."""
    return btopl(mtob(m))


def mtoq(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to commonality function *q* using the FMT (Smets)."""
    m = np.copy(m)
    lm = len(m)
    natoms = int(np.round(np.log2(lm)))
    for step in range(1, natoms + 1):
        i124 = 2 ** (step - 1)
        i842 = 2 ** (natoms + 1 - step)
        i421 = 2 ** (natoms - step)
        m = m.reshape(i124, i842, order="F")
        m[:, np.arange(1, i421 + 1) * 2 - 2] += m[:, np.arange(1, i421 + 1) * 2 - 1]
    return m.reshape(1, lm, order="F").ravel()


def mtov(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to disjunctive weight function *v*."""
    return btov(mtob(m))


def mtow(m: np.ndarray) -> np.ndarray:
    """Convert mass function *m* to conjunctive weight function *w*."""
    return qtow(mtoq(m))


# ---------------------------------------------------------------------------
# plto* — from plausibility function
# ---------------------------------------------------------------------------

def pltob(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to implicability *b*.

    b(A) = 1 - pl(Ā)
    """
    pl = np.copy(pl)
    lpl = len(pl)
    pl = pl.reshape(1, lpl, order="F")
    return 1 - np.fliplr(pl).ravel()


def pltobel(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to belief function *bel*."""
    return btobel(pltob(pl))


def pltom(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to mass function *m*."""
    return btom(pltob(pl))


def pltoq(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to commonality function *q* (Smets 2002)."""
    q = np.abs(btom(pl))
    q[0] = 1
    return q


def pltov(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to disjunctive weight function *v*."""
    return btov(pltob(pl))


def pltow(pl: np.ndarray) -> np.ndarray:
    """Convert plausibility function *pl* to conjunctive weight function *w*."""
    return qtow(pltoq(pl))


# ---------------------------------------------------------------------------
# qto* — from commonality function
# ---------------------------------------------------------------------------

def qtob(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to implicability *b*."""
    return pltob(qtopl(q))


def qtobel(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to belief function *bel*."""
    return btobel(qtob(q))


def qtom(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to mass function *m* using the FMT (Smets)."""
    q = np.copy(q)
    lq = len(q)
    natoms = int(np.round(np.log2(lq)))
    for step in range(1, natoms + 1):
        i124 = 2 ** (step - 1)
        i842 = 2 ** (natoms + 1 - step)
        i421 = 2 ** (natoms - step)
        q = q.reshape(i124, i842, order="F")
        q[:, np.arange(1, i421 + 1) * 2 - 2] -= q[:, np.arange(1, i421 + 1) * 2 - 1]
    return q.reshape(1, lq, order="F").ravel()


def qtopl(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to plausibility function *pl* (Smets 2002)."""
    q = np.copy(q)
    q[0] = 0
    return np.abs(btom(q))


def qtov(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to disjunctive weight function *v*."""
    return btov(qtob(q))


def qtow(q: np.ndarray) -> np.ndarray:
    """Convert commonality function *q* to conjunctive weight function *w* (Denoeux 2008)."""
    w = np.exp(-qtom(np.log(q)))
    w[-1] = 1
    return w


# ---------------------------------------------------------------------------
# vto* — from disjunctive weight function
# ---------------------------------------------------------------------------

def vtob(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to implicability *b* (Denoeux 2008)."""
    return np.prod(v) / np.exp(mtob(np.log(v)))


def vtobel(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to belief function *bel*."""
    return btobel(vtob(v))


def vtom(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to mass function *m*."""
    return btom(vtob(v))


def vtopl(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to plausibility function *pl*."""
    return btopl(vtob(v))


def vtoq(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to commonality function *q*."""
    return btoq(vtob(v))


def vtow(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to conjunctive weight function *w*."""
    return btow(vtob(v))


# ---------------------------------------------------------------------------
# wto* — from conjunctive weight function
# ---------------------------------------------------------------------------

def wtob(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to implicability *b*."""
    return qtob(wtoq(w))


def wtobel(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to belief function *bel*."""
    return qtobel(wtoq(w))


def wtom(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to mass function *m*."""
    return qtom(wtoq(w))


def wtopl(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to plausibility function *pl*."""
    return qtopl(wtoq(w))


def wtoq(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to commonality function *q* (Denoeux 2008)."""
    return np.prod(w) / np.exp(mtoq(np.log(w)))


def wtov(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to disjunctive weight function *v*."""
    return qtov(wtoq(w))


# ---------------------------------------------------------------------------
# Conditioning and deconditioning matrices (Smets 2002, Section 9)
# ---------------------------------------------------------------------------

def conditioning_matrix(frame: list, a: frozenset) -> np.ndarray:
    """
    Build the Dempster conditioning matrix C_A for event A.

    C_A is a 2^n × 2^n specialization matrix (Smets 2002, Section 9)
    such that m[A] = C_A @ m, where m[A] is the BBA conditioned on A.

    The matrix entries are:

        C_A(B, C) = 1  if B = C ∩ A
        C_A(B, C) = 0  otherwise

    In terms of binary indices: C_A[i, j] = 1 iff i == (j & a_mask),
    where a_mask is the bitmask of A in the binary index ordering.

    Parameters
    ----------
    frame : list[str]
        Ordered list of atoms defining the binary index ordering.
    a : frozenset
        The conditioning event A ⊆ Ω.

    Returns
    -------
    np.ndarray
        Square matrix of shape (2^n, 2^n).

    References
    ----------
    Smets, P. (2002). IJAR, 31(1-2), 1-30. Section 9.
    """
    n = len(frame)
    size = 2 ** n
    a_mask = sum(1 << k for k, atom in enumerate(frame) if atom in a)
    C = np.zeros((size, size), dtype=float)
    for j in range(size):
        i = j & a_mask   # B = C ∩ A
        C[i, j] = 1.0
    return C


def deconditioning_matrix(frame: list, a: frozenset) -> np.ndarray:
    """
    Build the deconditioning matrix D_A for event A.

    D_A is a 2^n × 2^n generalization matrix (Smets 2002, Section 9)
    such that m* = D_A @ m, where m* is the least committed BBA whose
    conditioning on A yields m.

    The matrix entries are:

        D_A(B, C) = 1  if B = C ∪ Ā
        D_A(B, C) = 0  otherwise

    where Ā = Ω \\ A. In terms of binary indices: D_A[i, j] = 1 iff
    i == (j | abar_mask), where abar_mask is the bitmask of Ā.

    Equivalently: D_A = J · C_A · J (Smets 2002, Theorem 7.4).

    Parameters
    ----------
    frame : list[str]
        Ordered list of atoms defining the binary index ordering.
    a : frozenset
        The conditioning event A ⊆ Ω.

    Returns
    -------
    np.ndarray
        Square matrix of shape (2^n, 2^n).

    References
    ----------
    Smets, P. (2002). IJAR, 31(1-2), 1-30. Section 9.
    """
    n = len(frame)
    size = 2 ** n
    omega_mask = size - 1
    a_mask = sum(1 << k for k, atom in enumerate(frame) if atom in a)
    abar_mask = omega_mask & ~a_mask  # Ā = Ω \ A
    D = np.zeros((size, size), dtype=float)
    for j in range(size):
        i = j | abar_mask   # B = C ∪ Ā
        D[i, j] = 1.0
    return D


# ---------------------------------------------------------------------------
# Pignistic and plausibility probability transformations
# ---------------------------------------------------------------------------

def betp(m: np.ndarray) -> np.ndarray:
    """
    Pignistic probability transformation (BetP).

    Transforms a BBA m into a probability distribution over the n atoms
    of the frame, following the Transferable Belief Model decision rule
    (Smets & Kennes 1994).

    The pignistic probability of singleton {x} is:

        BetP({x}) = Σ_{A ∋ x} m(A) / (|A| · (1 - m(∅)))

    The result is a vector of length n (one value per atom), not 2^n.
    It sums to 1 when the BBA is not fully contradictory (m(∅) < 1).

    Parameters
    ----------
    m : np.ndarray
        Dense BBA vector of length 2^n, in binary index ordering.
        m[0] = m(∅).

    Returns
    -------
    np.ndarray
        Probability vector of length n, one entry per atom.

    Raises
    ------
    ValueError
        If m(∅) = 1 (fully contradictory BBA — BetP is undefined).

    References
    ----------
    Smets, P., Kennes, R. (1994). The transferable belief model.
    Artificial Intelligence, 66(2), 191-234.
    Smets, P. (2002). IJAR, 31(1-2), 1-30. Section 4.
    """
    conflict = m[0]
    if np.isclose(conflict, 1.0):
        raise ValueError(
            "betp: m(∅) = 1 (fully contradictory BBA). "
            "BetP is undefined in this case."
        )

    size = len(m)
    n = int(np.log2(size))
    result = np.zeros(n)
    norm = 1.0 - conflict

    for i in range(size):
        if abs(m[i]) < ZERO_MASS or i == 0:
            continue
        # Cardinality of subset i: number of 1-bits
        cardinality = bin(i).count("1")
        mass_share = m[i] / (cardinality * norm)
        # Distribute equally to all atoms in the subset
        for k in range(n):
            if i >> k & 1:
                result[k] += mass_share

    return result


def plp(m: np.ndarray) -> np.ndarray:
    """
    Plausibility probability transformation (PlP).

    Transforms a BBA m into a probability distribution over the n atoms
    of the frame by normalizing the plausibility of singletons.

    The plausibility probability of singleton {x} is:

        PlP({x}) = pl({x}) / Σ_{y ∈ Ω} pl({y})

    The result is a vector of length n (one value per atom), summing to 1.

    Parameters
    ----------
    m : np.ndarray
        Dense BBA vector of length 2^n, in binary index ordering.

    Returns
    -------
    np.ndarray
        Probability vector of length n, one entry per atom.

    Raises
    ------
    ValueError
        If all singleton plausibilities are zero (degenerate BBA).

    References
    ----------
    Cobb, B.R., Shenoy, P.P. (2006). On the plausibility transformation
    method for translating belief function models to probability models.
    International Journal of Approximate Reasoning, 41(3), 314-330.
    """
    size = len(m)
    n = int(np.log2(size))
    pl = mtopl(m)

    # pl of singletons: indices 1, 2, 4, 8, ... (powers of 2)
    singleton_pl = np.array([pl[1 << k] for k in range(n)])
    total = singleton_pl.sum()

    if np.isclose(total, 0.0):
        raise ValueError(
            "plp: all singleton plausibilities are zero. "
            "PlP is undefined for this BBA."
        )

    return singleton_pl / total
