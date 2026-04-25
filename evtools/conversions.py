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


# ---------------------------------------------------------------------------
# bto* — from commonality function
# ---------------------------------------------------------------------------

def btobel(b: np.ndarray) -> np.ndarray:
    """Convert commonality *b* to belief function *bel*.

    bel(A) = b(A) - m(∅)
    """
    return b - b[0]


def btom(b: np.ndarray) -> np.ndarray:
    """Convert commonality *b* to mass function *m* using the FMT (Smets)."""
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
    """Convert commonality *b* to plausibility function *pl*.

    pl(A) = 1 - b(Ā)
    """
    b = np.copy(b)
    lb = len(b)
    b = b.reshape(1, lb, order="F")
    b = b[0][-1] - np.fliplr(b).ravel()
    b[0] = 0
    return b


def btoq(b: np.ndarray) -> np.ndarray:
    """Convert commonality *b* to implicability function *q*."""
    return pltoq(btopl(b))


def btov(b: np.ndarray) -> np.ndarray:
    """Convert commonality *b* to disjunctive weight function *v* (Denoeux 2008)."""
    v = np.exp(-btom(np.log(b)))
    v[0] = 1
    return v


def btow(b: np.ndarray) -> np.ndarray:
    """Convert commonality *b* to conjunctive weight function *w*."""
    return qtow(btoq(b))


# ---------------------------------------------------------------------------
# belto* — from belief function
# ---------------------------------------------------------------------------

def beltob(bel: np.ndarray) -> np.ndarray:
    """Convert belief function *bel* to commonality *b*.

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
    """Convert belief function *bel* to implicability function *q*."""
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
    """Convert mass function *m* to commonality *b* using the FMT (Smets)."""
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
    """Convert mass function *m* to implicability function *q* using the FMT (Smets)."""
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
    """Convert plausibility function *pl* to commonality *b*.

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
    """Convert plausibility function *pl* to implicability function *q* (Smets 2002)."""
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
# qto* — from implicability function
# ---------------------------------------------------------------------------

def qtob(q: np.ndarray) -> np.ndarray:
    """Convert implicability function *q* to commonality *b*."""
    return pltob(qtopl(q))


def qtobel(q: np.ndarray) -> np.ndarray:
    """Convert implicability function *q* to belief function *bel*."""
    return btobel(qtob(q))


def qtom(q: np.ndarray) -> np.ndarray:
    """Convert implicability function *q* to mass function *m* using the FMT (Smets)."""
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
    """Convert implicability function *q* to plausibility function *pl* (Smets 2002)."""
    q = np.copy(q)
    q[0] = 0
    return np.abs(btom(q))


def qtov(q: np.ndarray) -> np.ndarray:
    """Convert implicability function *q* to disjunctive weight function *v*."""
    return btov(qtob(q))


def qtow(q: np.ndarray) -> np.ndarray:
    """Convert implicability function *q* to conjunctive weight function *w* (Denoeux 2008)."""
    w = np.exp(-qtom(np.log(q)))
    w[-1] = 1
    return w


# ---------------------------------------------------------------------------
# vto* — from disjunctive weight function
# ---------------------------------------------------------------------------

def vtob(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to commonality *b* (Denoeux 2008)."""
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
    """Convert disjunctive weight function *v* to implicability function *q*."""
    return btoq(vtob(v))


def vtow(v: np.ndarray) -> np.ndarray:
    """Convert disjunctive weight function *v* to conjunctive weight function *w*."""
    return btow(vtob(v))


# ---------------------------------------------------------------------------
# wto* — from conjunctive weight function
# ---------------------------------------------------------------------------

def wtob(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to commonality *b*."""
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
    """Convert conjunctive weight function *w* to implicability function *q* (Denoeux 2008)."""
    return np.prod(w) / np.exp(mtoq(np.log(w)))


def wtov(w: np.ndarray) -> np.ndarray:
    """Convert conjunctive weight function *w* to disjunctive weight function *v*."""
    return qtov(wtoq(w))
