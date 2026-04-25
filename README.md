# evtools

**Evidence Theory Tools** — a growing Python library of utilities for working
with belief functions in the Dempster-Shafer theory of evidence.

## Modules

| Module | Description |
|--------|-------------|
| `evtools.dsvector` | `DSVector` — unified container for any belief function representation |
| `evtools.conversions` | Low-level conversions between all standard representations via the Fast Möbius Transform |

---

## `evtools.dsvector`

`DSVector` is the central object of evtools. It represents any belief function
as a vector on `2^Ω`, in both **sparse** (dict) and **dense** (numpy array)
forms. The sparse representation is the master; the dense array is computed
on demand and cached.

### Kind enum

| `Kind` | Symbol | Name |
|--------|--------|------|
| `Kind.M`   | `m`   | Basic Belief Assignment (mass function) |
| `Kind.BEL` | `bel` | Belief function |
| `Kind.PL`  | `pl`  | Plausibility function |
| `Kind.B`   | `b`   | Commonality function |
| `Kind.Q`   | `q`   | Implicability function |
| `Kind.V`   | `v`   | Conjunctive weight function |
| `Kind.W`   | `w`   | Disjunctive weight function |

### Constructors

```python
from evtools.dsvector import DSVector, Kind

# Human-friendly: name focal elements as strings
m = DSVector.from_focal(["a", "b", "c"], {"a": 0.3, "b,c": 0.5})

# From a dense numpy array
m = DSVector.from_dense(["a", "b", "c"], np.array([0, 0.3, 0, 0, 0.5, 0, 0, 0.2]))

# From a sparse dict of frozensets
m = DSVector.from_sparse(["a", "b", "c"], {
    frozenset({"a"}):          0.3,
    frozenset({"b", "c"}):    0.5,
    frozenset({"a","b","c"}): 0.2,
})
```

### Conversions

```python
pl  = m.to(Kind.PL)   # returns a new DSVector with kind=Kind.PL
bel = m.to_bel()      # shortcut
b   = m.to_b()        # commonality
q   = m.to_q()        # implicability
v   = m.to_v()        # conjunctive weights (requires subnormal BBA)
w   = m.to_w()        # disjunctive weights (requires subnormal BBA)
```

### Accessing values

```python
m.sparse                     # dict[frozenset, float]
m.dense                      # np.ndarray of length 2^n
m[frozenset({"a"})]          # value for a given subset (0.0 if absent)
for subset, value in m: ...  # iterate over non-zero focal elements
```

---

## `evtools.conversions`

Low-level conversion functions between all standard representations, implemented
using the **Fast Möbius Transform** (FMT) from Smets (2002) and Denoeux (2008).
All functions operate on plain `np.ndarray` vectors of length `2^n`.

Every conversion is available as a `<source>to<target>` function,
e.g. `mtob`, `pltom`, `qtow`, `beltov`, etc.

```python
import numpy as np
from evtools.conversions import mtob, mtopl, mtobel, mtow

m = np.array([0.0, 0.3, 0.5, 0.2])  # ∅, {a}, {b}, {a,b}

print(mtob(m))    # commonality function
print(mtopl(m))   # plausibility function
print(mtobel(m))  # belief function
print(mtow(m))    # disjunctive weight function
```

---

## Installation

```bash
pip install evtools-dst
```

Or from source:

```bash
git clone https://github.com/daviddavkanmercier/evtools.git
cd evtools
pip install -e .
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## References

- P. Smets. *The application of the matrix calculus to belief functions*, International Journal of Approximate Reasoning, 31(1–2):1–30, 2002.
- T. Denœux. *Conjunctive and disjunctive combination of belief functions induced
by non-distinct bodies of evidence*, Artificial Intelligence, 172:234–264, 2008.
- D. Mercier, B. Quost, T. Denœux, *Refined modeling of sensor reliability in the belief function framework using contextual discounting*, Information Fusion, Vol. 9, Issue 2, pp 246-258, April 2008.

## License

MIT