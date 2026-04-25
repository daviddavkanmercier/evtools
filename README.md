# evtools

**Evidence Theory Tools** — a growing Python library of utilities for working
with belief functions in the Dempster-Shafer theory of evidence.

## Modules

| Module | Description |
|--------|-------------|
| `evtools.dsvector` | `DSVector` — unified container for any belief function representation |
| `evtools.conversions` | Low-level conversions between all standard representations via the Fast Möbius Transform |
| `evtools.mass` | Human-friendly construction of mass functions |

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

## `evtools.mass`

Utility functions for constructing mass functions from human-readable input.

```python
from evtools.mass import mass, frame_labels

# Build a mass vector; missing mass goes to Ω automatically
m = mass(["a", "b"], {"a": 0.3})
# → array([0. , 0.3, 0. , 0.7])

# Human-readable labels for all 2^n subsets
frame_labels(["a", "b"])
# → ['∅', 'a', 'b', 'a,b']
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

- Smets, P. (2002). *The application of the matrix calculus to belief functions.*
  International Journal of Approximate Reasoning.
- Denoeux, T. (2008). *Conjunctive and disjunctive combination of belief functions
  induced by non-distinct bodies of evidence.* Artificial Intelligence.
- Mercier, D., Quost, B., & Denoeux, T. (2008). *Refined modeling of sensor
  reliability in the belief function framework using contextual discounting.*
  Information Sciences.

## License

MIT