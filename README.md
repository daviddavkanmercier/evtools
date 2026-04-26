# evtools

**Evidence Theory Tools** — a Python library for working with belief functions
in the Dempster-Shafer theory / Transferable Belief Model.

## Modules

| Module | Description |
|--------|-------------|
| `evtools.dsvector` | `DSVector` — unified container for any belief function representation |
| `evtools.conversions` | Low-level conversions via the Fast Möbius Transform |
| `evtools.combinations` | Combination rules: CRC, Dempster, DRC, Cautious, Bold |
| `evtools.corrections` | Correction mechanisms: discounting, reinforcement, negating |

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
| `Kind.V`   | `v`   | Disjunctive weight function |
| `Kind.W`   | `w`   | Conjunctive weight function |

### Constructors

```python
from evtools.dsvector import DSVector, Kind

# Human-friendly: name focal elements as strings
m = DSVector.from_focal(["a", "b", "c"], {"a": 0.3, "b,c": 0.5})

# From a dense numpy array (binary index ordering, Smets 2002)
m = DSVector.from_dense(["a", "b", "c"], np.array([0, 0.3, 0, 0, 0.5, 0, 0, 0.2]))

# From a sparse dict of frozensets
m = DSVector.from_sparse(["a", "b", "c"], {
    frozenset({"a"}):          0.3,
    frozenset({"b", "c"}):    0.5,
    frozenset({"a","b","c"}): 0.2,
})
```

### Special constructors

```python
# Simple MF A^β — focal sets Ω (mass β) and A (mass 1−β)
s = DSVector.simple(["a", "b", "c"], frozenset({"a"}), beta=0.6)

# Negative simple MF θ^β — focal sets ∅ (mass β) and θ (mass 1−β)
ns = DSVector.negative_simple(["a", "b", "c"], frozenset({"a"}), beta=0.4)
```

### Conversions

```python
pl  = m.to(Kind.PL)   # returns a new DSVector with kind=Kind.PL
bel = m.to_bel()      # shortcut
b   = m.to_b()        # commonality
q   = m.to_q()        # implicability
v   = m.to_v()        # disjunctive weights (requires subnormal BBA)
w   = m.to_w()        # conjunctive weights (requires subnormal BBA)
```

### Accessing values

```python
m.sparse                     # dict[frozenset, float]
m.dense                      # np.ndarray of length 2^n
m.is_valid                   # True if all masses ≥ 0 and sum = 1 (Kind.M only)
m[frozenset({"a"})]          # value for a given subset (0.0 if absent)
for subset, value in m: ...  # iterate over non-zero focal elements
```

---

## `evtools.combinations`

Combination rules for aggregating beliefs from multiple sources.

```python
from evtools.combinations import crc, dempster, drc, cautious, bold

m12 = crc(m1, m2)        # m1 & m2  — Conjunctive Rule (TBM), distinct reliable sources
m12 = dempster(m1, m2)   # m1 @ m2  — Dempster's normalized rule
m12 = drc(m1, m2)        # m1 | m2  — Disjunctive Rule, at least one reliable
m12 = cautious(m1, m2)   # Cautious rule, nondistinct reliable sources
m12 = bold(m1, m2)       # Bold disjunctive rule, nondistinct possibly unreliable

# Decombination (inverse operations — result may not be a valid BBA, check .is_valid)
m1  = decombine_crc(m12, m2)  # m12 6∩ m2 — removes m2 from a conjunctive combination
m1  = decombine_drc(m12, m2)  # m12 6∪ m2 — removes m2 from a disjunctive combination
```

Choice of rule:

|                     | All sources reliable | At least one reliable |
|---------------------|---------------------|-----------------------|
| **Distinct sources**    | `crc` / `dempster`  | `drc`                 |
| **Nondistinct sources** | `cautious`          | `bold`                |

Both `crc` and `drc` support `method="sparse"` (default) or `method="dense"`.

---

## `evtools.corrections`

Correction mechanisms for adjusting a BBA based on knowledge about the
quality of a source (reliability, truthfulness).

```python
from evtools.corrections import (
    discount,
    contextual_discount,
    theta_contextual_discount,
    contextual_reinforce,
    contextual_dediscount,
    contextual_dereinforce,
    contextual_negate,
)

# Classical discounting — source reliable with degree 1-α
m_disc = discount(m, alpha=0.4)

# Contextual discounting — reliability depends on each singleton context
betas = {frozenset({"a"}): 0.6, frozenset({"h"}): 1.0, frozenset({"r"}): 1.0}
m_cd = contextual_discount(m, betas)

# Θ-contextual discounting — reliability per coarsening partition
betas_theta = {frozenset({"a"}): 0.4, frozenset({"h","r"}): 0.9}
m_theta = theta_contextual_discount(m, betas_theta)

# Contextual reinforcement — dual of discounting (uses CRC instead of DRC)
m_cr = contextual_reinforce(m, betas)

# Inverse operations (result may not be a valid BBA — check .is_valid)
m_cdd = contextual_dediscount(m_cd, betas)   # reverses contextual_discount
m_cdr = contextual_dereinforce(m_cr, betas)  # reverses contextual_reinforce

# Contextual negating — source lies for some contexts
m_cn = contextual_negate(m, {frozenset({"a"}): 0.7})
```

Hierarchy of discounting:

```
discount(m, α)
  └── theta_contextual_discount(m, {Ω: 1-α})

contextual_discount(m, β)
  └── theta_contextual_discount(m, β)   [Θ = singletons]

theta_contextual_discount(m, β)         [general Θ partition]
```

---

## `evtools.conversions`

Low-level conversion functions operating on plain numpy arrays (length `2^n`),
using the Fast Möbius Transform. Every conversion is available as
`<source>to<target>`, e.g. `mtob`, `pltom`, `qtow`, `beltov`, etc.

```python
from evtools.conversions import mtob, mtopl, mtobel, mtoq

m = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
print(mtoq(m))    # commonality function
print(mtopl(m))   # plausibility function
```

Array indices follow the binary ordering of Smets (2002): index `i` corresponds
to the subset whose members are the frame atoms at the bit positions set in `i`.

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
- F. Pichon, D. Mercier, É. Lefèvre, F. Delmotte, *Proposition and learning of some belief function contextual correction mechanisms*, International Journal of Approximate Reasoning, Vol. 72, pp 4-42, May 2016.

## License

MIT
