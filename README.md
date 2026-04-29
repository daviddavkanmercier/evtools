# evtools

**Evidence Theory Tools** — a Python library for working with belief functions
in the Dempster-Shafer theory / Transferable Belief Model. Version 0.18.0.

## Modules

| Module | Description |
|--------|-------------|
| `evtools.dsvector` | `DSVector` — unified container for any belief function representation |
| `evtools.conversions` | Low-level conversions via the Fast Möbius Transform |
| `evtools.combinations` | Combination rules: CRC, Dempster, DRC, Cautious, Bold, and decombinations |
| `evtools.corrections` | Correction mechanisms: discounting, reinforcement, negating |
| `evtools.decision` | Decision criteria: maximin, maximax, pignistic, plp, hurwicz, dominance |
| `evtools.metrics` | Performance metrics: discounted_accuracy, u65, u80 + mean aggregators |
| `evtools.display` | Display formats: ANSI terminal, plain text, HTML, LaTeX |
| `evtools.constants` | Numerical tolerance constants |

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
# Missing mass is automatically assigned to Ω
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

### Simple MF constructors

Simple MFs are the elementary building blocks of correction mechanisms.

```python
# Simple MF A^β — focal sets Ω (mass β) and A (mass 1−β)
# Used in Contextual Reinforcement (CR), CdR, CN
s = DSVector.simple(["a", "b", "c"], frozenset({"a"}), beta=0.6)

# Negative simple MF A_β — focal sets ∅ (mass β) and A (mass 1−β)
# Used in Contextual Discounting (CD), CdD
ns = DSVector.negative_simple(["a", "b", "c"], frozenset({"a"}), beta=0.4)
```

### Conversions

```python
pl  = m.to(Kind.PL)   # returns a new DSVector with kind=Kind.PL
bel = m.to_bel()      # shortcut
b   = m.to_b()        # commonality
q   = m.to_q()        # implicability
v   = m.to_v()        # disjunctive weights (requires subnormal BBA, m(∅) > 0)
w   = m.to_w()        # conjunctive weights (requires non-dogmatic BBA, m(Ω) > 0)
```

### Accessing values

```python
m.sparse                     # dict[frozenset, float]
m.dense                      # np.ndarray of length 2^n
m.is_valid                   # True if all masses ≥ 0 and sum = 1 (Kind.M only)
m[frozenset({"a"})]          # value for a given subset (0.0 if absent)
for subset, value in m: ...  # iterate over non-zero focal elements
```

### Display

```python
m.display("ansi")    # colored terminal (default __repr__)
m.display("plain")   # plain text, no colors
m.display("html")    # HTML table (Jupyter renders this automatically)
m.display("latex")   # LaTeX tabular for papers
```

---

## `evtools.combinations`

Combination rules for aggregating beliefs from multiple sources.

```python
from evtools.combinations import crc, dempster, drc, cautious, bold
from evtools.combinations import decombine_crc, decombine_drc

m12 = crc(m1, m2)        # m1 & m2  — Conjunctive Rule (TBM), distinct reliable sources
m12 = dempster(m1, m2)   # m1 @ m2  — Dempster's normalized rule
m12 = drc(m1, m2)        # m1 | m2  — Disjunctive Rule, at least one reliable
m12 = cautious(m1, m2)   # Cautious rule, nondistinct reliable sources (idempotent)
m12 = bold(m1, m2)       # Bold disjunctive rule, nondistinct possibly unreliable (idempotent)

# Decombination — inverse operations (result may not be valid, check .is_valid)
m1 = decombine_crc(m12, m2)  # m12 6∩ m2 — removes m2 from a conjunctive combination
m1 = decombine_drc(m12, m2)  # m12 6∪ m2 — removes m2 from a disjunctive combination

# Conditioning and deconditioning (Smets 2002, Section 9)
A  = frozenset({"a", "h"})
m_cond   = condition(m, A)             # m[A]: B → B ∩ A
m_decond = decondition(m_cond, A)      # m*:   B → B ∪ Ā

# Conditioning matrices (dense mode)
from evtools.conversions import conditioning_matrix, deconditioning_matrix
CA = conditioning_matrix(frame, A)     # 2^n × 2^n specialization matrix
DA = deconditioning_matrix(frame, A)   # 2^n × 2^n generalization matrix
```

Choice of rule:

|                         | All sources reliable   | At least one reliable |
|-------------------------|------------------------|-----------------------|
| **Distinct sources**    | `crc` / `dempster`     | `drc`                 |
| **Nondistinct sources** | `cautious`             | `bold`                |

Both `crc` and `drc` support `method="sparse"` (default) or `method="dense"`.

---

## `evtools.corrections`

Correction mechanisms for adjusting a BBA based on knowledge about the
quality of a source (reliability, truthfulness).

Notation:
- **A^β** — simple MF: focal sets Ω (mass β) and A (mass 1−β)
- **A_β** — negative simple MF: focal sets ∅ (mass β) and A (mass 1−β)

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

# Classical discounting — source reliable with degree β ∈ [0,1]
# β=1: unchanged; β=0: vacuous BBA
m_disc = discount(m, beta=0.6)

# Contextual discounting (CD) — reliability per singleton context
# Uses negative simple MFs A_β and the DRC
betas = {frozenset({"a"}): 0.6, frozenset({"h"}): 1.0, frozenset({"r"}): 1.0}
m_cd = contextual_discount(m, betas)

# Θ-contextual discounting — reliability per coarsening partition
betas_theta = {frozenset({"a"}): 0.4, frozenset({"h","r"}): 0.9}
m_theta = theta_contextual_discount(m, betas_theta)

# Contextual Reinforcement (CR) — dual of CD, uses simple MFs A^β and the CRC
m_cr = contextual_reinforce(m, betas)

# Inverse operations (result may not be valid — check .is_valid)
m_cdd = contextual_dediscount(m_cd, betas)    # reverses CD
m_cdr = contextual_dereinforce(m_cr, betas)   # reverses CR

# Contextual Negating (CN) — source non-truthful with probability 1−β
m_cn = contextual_negate(m, {frozenset({"a"}): 0.7})
```

Hierarchy of discounting:

```
discount(m, β)
  └── theta_contextual_discount(m, {Ω: β})

contextual_discount(m, β)
  └── theta_contextual_discount(m, β)   [Θ = singletons]

theta_contextual_discount(m, β)         [general Θ partition]
```

---

## `evtools.decision`

Decision criteria for selecting an act from a BBA. Two families:

- **Complete preference relations** return a single optimal act `(index, atom)`.
- **Partial preference relations** return a `frozenset[str]` of non-dominated atoms.

```python
from evtools.decision import (
    maximin, maximax, pignistic_decision, plp_decision, probability_decision,
    hurwicz, strong_dominance, weak_dominance,
)

# Complete preference relations — return (index, atom)
maximin(m)             # pessimistic: max lower expected utility
maximax(m)             # optimistic:  max upper expected utility
pignistic_decision(m)  # MEU with BetP (Smets pignistic)
plp_decision(m)        # MEU with PlP  (Cobb & Shenoy plausibility-prob.)
hurwicz(m, alpha=0.5)  # convex combination of maximin and maximax

# Generic MEU — pass any m → probability transform
from evtools.conversions import betp, plp
probability_decision(m, transform=betp)        # ≡ pignistic_decision
probability_decision(m, transform=plp)         # ≡ plp_decision
probability_decision(m, transform=my_custom)   # bring your own

# With a custom utility matrix U of shape (n, n)
import numpy as np
U = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 1]])  # u(a_i, ω_j)
maximin(m, U)

# Partial preference relations — return frozenset of non-dominated atoms
strong_dominance(m)    # ω ≻ ω'  ⟺  Bel({ω}) ≥ Pl({ω'})
weak_dominance(m)      # ω ≻ ω'  ⟺  Bel({ω}) ≥ Bel({ω'}) and Pl({ω}) ≥ Pl({ω'})

```

Default utility (when `U` is omitted) is the identity matrix (0-1 utility, the
standard classification setting). With identity utility, `pignistic_decision`
returns the atom with maximum BetP.

---

## `evtools.metrics`

Performance metrics for evaluating decisions and predictions.

### Per-instance metrics on partial decisions

Score a partial decision `d ⊆ Ω` against a true class `ω` using the discounted
accuracy `x = I(ω ∈ d) / |d|` (Zaffalon et al. 2012).

```python
from evtools.metrics import discounted_accuracy, u65, u80, utility_score

discounted_accuracy(d, omega)             # x
u65(d, omega)                             # 1.6·x − 0.6·x²  (≡ 0.65 if |d|=2 correct)
u80(d, omega)                             # 2.2·x − 1.2·x²  (≡ 0.80 if |d|=2 correct)
utility_score(d, omega, a=1.6, b=0.6)     # generic a·x − b·x²
```

### Mean aggregators over a dataset

```python
from evtools.metrics import mean_u65, mean_u80, mean_discounted_accuracy

predictions = [strong_dominance(m_i) for m_i in classifier_outputs]
print(mean_u65(predictions, true_labels))
print(mean_u80(predictions, true_labels))
```

### Hard-classification metrics: use scikit-learn

For ROC, AUC, accuracy, precision/recall, etc. on hard predictions, extract a
probability vector (e.g. via `m.to_betp()` or `m.to_plp()`) and feed it to
`sklearn.metrics`. The tutorials show end-to-end examples.

---

## `evtools.display`

Four output formats, all adapting the column header to the kind (`m`, `bel`, `pl`, ...).
In Jupyter notebooks, `DSVector._repr_html_()` is called automatically.

```python
from evtools.display import repr_plain, repr_html, repr_latex, display_all

print(repr_plain(m))   # plain text, no colors
print(repr_latex(m))   # LaTeX tabular for papers
m.display("ansi")      # colored terminal (default)
m.display("html")      # HTML table

# Show all representations in one table
# v added if m is subnormal (m(∅) > 0)
# w added if m is non-dogmatic (m(Ω) > 0)
print(display_all(m, "plain"))
m.display_all()        # same via method
```

---

## `evtools.conversions`

Low-level conversion functions operating on plain numpy arrays (length `2^n`),
using the Fast Möbius Transform (Smets 2002). Every conversion is available as
`<source>to<target>`, e.g. `mtob`, `pltom`, `qtow`, `beltov`, etc.

Also includes conditioning matrices and probability transformations:

```python
from evtools.conversions import mtob, mtopl, mtobel, mtoq
from evtools.conversions import betp, plp
from evtools.conversions import conditioning_matrix, deconditioning_matrix

m = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
print(mtoq(m))    # commonality function
print(mtopl(m))   # plausibility function

# Probability transformations (return np.ndarray of length n, not 2^n)
print(betp(m))    # pignistic probability BetP (Smets & Kennes 1994)
print(plp(m))     # plausibility probability PlP (Cobb & Shenoy 2006)

# Equivalently via DSVector methods
m_vec = DSVector.from_dense(frame, m)
print(m_vec.to_betp())  # np.ndarray of length n
print(m_vec.to_plp())

# Conditioning matrices
CA = conditioning_matrix(frame, frozenset({"a", "h"}))  # 2^n × 2^n
DA = deconditioning_matrix(frame, frozenset({"a", "h"}))
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
- T. Denœux. *Conjunctive and disjunctive combination of belief functions induced by non-distinct bodies of evidence*, Artificial Intelligence, 172:234–264, 2008.
- D. Mercier, B. Quost, T. Denœux. *Refined modeling of sensor reliability in the belief function framework using contextual discounting*, Information Fusion, Vol. 9, Issue 2, pp 246-258, April 2008.
- F. Pichon, D. Mercier, É. Lefèvre, F. Delmotte. *Proposition and learning of some belief function contextual correction mechanisms*, International Journal of Approximate Reasoning, Vol. 72, pp 4-42, May 2016.
- T. M. Strat. *Decision analysis using belief functions*, International Journal of Approximate Reasoning, Vol. 4, Issues 5-6, pp 391-417, 1990.
- M. C. M. Troffaes. *Decision making under uncertainty using imprecise probabilities*, International Journal of Approximate Reasoning, Vol. 45, Issue 1, pp 17-29, 2007.
- L. Ma, T. Denœux. *Partial classification in the belief function framework*, Knowledge-Based Systems, Vol. 214, 106742, 2021.
- M. Zaffalon, G. Corani, D. Mauá. *Evaluating credal classifiers by utility-discounted predictive accuracy*, International Journal of Approximate Reasoning, Vol. 53, Issue 8, pp 1282-1301, 2012.
- S. Mutmainah. *Imperfect labels and belief functions for supervised classification*, PhD thesis, Université d'Artois, 2021.

## License

MIT
