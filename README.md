# evtools

**Evidence Theory Tools** — a growing Python library of utilities for working
with belief functions in the Dempster-Shafer theory of evidence.

## Modules

| Module | Description |
|--------|-------------|
| `evtools.conversions` | Conversions between all standard belief function representations |

---

## `evtools.conversions`

Converts between all standard representations of belief functions using the
**Fast Möbius Transform** (FMT) from Smets (2002) and Denoeux (2008).

### Supported representations

| Symbol | Name |
|--------|------|
| `m`   | Basic Belief Assignment (mass function) |
| `bel` | Belief function |
| `pl`  | Plausibility function |
| `b`   | Commonality function |
| `q`   | Implicability function |
| `v`   | Conjunctive weight function |
| `w`   | Disjunctive weight function |

Every conversion is available as a `<source>to<target>` function.
For example `mtob`, `pltom`, `qtow`, `beltov`, etc.

---

## Installation

```bash
pip install evtools-dst
```

Or from source:

```bash
git clone https://github.com/<your-username>/evtools.git
cd evtools
pip install -e .
```

## Quick start

```python
import numpy as np
from evtools.conversions import mtob, mtopl, mtobel, mtow

# Mass function over a 2-atom frame {a, b}
# Index order: ∅, {a}, {b}, {a,b}
m = np.array([0.0, 0.3, 0.5, 0.2])

print(mtob(m))    # commonality function
print(mtopl(m))   # plausibility function
print(mtobel(m))  # belief function
print(mtow(m))    # disjunctive weight function
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## References

- Smets, P. (2002). *The application of the transferable belief model to diagnostic problems.* International Journal of Intelligent Systems.
- Denoeux, T. (2008). *Conjunctive and disjunctive combination of belief functions induced by non-distinct bodies of evidence.* Artificial Intelligence.

## License

MIT
