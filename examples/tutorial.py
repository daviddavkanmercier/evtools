"""
evtools — Tutorial
==================

This tutorial walks through all main features of evtools, using a running
example from belief function theory: a sensor classifies aerial targets as
airplane (a), helicopter (h), or rocket (r).

Sections
--------
1.  Building a BBA          — from_focal, from_dense, from_sparse
2.  Accessing values        — sparse, dense, iteration, is_valid
3.  Subnormal BBA           — m(∅) > 0
4.  Conversions             — bel, pl, b, q, v, w
5.  v and w weights         — disjunctive / conjunctive weight functions
6.  Round-trip consistency  — m → kind → m sanity check
7.  Low-level API           — numpy arrays via conversions module
8.  Combination rules       — CRC, Dempster, DRC (distinct sources)
9.  DRC                     — disjunctive rule (s1 | s2)
10. Cautious & Bold         — nondistinct sources
11. Correction mechanisms   — discount, contextual_*, CdD, CN
12. Simple MFs              — DSVector.simple, .negative_simple, decombination
13. Display formats         — to_string, to_ansi, to_html, to_latex
14. all_kinds=True          — all representations in one table
15. Conditioning            — condition(m, A), decondition(m, A), C_A, D_A
16. BetP and PlP            — pignistic and plausibility probability transforms
17. Decision criteria       — maximin, maximax, pignistic, plp, hurwicz, dominance
18. Performance metrics     — u65, u80, pl_loss; sklearn for ROC/AUC
19. Learning corrections    — fit_cd, fit_cr, fit_cn (Pichon 2016)
20. Per-group learning      — fit_per_group, apply_per_group (Mutmainah 2021)
21. EkNN + K-fold CV        — Denoeux 1995, Zouhal 1998 + KFold/StratifiedKFold
22. EkNN on UCI benchmarks  — Sonar and Ionosphere (Zouhal 1998 Tables/Figures)
23. EkNN + CD/CR/CN         — train classifier on L1, fit best correction on L2
24. EkNN + per-group corr.  — fit_per_group with strong/weak dominance partition

References
----------
- P. Smets. The application of the matrix calculus to belief functions.
  IJAR, 31(1-2):1-30, 2002.
- T. Denœux. Conjunctive and disjunctive combination of belief functions
  induced by nondistinct bodies of evidence. AI, 172:234-264, 2008.
- D. Mercier, B. Quost, T. Denœux. Refined modeling of sensor reliability
  in the belief function framework using contextual discounting.
  Information Fusion, 9(2):246-258, 2008.
- F. Pichon, D. Mercier, É. Lefèvre, F. Delmotte. Proposition and learning
  of some belief function contextual correction mechanisms.
  IJAR, 72:4-42, 2016.
- T. M. Strat. Decision analysis using belief functions.
  IJAR, 4(5-6):391-417, 1990.
- M. C. M. Troffaes. Decision making under uncertainty using imprecise
  probabilities. IJAR, 45(1):17-29, 2007.
- L. Ma, T. Denœux. Partial classification in the belief function framework.
  Knowledge-Based Systems, 214, 106742, 2021.
- M. Zaffalon, G. Corani, D. Mauá. Evaluating credal classifiers by
  utility-discounted predictive accuracy. IJAR, 53(8), 1282-1301, 2012.
- S. Mutmainah, S. Hachour, F. Pichon, D. Mercier. On learning evidential
  contextual corrections from soft labels using a measure of discrepancy
  between contour functions. SUM 2019, LNCS 11940, pp 405-411.
- S. Mutmainah, S. Hachour, F. Pichon, D. Mercier. Improving an evidential
  source of information using contextual corrections depending on partial
  decisions. BELIEF 2021, pp 247-256.
- S. Mutmainah. Learning to adjust an evidential source of information using
  partially labeled data and partial decisions. PhD thesis, Univ. d'Artois, 2021.
"""

import numpy as np
from evtools.dsvector import DSVector, Kind

B   = "\033[1m"
R   = "\033[0m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED   = "\033[31m"

frame = ["a", "h", "r"]  # airplane, helicopter, rocket


def section(title):
    print(f"\n{B}{'─' * 60}{R}")
    print(f"{B}  {title}{R}")
    print(f"{B}{'─' * 60}{R}\n")


# ---------------------------------------------------------------------------
# 1. Building a BBA
# ---------------------------------------------------------------------------
# A Basic Belief Assignment (BBA) is a function m: 2^Ω → [0,1] with
# Σ m(A) = 1. Three constructors are available.

section("1. Building a BBA")

# from_focal: human-friendly string keys.
# Missing mass is automatically assigned to Ω = {a, h, r}.
print(f"{DIM}# from_focal — human-friendly string keys{R}")
m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(m)

# from_dense: numpy array in binary index order (Smets 2002).
# For frame=[a,h,r]: index 0=∅, 1={a}, 2={h}, 3={a,h}, 4={r}, ...
print(f"\n{DIM}# from_dense — numpy array (binary index order){R}")
array = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
m2 = DSVector.from_dense(frame, array)
print(m2)

# from_sparse: dict of frozensets.
print(f"\n{DIM}# from_sparse — dict of frozensets{R}")
m3 = DSVector.from_sparse(frame, {
    frozenset({"a"}): 0.5,
    frozenset({"r"}): 0.5,
})
print(m3)

# ---------------------------------------------------------------------------
# 2. Accessing values
# ---------------------------------------------------------------------------

section("2. Accessing values")

print(f"m[{{a}}]    = {m[frozenset({'a'})]}")
print(f"m[{{h}}]    = {m[frozenset({'h'})]}   {DIM}← not a focal element{R}")
print(f"n_focal   = {m.n_focal}")
print(f"dense     = {m.dense}")
print(f"is_valid  = {m.is_valid}")

print(f"\n{DIM}# Iterating over focal elements:{R}")
for subset, value in m:
    label = "{" + ", ".join(sorted(subset)) + "}" if subset else "∅"
    print(f"  m({label}) = {value}")

# ---------------------------------------------------------------------------
# 3. Subnormal BBA  —  m(∅) > 0
# ---------------------------------------------------------------------------
# In the TBM, m(∅) > 0 is allowed and represents internal conflict.

section("3. Subnormal BBA  —  m(∅) > 0")

m_sub = DSVector.from_focal(
    frame, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False
)
print(m_sub)

# ---------------------------------------------------------------------------
# 4. Conversions
# ---------------------------------------------------------------------------
# All standard representations are available via .to(Kind) or shortcuts.

section("4. Conversions")

print(m.to_bel())
print()
print(m.to_pl())
print()
print(m.to_b())
print()
print(m.to_q())
print()
print(f"{DIM}# v and w require a subnormal BBA (b > 0 everywhere){R}\n")
print(m_sub.to_v())
print()
print(m_sub.to_w())

# ---------------------------------------------------------------------------
# 5. Conjunctive and disjunctive weights (v, w)
# ---------------------------------------------------------------------------
# v (disjunctive weights) and w (conjunctive weights) require a subnormal BBA
# (m(∅) > 0), so that b(A) > 0 for all A ⊆ Ω.
# They are defined via a Möbius transform on log(b) and log(q) respectively
# (Denoeux 2008, Section 2.2).

section("5. Conjunctive and disjunctive weights (v, w)")

print(f"{DIM}# Subnormal BBA required — m(∅) > 0{R}")
m_sub2 = DSVector.from_focal(frame, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False)
print(m_sub2)

print()
print(f"{DIM}# Disjunctive weight function v{R}")
print(m_sub2.to_v())

print()
print(f"{DIM}# Conjunctive weight function w{R}")
print(m_sub2.to_w())

print()
print(f"{DIM}# Round-trip v and w{R}")
for kind in [Kind.V, Kind.W]:
    back = m_sub2.to(kind).to(Kind.M)
    ok = np.allclose(back.dense, m_sub2.dense, atol=1e-10)
    status = f"{GREEN}✓ OK{R}" if ok else f"{RED}✗ MISMATCH{R}"
    print(f"  m_sub → {kind.value} → m   {status}")

# ---------------------------------------------------------------------------
# 6. Round-trip consistency
# ---------------------------------------------------------------------------

section("6. Round-trip consistency")

for kind in [Kind.BEL, Kind.PL, Kind.B, Kind.Q]:
    back = m.to(kind).to(Kind.M)
    ok = np.allclose(back.dense, m.dense, atol=1e-10)
    status = f"{GREEN}✓ OK{R}" if ok else f"{RED}✗ MISMATCH{R}"
    print(f"  m → {kind.value:>3} → m   {status}")

for kind in [Kind.V, Kind.W]:
    back = m_sub.to(kind).to(Kind.M)
    ok = np.allclose(back.dense, m_sub.dense, atol=1e-10)
    status = f"{GREEN}✓ OK{R}" if ok else f"{RED}✗ MISMATCH{R}"
    print(f"  m_sub → {kind.value} → m   {status}")

# ---------------------------------------------------------------------------
# 6. Low-level conversions API  (numpy arrays)
# ---------------------------------------------------------------------------
# All conversions are also available as standalone functions on numpy arrays,
# using the Fast Möbius Transform (Smets 2002, Section 3).

section("7. Low-level conversions API  (numpy arrays)")

from evtools.conversions import mtobel, mtopl, mtob

m_array = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
print(f"bel : {mtobel(m_array)}")
print(f"pl  : {mtopl(m_array)}")
print(f"b   : {mtob(m_array)}")

# ---------------------------------------------------------------------------
# 7. Combination rules
# ---------------------------------------------------------------------------
# Five combination rules are available. Choice depends on source properties:
#
#   | | All sources reliable | At least one reliable |
#   |---|---|---|
#   | Distinct sources    | crc / dempster | drc    |
#   | Nondistinct sources | cautious       | bold   |

section("8. Combination rules")

from evtools.combinations import crc, dempster, drc, cautious, bold

s1 = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
s2 = DSVector.from_focal(frame, {"h": 0.3, "r": 0.4, "a,h,r": 0.3})

print(f"{DIM}# Sensor 1{R}")
print(s1)
print(f"\n{DIM}# Sensor 2{R}")
print(s2)

print(f"\n{DIM}# CRC — distinct, both reliable  (s1 & s2){R}")
m12_crc = s1 & s2
print(m12_crc)
print(f"  Conflict m(∅) = {m12_crc[frozenset()]:.4f}")

print(f"\n{DIM}# Dempster's rule — normalized CRC  (s1 @ s2){R}")
print(s1 @ s2)

ok = np.allclose((s1 & s2).dense, (s2 & s1).dense, atol=1e-10)
print(f"\n  Commutativity s1 & s2 == s2 & s1 :  {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

# ---------------------------------------------------------------------------
# 8. Disjunctive Rule of Combination (DRC)
# ---------------------------------------------------------------------------

section("9. DRC — at least one source reliable  (s1 | s2)")

print(s1 | s2)

# ---------------------------------------------------------------------------
# 9. Cautious and Bold rules (nondistinct sources)
# ---------------------------------------------------------------------------

section("10. Cautious and Bold rules (nondistinct sources)")

# Cautious: nondogmatic BBAs (m(Ω) > 0), commutative, associative, idempotent
c1 = DSVector.from_focal(frame, {"a": 0.3, "h": 0.2, "a,h,r": 0.5})
c2 = DSVector.from_focal(frame, {"a": 0.4, "r": 0.1, "a,h,r": 0.5})

print(f"{DIM}# Cautious combination — sources reliable but possibly overlapping{R}\n")
print(cautious(c1, c2))
ok = np.allclose(cautious(c1, c1).dense, c1.dense, atol=1e-10)
print(f"\n  Idempotence c1 ∧ c1 == c1 :  {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

# Bold: subnormal BBAs (m(∅) > 0), commutative, associative, idempotent
b1 = DSVector.from_focal(frame, {"": 0.1, "a": 0.4, "a,h,r": 0.5}, complete=False)
b2 = DSVector.from_focal(frame, {"": 0.2, "r": 0.3, "a,h,r": 0.5}, complete=False)

print(f"\n{DIM}# Bold disjunctive rule — sources possibly overlapping, ≥1 reliable{R}\n")
print(bold(b1, b2))

# ---------------------------------------------------------------------------
# 10. Correction mechanisms
# ---------------------------------------------------------------------------
# Corrections adjust a BBA based on knowledge about the source quality.
#
#   discount(m, β)                  — classical (single reliability β)
#   contextual_discount(m, betas)   — Ω-contextual (per singleton)
#   theta_contextual_discount(m, β) — Θ-contextual (per partition element)
#   contextual_reinforce(m, betas)  — dual of CD (uses CRC)
#   contextual_dediscount(m, betas) — inverse of CD
#   contextual_dereinforce(m,betas) — inverse of CR
#   contextual_negate(m, betas)     — source lies contextually

section("11. Correction mechanisms")

from evtools.corrections import (
    discount, contextual_discount, theta_contextual_discount,
    contextual_reinforce, contextual_dediscount, contextual_dereinforce,
    contextual_negate,
)

s = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(f"{DIM}# Original BBA{R}")
print(s)

# Classical discounting: β=0.6, source 60% reliable
print(f"\n{DIM}# Classical discounting β=0.6 — source 60% reliable{R}")
print(discount(s, beta=0.6))

# Contextual discounting: unreliable only when airplane (β_a=0.6)
# Example 1, Case 1 of Mercier et al. (2008)
print(f"\n{DIM}# Contextual discounting β_a=0.6, β_h=β_r=1.0{R}")
betas_cd = {frozenset({"a"}): 0.6, frozenset({"h"}): 1.0, frozenset({"r"}): 1.0}
mcd = contextual_discount(s, betas_cd)
print(mcd)

# Θ-contextual discounting: coarser partition Θ = {{a}, {h,r}}
print(f"\n{DIM}# Θ-contextual discounting — Θ={{{{a}}}}, {{{{h,r}}}}{R}")
print(theta_contextual_discount(s, {frozenset({"a"}): 0.4, frozenset({"h","r"}): 0.9}))

# Contextual reinforcement (dual of CD)
print(f"\n{DIM}# Contextual reinforcement — dual of CD{R}")
print(contextual_reinforce(s, betas_cd))

# CdD: inverse of CD
print(f"\n{DIM}# Contextual de-discounting — reverses CD{R}")
mdd = contextual_dediscount(mcd, betas_cd)
print(mdd)
ok = np.allclose(mdd.dense, s.dense, atol=1e-6)
print(f"  Recovers original: {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

# Contextual negating
print(f"\n{DIM}# Contextual negating β_a=0.7 — source 70% truthful for airplane{R}")
print(contextual_negate(s, {frozenset({"a"}): 0.7}))

print(f"\n{DIM}# is_valid — useful after inverse operations{R}")
print(f"  s.is_valid         = {s.is_valid}")
print(f"  discount(s,.6).is_valid = {discount(s, 0.6).is_valid}")

# ---------------------------------------------------------------------------
# 11. Simple MFs and decombination
# ---------------------------------------------------------------------------
# Simple MFs are the building blocks of correction mechanisms.
#   A^β  (DSVector.simple)          — focal sets Ω (mass β) and A (mass 1-β)
#   A_β  (DSVector.negative_simple) — focal sets ∅ (mass β) and A (mass 1-β)

section("12. Simple MFs and decombination")

print(f"{DIM}# Simple MF A^β — focal sets Ω and A (used in CR, CdR, CN){R}")
s_simple = DSVector.simple(frame, frozenset({"a"}), beta=0.6)
print(s_simple)

print(f"\n{DIM}# Negative simple MF A_β — focal sets ∅ and A (used in CD, CdD){R}")
ns = DSVector.negative_simple(frame, frozenset({"a"}), beta=0.4)
print(ns)

print(f"\n{DIM}# decombine_crc: m1 6∩ m2 — removes m2 from a conjunctive combination{R}")
from evtools.combinations import decombine_crc, decombine_drc

m1 = DSVector.from_focal(frame, {"a": 0.4, "a,h,r": 0.6})
m2 = DSVector.from_focal(frame, {"h": 0.3, "a,h,r": 0.7})
m12 = crc(m1, m2)
m1_rec = decombine_crc(m12, m2)
ok = m1_rec.is_valid and np.allclose(m1_rec.dense, m1.dense, atol=1e-6)
print(m1_rec)
print(f"  Recovers m1: {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

print(f"\n{DIM}# decombine_drc: m1 6∪ m2 — removes m2 from a disjunctive combination{R}")
m1_sub = DSVector.from_focal(frame, {"": 0.1, "a": 0.4, "a,h,r": 0.5}, complete=False)
m2_sub = DSVector.from_focal(frame, {"": 0.2, "h": 0.3, "a,h,r": 0.5}, complete=False)
m12_d = drc(m1_sub, m2_sub)
m1_rec_d = decombine_drc(m12_d, m2_sub)
ok_d = m1_rec_d.is_valid and np.allclose(m1_rec_d.dense, m1_sub.dense, atol=1e-6)
print(m1_rec_d)
print(f"  Recovers m1_sub: {GREEN}✓ OK{R}" if ok_d else f"  {RED}✗ MISMATCH{R}")

# ---------------------------------------------------------------------------
# 12. Display formats
# ---------------------------------------------------------------------------
# Four output formats are available via the display module.
# The column header adapts to the kind (m, bel, pl, b, q, v, w).

section("13. Display formats")

from evtools.display import to_string, to_html, to_latex

m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})

# ANSI (default __repr__)
print(f"{DIM}# to_ansi — colored terminal (also used by __repr__){R}\n")
print(m)

# Plain text
print(f"\n{DIM}# to_string — no colors, for logs and files{R}\n")
print(m.to_string())

# Column header adapts to kind
print(f"\n{DIM}# Column header adapts to the kind{R}")
for kind_fn, label in [("to_bel","Belief"), ("to_pl","Plausibility"), ("to_q","Implicability")]:
    v = getattr(m, kind_fn)()
    print(f"\n{DIM}# {label} function — header shows '{v.kind.value}'{R}")
    print(v.to_string())

# LaTeX
print(f"\n{DIM}# to_latex — ready to paste in a LaTeX paper{R}\n")
print(m.to_latex())

# Module-level function form
print(f"\n{DIM}# Module-level functions — same result, useful when m is a generic argument{R}")
print(to_string(m).split('\n')[0], "  …")

# Jupyter auto-display via _repr_html_
print(f"\n{DIM}# In Jupyter: DSVector renders as HTML table automatically{R}")
print(f"{DIM}# via _repr_html_() — no extra call needed{R}")
print(f"  HTML preview (first 3 lines):")
for line in m.to_html().split("\n")[:3]:
    print(f"    {line}")

# ---------------------------------------------------------------------------
# 14. all_kinds=True — all representations in one table
# ---------------------------------------------------------------------------
# m.to_string(all_kinds=True) shows m, bel, pl, b, q in one table.
# Column v (disjunctive weights) is added automatically if m is subnormal
# (m(∅) > 0, so that b(A) > 0 for all A ⊆ Ω).
# Column w (conjunctive weights) is added automatically if m is non-dogmatic
# (m(Ω) > 0, so that q(A) > 0 for all A ⊂ Ω).

section("14. all_kinds=True — all representations in one table")

# Normal, dogmatic BBA: m(∅)=0, m(Ω)=0 → only m, bel, pl, b, q
m_dog = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(f"{DIM}# Normal dogmatic BBA — columns: m, bel, pl, b, q{R}")
print(m_dog.to_string(all_kinds=True))

# Non-dogmatic BBA: m(Ω)>0 → w column added
m_nd = DSVector.from_focal(frame, {"a": 0.3, "r": 0.3, "a,h,r": 0.4})
print(f"\n{DIM}# Non-dogmatic BBA — w column added (m(Ω)={m_nd[frozenset({'a','h','r'})]:.1f}){R}")
print(m_nd.to_string(all_kinds=True))

# Subnormal BBA: m(∅)>0 → v column added
m_sub3 = DSVector.from_focal(frame, {"": 0.1, "a": 0.5, "r": 0.4}, complete=False)
print(f"\n{DIM}# Subnormal dogmatic BBA — v column added (m(∅)={m_sub3[frozenset()]:.1f}){R}")
print(m_sub3.to_string(all_kinds=True))

# Subnormal non-dogmatic: both v and w
m_full = DSVector.from_focal(frame, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False)
print(f"\n{DIM}# Subnormal non-dogmatic — both v and w columns{R}")
print(m_full.to_string(all_kinds=True))

# LaTeX version
print(f"\n{DIM}# LaTeX output{R}")
print(m_nd.to_latex(all_kinds=True))

# ---------------------------------------------------------------------------
# 15. Conditioning and deconditioning
# ---------------------------------------------------------------------------
# Conditioning m on A gives the least committed specialization of m such
# that the complement Ā becomes impossible (pl(Ā) = 0).
# Deconditioning is the inverse: restores the least committed generalization.
#
# Two implementations: sparse (default, O(k)) and dense (O(2^n) via C_A/D_A).

section("15. Conditioning and deconditioning")

from evtools.combinations import condition, decondition
from evtools.conversions import conditioning_matrix, deconditioning_matrix

m_base = DSVector.from_focal(frame, {"a": 0.3, "h": 0.2, "a,h": 0.1, "a,h,r": 0.4})
A = frozenset({"a", "h"})

print(f"{DIM}# Original BBA{R}")
print(m_base)

print(f"\n{DIM}# Conditioning on A = {{a, h}}{R}")
print(f"{DIM}# m[A](B) = Σ_{{C∩A=B}} m(C){R}\n")
m_cond = condition(m_base, A)
print(m_cond)

print(f"\n{DIM}# Deconditioning — B → B ∪ Ā = B ∪ {{r}}{R}\n")
m_decond = decondition(m_cond, A)
print(m_decond)

# Round-trip: condition(decondition(m[A], A), A) == m[A]
ok = np.allclose(condition(m_decond, A).dense, m_cond.dense, atol=1e-10)
print(f"\n  condition(decondition(m[A], A), A) == m[A] :  {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

# Sparse vs dense
ok2 = np.allclose(
    condition(m_base, A, method="sparse").dense,
    condition(m_base, A, method="dense").dense, atol=1e-10)
print(f"  sparse == dense :  {GREEN}✓ OK{R}" if ok2 else f"  {RED}✗ MISMATCH{R}")

# Conditioning matrices C_A and D_A
print(f"\n{DIM}# Conditioning matrix C_A (2^n × 2^n specialization matrix){R}")
CA = conditioning_matrix(frame, A)
print(f"  Shape: {CA.shape},  column sums = 1: {np.allclose(CA.sum(axis=0), 1.0)}")

print(f"\n{DIM}# Deconditioning matrix D_A (2^n × 2^n generalization matrix){R}")
DA = deconditioning_matrix(frame, A)
print(f"  Shape: {DA.shape},  column sums = 1: {np.allclose(DA.sum(axis=0), 1.0)}")

print(f"\n{DIM}# C_A @ m matches sparse result{R}")
m_via_matrix = DSVector.from_dense(frame, CA @ m_base.dense)
ok3 = np.allclose(m_via_matrix.dense, m_cond.dense, atol=1e-10)
print(f"  C_A @ m == condition(m, A) :  {GREEN}✓ OK{R}" if ok3 else f"  {RED}✗ MISMATCH{R}")

# ---------------------------------------------------------------------------
# 16. Pignistic and plausibility probability transformations
# ---------------------------------------------------------------------------
# BetP and PlP transform a BBA into a probability vector of length n
# (one value per atom), used for decision making in the TBM.
#
# BetP({x}) = Σ_{A∋x} m(A) / (|A| · (1 - m(∅)))   (Smets & Kennes 1994)
# PlP({x})  = pl({x}) / Σ_{y∈Ω} pl({y})             (Cobb & Shenoy 2006)

section("16. Pignistic and plausibility probability transformations")

from evtools.conversions import betp, plp

m_demo = DSVector.from_focal(frame, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3})
print(f"{DIM}# BBA{R}")
print(m_demo)

print(f"\n{DIM}# BetP — pignistic transformation{R}")
bp = m_demo.to_betp()
for atom, val in zip(frame, bp):
    print(f"  BetP({{{atom}}}) = {val:.4f}")
print(f"  Sum = {bp.sum():.4f}")

print(f"\n{DIM}# PlP — plausibility probability{R}")
pp = m_demo.to_plp()
for atom, val in zip(frame, pp):
    print(f"  PlP({{{atom}}}) = {val:.4f}")
print(f"  Sum = {pp.sum():.4f}")

print(f"\n{DIM}# Special cases{R}")
m_vac = DSVector.from_focal(frame, {})
print(f"  Vacuous BBA → BetP = {m_vac.to_betp().round(4)} (uniform)")
m_cat = DSVector.from_focal(frame, {"a": 1.0})
print(f"  Categorical m({{a}})=1 → BetP = {m_cat.to_betp().round(4)}")

print(f"\n{DIM}# Note: result is a numpy array of length n={len(frame)}, not a DSVector{R}")
print(f"  type(m.to_betp()) = {type(m_demo.to_betp())}")
print(f"  len(m.to_betp())  = {len(m_demo.to_betp())}")


# ---------------------------------------------------------------------------
# 17. Decision criteria
# ---------------------------------------------------------------------------
# Two families:
#  - Complete preferences  → return (index, atom)
#  - Partial preferences   → return frozenset of non-dominated atoms

section("17. Decision criteria")

from evtools.decision import (
    maximin, maximax, pignistic_decision, plp_decision, probability_decision,
    hurwicz, strong_dominance, weak_dominance,
)
from evtools.conversions import betp, plp

m_dec = DSVector.from_focal(frame, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3})
print(f"{DIM}# BBA{R}")
print(m_dec)

print(f"\n{DIM}# Complete preference relations (identity utility){R}")
print(f"  maximin              = {maximin(m_dec)}")
print(f"  maximax              = {maximax(m_dec)}")
print(f"  pignistic_decision   = {pignistic_decision(m_dec)}")
print(f"  plp_decision         = {plp_decision(m_dec)}")
print(f"  hurwicz(α=0.5)       = {hurwicz(m_dec)}")
print(f"  hurwicz(α=1.0)       = {hurwicz(m_dec, alpha=1.0)}  {DIM}← ≡ maximin{R}")
print(f"  hurwicz(α=0.0)       = {hurwicz(m_dec, alpha=0.0)}  {DIM}← ≡ maximax{R}")

print(f"\n{DIM}# Generic MEU — pass the m → probability transform explicitly{R}")
print(f"  probability_decision(m, transform=plp)   = {probability_decision(m_dec, transform=plp)}  "
      f"{DIM}(default){R}")
print(f"  probability_decision(m, transform=betp)  = {probability_decision(m_dec, transform=betp)}")

print(f"\n{DIM}# Custom utility matrix U[i, j] = u(a_i, ω_j){R}")
U = np.array([[1.0, 0.0, 0.0],
              [0.0, 2.0, 0.0],
              [0.0, 0.0, 3.0]])
print(f"  pignistic_decision(m, U) = {pignistic_decision(m_dec, U)}")
print(f"  plp_decision(m, U)       = {plp_decision(m_dec, U)}")
print(f"  maximin(m, U)            = {maximin(m_dec, U)}")
print(f"  maximax(m, U)            = {maximax(m_dec, U)}")

print(f"\n{DIM}# Partial preference relations — non-dominated atoms{R}")
print(f"  strong_dominance(m) = {set(strong_dominance(m_dec))}")
print(f"  weak_dominance(m)   = {set(weak_dominance(m_dec))}")

print(f"\n{DIM}# Edge cases{R}")
m_cat_dec = DSVector.from_focal(frame, {"a": 1.0})
m_vac_dec = DSVector.from_focal(frame, {})
print(f"  Categorical {{a}} → pignistic = {pignistic_decision(m_cat_dec)}, "
      f"strong_dom = {set(strong_dominance(m_cat_dec))}")
print(f"  Vacuous     → pignistic = {pignistic_decision(m_vac_dec)} "
      f"{DIM}(picks index 0 — uniform tie-break){R}, "
      f"strong_dom = {set(strong_dominance(m_vac_dec))}")



# ---------------------------------------------------------------------------
# 18. Performance metrics (evtools.metrics) and scikit-learn interop
# ---------------------------------------------------------------------------
# evtools.metrics provides utility-discounted accuracies for partial decisions
# (Zaffalon et al. 2012). For ROC/AUC/etc. on hard predictions, extract a
# probability vector and feed it to sklearn.metrics.

section("18. Performance metrics  (evtools.metrics + sklearn)")

from evtools.metrics import (
    discounted_accuracy, u65, u80, utility_score,
    mean_u65, mean_u80, mean_discounted_accuracy,
)

print(f"{DIM}# Per-instance metrics on partial decisions{R}")
true_label = "a"
for d in [frozenset({"a"}), frozenset({"a", "h"}), frozenset({"a", "h", "r"}), frozenset({"r"})]:
    label = str(set(d)) if d else "∅"
    x = discounted_accuracy(d, true_label)
    print(f"  d = {label:<22} x={x:.4f}  u65={u65(d, true_label):.4f}  u80={u80(d, true_label):.4f}")

print(f"\n{DIM}# Mean aggregators on a paired (predictions, labels) iterable{R}")
preds  = [frozenset({"a"}),  frozenset({"a", "h"}), frozenset({"r"})]
labels = ["a",               "a",                    "a"]
print(f"  predictions = {[set(p) for p in preds]}")
print(f"  labels      = {labels}")
print(f"  mean_discounted_accuracy = {mean_discounted_accuracy(preds, labels):.4f}")
print(f"  mean_u65                 = {mean_u65(preds, labels):.4f}")
print(f"  mean_u80                 = {mean_u80(preds, labels):.4f}")

print(f"\n{DIM}# BBA-valued predictions: pl_loss (E_pl / Ẽ_pl){R}")
print(f"{DIM}# Same formula for hard and soft labels — they can be mixed.{R}")
from evtools.metrics import pl_loss, mean_pl_loss

m_pred1 = DSVector.from_focal(frame, {"a": 0.6, "h": 0.2, "r": 0.2})
m_pred2 = DSVector.from_focal(frame, {"a": 0.5, "h": 0.5})
m_soft  = DSVector.from_focal(frame, {"a": 0.5, "h": 0.5})  # uncertain between a and h

# Hard labels (E_pl)
loss_hard = pl_loss([m_pred1, m_pred2], ["a", "h"])
# Soft labels (Ẽ_pl)
loss_soft = pl_loss([m_pred1, m_pred2], [m_soft, m_soft])
# Mixed
loss_mix  = pl_loss([m_pred1, m_pred2], ["a", m_soft])
print(f"  pl_loss (hard {{a, h}})           = {loss_hard:.4f}")
print(f"  pl_loss (soft, both same m_soft)  = {loss_soft:.4f}")
print(f"  pl_loss (mixed: 'a' + m_soft)     = {loss_mix:.4f}")
print(f"  mean_pl_loss (hard, n=2)          = {mean_pl_loss([m_pred1, m_pred2], ['a', 'h']):.4f}")

print(f"\n{DIM}# Hard-prediction metrics: extract a probability vector → sklearn{R}")
try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np

    # Two BBAs as classifier outputs (both for class 'a')
    m1 = DSVector.from_focal(frame, {"a": 0.7, "a,h": 0.3})            # confident a
    m2 = DSVector.from_focal(frame, {"r": 0.5, "a,h,r": 0.5})           # confused
    bba_outputs = [m1, m2]
    y_true      = ["a", "a"]

    # Hard prediction = argmax BetP
    y_pred = [m.frame[int(np.argmax(m.to_betp()))] for m in bba_outputs]
    print(f"  y_true = {y_true}, y_pred = {y_pred}")
    print(f"  sklearn.accuracy_score = {accuracy_score(y_true, y_pred):.4f}")

    # Binary AUC: take BetP({a}) as the probability score for class 'a'
    score_a = [m.to_betp()[0] for m in bba_outputs]  # idx 0 = 'a' in this frame
    y_bin   = [1 if y == "a" else 0 for y in y_true]
    # roc_auc_score requires both classes — degenerate here but illustrative
    if len(set(y_bin)) == 2:
        print(f"  sklearn.roc_auc_score  = {roc_auc_score(y_bin, score_a):.4f}")
    else:
        print(f"  {DIM}roc_auc_score skipped: only one class in y_true (illustrative dataset){R}")
except ImportError:
    print(f"  {DIM}(install scikit-learn to run this part: `pip install scikit-learn`){R}")


# ---------------------------------------------------------------------------
# 19. Learning contextual correction parameters from labeled data
# ---------------------------------------------------------------------------
# evtools.learning provides closed-form least-squares fits of the β
# parameters of CD, CR, CN that minimize pl_loss (Pichon et al. 2016,
# Propositions 12, 14, 16). It accepts hard labels (str, → E_pl) or
# soft labels (DSVector, → Ẽ_pl), mixable in the same call.

section("19. Learning contextual corrections (fit_cd / fit_cr / fit_cn)")

from evtools.learning import fit_cd, fit_cr, fit_cn
from evtools.corrections import contextual_discount, contextual_reinforce, contextual_negate
from evtools.metrics import pl_loss

# Pichon 2016 Table 4 — Sensor 1: 4 BBAs with known true classes.
sensor1 = [
    DSVector.from_focal(frame, {"r": 0.5, "h,r": 0.3, "a,h,r": 0.2}),  # truth = a
    DSVector.from_focal(frame, {"h": 0.5, "r": 0.2, "a,h,r": 0.3}),    # truth = h
    DSVector.from_focal(frame, {"h": 0.4, "a,r": 0.6}),                # truth = a
    DSVector.from_focal(frame, {"a,r": 0.6, "h,r": 0.4}),              # truth = r
]
truth1 = ["a", "h", "a", "r"]

print(f"{DIM}# Source outputs and ground truth (Pichon 2016 Table 4, Sensor 1){R}")
for i, (m_S, y) in enumerate(zip(sensor1, truth1)):
    print(f"  o{i+1}: pl(singletons) = {m_S.contour().round(3)}, truth = {y!r}")

print(f"\n{DIM}# Baseline pl_loss (no correction){R}")
loss0 = pl_loss(sensor1, truth1)
print(f"  pl_loss = {loss0:.4f}")

print(f"\n{DIM}# Fit each correction → apply → measure pl_loss{R}")
print(f"{DIM}#   (expected from Pichon 2016 Table 6, Sensor 1):{R}")
print(f"{DIM}#   CD: β=(0.76, 1.00, 1.00),  pl_loss=3.39{R}")
print(f"{DIM}#   CR: β=(0.94, 0.66, 0.38),  pl_loss=2.33{R}")
print(f"{DIM}#   CN: β=(0.33, 1.00, 0.45),  pl_loss=2.59{R}")
print()

for name, fit_fn, apply_fn in [
    ("CD", fit_cd, contextual_discount),
    ("CR", fit_cr, contextual_reinforce),
    ("CN", fit_cn, contextual_negate),
]:
    betas    = fit_fn(sensor1, truth1)
    corrected = [apply_fn(m, betas) for m in sensor1]
    loss     = pl_loss(corrected, truth1)
    # Print β as (β_a, β_h, β_r). For CD keys are singletons; for CR/CN they are complements.
    if name == "CD":
        bs = tuple(betas[frozenset({a})] for a in frame)
    else:
        omega = frozenset(frame)
        bs = tuple(betas[omega - {a}] for a in frame)
    print(f"  {name}:  β = ({bs[0]:.2f}, {bs[1]:.2f}, {bs[2]:.2f})   pl_loss = {loss:.4f}")

print(f"\n{DIM}# Soft labels: same API, returns Ẽ_pl optimal β{R}")
soft_labels = [DSVector.from_focal(frame, {y: 1.0}) for y in truth1]
betas_soft  = fit_cd(sensor1, soft_labels)
print(f"  fit_cd with categorical soft labels matches hard:")
print(f"    β values (sorted) = {sorted(round(v, 4) for v in betas_soft.values())}")

print(f"\n{DIM}# Synthesizing soft labels from hard ones (Mutmainah 2021, Algorithm 2){R}")
print(f"{DIM}# Based on Côme et al. (2009) and Quost et al. (2017).{R}")
from evtools.learning import hard_to_soft_labels

rng = np.random.default_rng(seed=42)
synthetic_soft = hard_to_soft_labels(truth1, frame, mu=0.5, var=0.04, rng=rng)
for hard, sft in zip(truth1, synthetic_soft):
    print(f"  hard {hard!r} → soft contour = {sft.contour().round(3)}")

# Reuse the synthesized soft labels as targets for Ẽ_pl
betas_synth = fit_cd(sensor1, synthetic_soft)
print(f"\n  β learned from these soft labels:")
print(f"    {tuple(round(betas_synth[frozenset({a})], 3) for a in frame)}")


# ---------------------------------------------------------------------------
# 20. Per-group learning of contextual corrections (Mutmainah 2021, Alg. 1)
# ---------------------------------------------------------------------------
# fit_per_group partitions the training set by partial decision (strong or
# weak dominance) and fits the best of CD/CR/CN on each group. apply_per_group
# uses the chosen correction at predict time, falling back to a global model
# for unseen partial decisions. Soft labels (DSVector) work transparently:
# Chapter 4 (hard) and Section 5.3 (soft) of the thesis share the same API.

section("20. Per-group learning  (fit_per_group + apply_per_group)")

from evtools.learning import fit_per_group, apply_per_group
from evtools.decision import strong_dominance, weak_dominance

print(f"{DIM}# Reusing Pichon 2016 Sensor 1 dataset as the training set{R}")
model = fit_per_group(sensor1, truth1, dominance=strong_dominance)
print(f"  number of groups: {len(model.groups)}")
for d, gc in model.groups.items():
    label = "{" + ", ".join(sorted(d)) + "}" if d else "∅"
    print(f"    group {label:<12} → kind={gc.kind!r:<5} loss={gc.loss:.4f}")
print(f"  fallback        → kind={model.fallback.kind!r:<5} loss={model.fallback.loss:.4f}")

print(f"\n{DIM}# Compare per-group correction vs no correction vs single-global{R}")
loss_baseline = pl_loss(sensor1, truth1)
loss_global   = model.fallback.loss
corrected     = apply_per_group(model, sensor1)
loss_per_group = pl_loss(corrected, truth1)
print(f"  pl_loss baseline    : {loss_baseline:.4f}")
print(f"  pl_loss global CR   : {loss_global:.4f}  {DIM}(best single correction){R}")
print(f"  pl_loss per-group   : {loss_per_group:.4f}  {DIM}← best{R}")

print(f"\n{DIM}# Same API works with weak dominance{R}")
model_wd = fit_per_group(sensor1, truth1, dominance=weak_dominance)
corrected_wd = apply_per_group(model_wd, sensor1)
print(f"  pl_loss per-group (weak dom) : {pl_loss(corrected_wd, truth1):.4f}")

print(f"\n{DIM}# Same API works with soft labels (Section 5.3 of the thesis){R}")
rng = np.random.default_rng(seed=2026)
soft_train = hard_to_soft_labels(truth1, frame, rng=rng)
model_soft = fit_per_group(sensor1, soft_train, dominance=strong_dominance)
corrected_soft = apply_per_group(model_soft, sensor1)
print(f"  Ẽ_pl per-group (soft labels) : {pl_loss(corrected_soft, soft_train):.4f}")


# ---------------------------------------------------------------------------
# 21. EkNN classifier with K-fold cross-validation
# ---------------------------------------------------------------------------
# evtools.classifiers.EkNN implements the evidence-theoretic k-NN rule of
# Denoeux 1995 (heuristic γ) and Zouhal & Denoeux 1998 (γ optimized by
# minimizing the pl-loss on the training set).
#
# This section demonstrates EkNN on the classic Iris dataset, with both
# K-fold CV (random splits) and stratified K-fold CV (preserves class
# proportions in each fold — the standard practice for classification).

section("21. EkNN classifier  +  K-fold CV  (Iris and Wine)")

try:
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import KFold, StratifiedKFold
    sklearn_available = True
except ImportError:
    sklearn_available = False
    print(f"  {DIM}(install scikit-learn to run this section: `pip install scikit-learn`){R}")

if sklearn_available:
    from sklearn.preprocessing import StandardScaler
    from evtools.classifiers import EkNN
    from evtools.metrics import mean_pl_loss

    def _kfold_eval(X, y, splitter, k=5, optimize=True):
        """Run CV with the given splitter; return (acc_mean, acc_std, plloss_mean, plloss_std)."""
        accs, plloss = [], []
        # split() takes y for stratified, ignores it for plain KFold
        split_args = (X, y) if isinstance(splitter, StratifiedKFold) else (X,)
        for train_idx, test_idx in splitter.split(*split_args):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            clf = EkNN(k=k, optimize=optimize).fit(X_tr, y_tr)
            accs.append((clf.predict(X_te) == y_te).mean())
            bbas = clf.predict_bba(X_te)
            plloss.append(mean_pl_loss(bbas, [str(c) for c in y_te]))
        return np.mean(accs), np.std(accs), np.mean(plloss), np.std(plloss)

    DATASETS = [
        ("Iris", load_iris(return_X_y=True)),
        ("Wine", load_wine(return_X_y=True)),
    ]

    for name, (X, y) in DATASETS:
        print(f"{DIM}# {name}: n={len(X)}, n_features={X.shape[1]}, "
              f"n_classes={len(set(y))}{R}")

        # 5-fold CV (random splits) — raw features
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        a_no, _, _, _ = _kfold_eval(X, y, kfold, optimize=False)
        a, sa, pl, spl = _kfold_eval(X, y, kfold, optimize=True)
        print(f"  KFold       no-optim accuracy = {a_no:.4f}")
        print(f"  KFold       +optim   accuracy = {a:.4f} ± {sa:.4f}, "
              f"mean_pl_loss = {pl:.4f} ± {spl:.4f}")

        # 5-fold Stratified CV (preserves class proportions)
        skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        a, sa, pl, spl = _kfold_eval(X, y, skfold, optimize=True)
        print(f"  Stratified  +optim   accuracy = {a:.4f} ± {sa:.4f}, "
              f"mean_pl_loss = {pl:.4f} ± {spl:.4f}")

        # Same with standardized features (zero mean, unit variance)
        # — EkNN uses Euclidean distance, so feature scaling matters.
        X_std = StandardScaler().fit_transform(X)
        a, sa, pl, spl = _kfold_eval(X_std, y, skfold, optimize=True)
        print(f"  Stratified  +optim   accuracy = {a:.4f} ± {sa:.4f}, "
              f"mean_pl_loss = {pl:.4f} ± {spl:.4f}    {DIM}(standardized features){R}")
        print()

    # -----------------------------------------------------------
    # Notes
    # -----------------------------------------------------------
    print(f"{DIM}# Notes:{R}")
    print(f"{DIM}#  - K-fold:           random splits, may give imbalanced folds.{R}")
    print(f"{DIM}#  - Stratified K-fold: each fold preserves class proportions.{R}")
    print(f"{DIM}#                       Standard practice in classification.{R}")
    print(f"{DIM}#  - mean_pl_loss is a richer metric than accuracy: it uses the{R}")
    print(f"{DIM}#    full BBA (singleton + Ω masses), not just the argmax label.{R}")
    print(f"{DIM}#  - Feature scaling matters! EkNN uses Euclidean distance.{R}")
    print(f"{DIM}#    Wine has features on very different scales (alcohol vs ash) →{R}")
    print(f"{DIM}#    standardization ~doubles accuracy. Iris features already comparable.{R}")


# ---------------------------------------------------------------------------
# 22. EkNN on UCI benchmarks — Sonar and Ionosphere
# ---------------------------------------------------------------------------
# These two binary-classification UCI datasets are the standard benchmarks
# used in Zouhal & Denoeux 1998 (Figs 7-8 for Ionosphere, 11-12 for Sonar)
# to evaluate the EkNN rule with optimized γ. We load them via OpenML and
# evaluate EkNN with 5-fold Stratified CV, reporting accuracy + mean_pl_loss.

section("22. EkNN on UCI benchmarks  (Sonar, Ionosphere)")

if sklearn_available:
    from sklearn.datasets import fetch_openml

    def _evaluate(X, y, name, k=5, n_splits=5, seed=42):
        """Run 5-fold Stratified CV with EkNN and return (acc, plloss) as means ± stds."""
        skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs, plloss = [], []
        for train_idx, test_idx in skfold.split(X, y):
            clf = EkNN(k=k, optimize=True).fit(X[train_idx], y[train_idx])
            accs.append((clf.predict(X[test_idx]) == y[test_idx]).mean())
            bbas = clf.predict_bba(X[test_idx])
            plloss.append(mean_pl_loss(bbas, [str(c) for c in y[test_idx]]))
        return (np.mean(accs), np.std(accs)), (np.mean(plloss), np.std(plloss))

    print(f"{DIM}# 5-fold Stratified CV, EkNN k=5, optimize=True (Pl-loss){R}")
    print(f"{DIM}# Reference: Zouhal & Denoeux 1998, Table II — best k chosen per dataset.{R}")
    print()

    for name, openml_id in [("Sonar", 40)             # OpenML "sonar" id=40
                          , ("Ionosphere", 59)]:      # OpenML "ionosphere" id=59
        try:
            data = fetch_openml(data_id=openml_id, as_frame=False, parser="liac-arff")
            X = np.asarray(data.data, dtype=float)
            y = np.asarray(data.target)
            (acc_m, acc_s), (pl_m, pl_s) = _evaluate(X, y, name)
            print(f"  {name:<11}  n={len(X):>3}, d={X.shape[1]:>2}: "
                  f"accuracy={acc_m:.4f} ± {acc_s:.4f},  "
                  f"mean_pl_loss={pl_m:.4f} ± {pl_s:.4f}")
        except Exception as e:
            # Network / OpenML issues — fall back gracefully
            print(f"  {name:<11}  {DIM}skipped (could not load: {type(e).__name__}){R}")

    print()
    print(f"{DIM}# Published reference (Zouhal & Denoeux 1998, Table II, error = 1 - accuracy):{R}")
    print(f"{DIM}#   Sonar       ETO best ≈ 0.13   → accuracy ≈ 0.87{R}")
    print(f"{DIM}#   Ionosphere  ETO best ≈ 0.08   → accuracy ≈ 0.92{R}")
    print(f"{DIM}# Caveats: published numbers use a fixed train/test split, EUC distance,{R}")
    print(f"{DIM}#   and the optimized γ from gradient descent. Our 5-fold StratifiedCV{R}")
    print(f"{DIM}#   protocol with default settings gives a slightly different estimate.{R}")


# ---------------------------------------------------------------------------
# 23. EkNN classifier followed by a learned contextual correction (CD/CR/CN)
# ---------------------------------------------------------------------------
# Three-way split protocol:
#   L1 (50%): train the EkNN classifier
#   L2 (25%): fit each contextual correction (CD, CR, CN) on the EkNN's BBA
#             outputs; pick the one with lowest mean_pl_loss on L2
#   Test (25%): apply the chosen correction on the held-out test set;
#               compare mean_pl_loss before vs after correction.
#
# This mirrors the protocol of Mutmainah 2021 (Chap. 4) at a coarser
# granularity (no per-group split) — useful to demonstrate that a learned
# correction can improve a fixed classifier's BBA outputs.

section("23. EkNN + learned correction (CD / CR / CN)  on Wine raw")

if sklearn_available:
    from sklearn.model_selection import train_test_split
    from evtools.learning    import fit_cd, fit_cr, fit_cn
    from evtools.corrections import contextual_discount, contextual_reinforce, contextual_negate
    from collections import Counter

    candidates = [
        ("CD", fit_cd, contextual_discount),
        ("CR", fit_cr, contextual_reinforce),
        ("CN", fit_cn, contextual_negate),
    ]

    # Wine RAW (no scaling) — EkNN baseline ~0.72 accuracy, plenty of room for corrections.
    X_w, y_w = load_wine(return_X_y=True)
    print(f"{DIM}# Wine raw — n={len(X_w)}, n_features={X_w.shape[1]}, "
          f"n_classes={len(set(y_w))}{R}")
    print(f"{DIM}# Three-way split: L1 (50%) trains EkNN, L2 (25%) selects "
          f"best correction, test (25%) evaluates.{R}")
    print(f"{DIM}# Repeated over 10 random seeds for stability.{R}\n")

    seeds = list(range(10))
    losses_before, losses_after, picks = [], [], []
    for seed in seeds:
        X_L1, X_rest, y_L1, y_rest = train_test_split(
            X_w, y_w, test_size=0.5, stratify=y_w, random_state=seed)
        X_L2, X_test, y_L2, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)

        clf = EkNN(k=5, optimize=True).fit(X_L1, y_L1)

        # 2. Pick best correction on L2 (validation set for the correction).
        bbas_L2   = clf.predict_bba(X_L2)
        labels_L2 = [str(c) for c in y_L2]
        best = None
        for name, fit_fn, apply_fn in candidates:
            betas = fit_fn(bbas_L2, labels_L2)
            corrected = [apply_fn(m, betas) for m in bbas_L2]
            loss = mean_pl_loss(corrected, labels_L2)
            if best is None or loss < best[3]:
                best = (name, betas, apply_fn, loss)
        best_name, best_betas, best_apply, _ = best
        picks.append(best_name)

        # 3. Evaluate on test before/after correction.
        bbas_test   = clf.predict_bba(X_test)
        labels_test = [str(c) for c in y_test]
        losses_before.append(mean_pl_loss(bbas_test, labels_test))
        corrected_test = [best_apply(m, best_betas) for m in bbas_test]
        losses_after.append(mean_pl_loss(corrected_test, labels_test))

    delta = np.array(losses_before) - np.array(losses_after)
    n_pos = int(np.sum(delta > 0))
    n_seeds = len(seeds)

    print(f"  Mean test mean_pl_loss BEFORE correction : "
          f"{np.mean(losses_before):.4f} ± {np.std(losses_before):.4f}")
    print(f"  Mean test mean_pl_loss AFTER  correction : "
          f"{np.mean(losses_after):.4f} ± {np.std(losses_after):.4f}")
    print(f"  Mean Δ (before − after)                  : "
          f"{np.mean(delta):+.4f}    "
          f"({GREEN}improvement{R} in {n_pos}/{n_seeds} runs)")
    print(f"  Best correction picked over the {n_seeds} runs : "
          f"{dict(Counter(picks))}")

    print(f"\n{DIM}# Notes:{R}")
    print(f"{DIM}#  - L2 must be DISJOINT from L1 (otherwise the correction overfits){R}")
    print(f"{DIM}#  - mean_pl_loss is the criterion being optimized; on Wine raw the{R}")
    print(f"{DIM}#    correction reliably improves the EkNN baseline (10/10 seeds).{R}")
    print(f"{DIM}#  - For finer-grained corrections (per partial-decision group),{R}")
    print(f"{DIM}#    use evtools.learning.fit_per_group (Mutmainah 2021, Alg. 1).{R}")


# ---------------------------------------------------------------------------
# 24. EkNN + per-group correction (Mutmainah 2021, Algorithm 1)
# ---------------------------------------------------------------------------
# Same three-way split as Section 23, but the L2 correction step uses
# fit_per_group: instances are partitioned by their partial decision
# (strong or weak dominance), and the best of CD/CR/CN is chosen for
# each group separately. A fallback global correction handles partial
# decisions that don't appear in L2.
#
# This generalizes Section 23 — picking a single global correction is
# the "ungrouped" special case (one fallback group only).

section("24. EkNN + per-group correction  (Mutmainah 2021 Alg. 1)")

if sklearn_available:
    from evtools.decision import strong_dominance, weak_dominance
    from evtools.learning import fit_per_group, apply_per_group
    # train_test_split + load_wine + Counter already imported in Section 23

    # Same Wine raw + 10-seed protocol as Section 23.
    X_w, y_w = load_wine(return_X_y=True)
    print(f"{DIM}# Wine raw — 3-way split L1 (50%) / L2 (25%) / Test (25%), 10 seeds.{R}\n")

    seeds = list(range(10))
    losses_no    = []   # baseline (no correction)
    losses_glob  = []   # single best correction (Section 23)
    losses_sd    = []   # per-group with strong dominance
    losses_wd    = []   # per-group with weak dominance

    for seed in seeds:
        X_L1, X_rest, y_L1, y_rest = train_test_split(
            X_w, y_w, test_size=0.5, stratify=y_w, random_state=seed)
        X_L2, X_test, y_L2, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)

        clf = EkNN(k=5, optimize=True).fit(X_L1, y_L1)
        bbas_L2   = clf.predict_bba(X_L2)
        labels_L2 = [str(c) for c in y_L2]
        bbas_test = clf.predict_bba(X_test)
        labels_test = [str(c) for c in y_test]

        # Baseline (no correction)
        losses_no.append(mean_pl_loss(bbas_test, labels_test))

        # Single best correction on L2 (Section 23 protocol)
        best = None
        for name, fit_fn, apply_fn in candidates:
            betas = fit_fn(bbas_L2, labels_L2)
            loss  = mean_pl_loss([apply_fn(m, betas) for m in bbas_L2], labels_L2)
            if best is None or loss < best[3]:
                best = (name, betas, apply_fn, loss)
        losses_glob.append(mean_pl_loss(
            [best[2](m, best[1]) for m in bbas_test], labels_test))

        # Per-group with strong dominance
        model_sd = fit_per_group(bbas_L2, labels_L2, dominance=strong_dominance)
        losses_sd.append(mean_pl_loss(apply_per_group(model_sd, bbas_test), labels_test))

        # Per-group with weak dominance
        model_wd = fit_per_group(bbas_L2, labels_L2, dominance=weak_dominance)
        losses_wd.append(mean_pl_loss(apply_per_group(model_wd, bbas_test), labels_test))

    losses_no   = np.array(losses_no)
    losses_glob = np.array(losses_glob)
    losses_sd   = np.array(losses_sd)
    losses_wd   = np.array(losses_wd)

    def _fmt(arr, ref=None):
        s = f"{arr.mean():.4f} ± {arr.std():.4f}"
        if ref is not None:
            n_better = int(np.sum(ref - arr > 0))
            s += f"   ({n_better}/{len(arr)} better than baseline)"
        return s

    print(f"  No correction         : mean_pl_loss = {_fmt(losses_no)}")
    print(f"  Single best (Sec. 23) : mean_pl_loss = {_fmt(losses_glob, losses_no)}")
    print(f"  Per-group (SD)        : mean_pl_loss = {_fmt(losses_sd, losses_no)}")
    print(f"  Per-group (WD)        : mean_pl_loss = {_fmt(losses_wd, losses_no)}")

    print(f"\n{DIM}# Notes:{R}")
    print(f"{DIM}#  - Per-group fits a different correction in each partial-decision{R}")
    print(f"{DIM}#    group. With small L2 (n=44) and few groups (≤7 with SD),{R}")
    print(f"{DIM}#    each group has very few samples → the per-group estimate{R}")
    print(f"{DIM}#    can be noisier than a single global correction.{R}")
    print(f"{DIM}#  - Per-group shines on larger training sets — see Mutmainah 2021{R}")
    print(f"{DIM}#    Tables 4.4-4.7 (UCI datasets, 10-fold CV).{R}")
