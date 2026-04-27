"""
evtools — Tutorial
==================

This tutorial walks through all main features of evtools, using a running
example from belief function theory: a sensor classifies aerial targets as
airplane (a), helicopter (h), or rocket (r).

Sections
--------
1.  Building a BBA        — from_focal, from_dense, from_sparse
2.  Accessing values      — sparse, dense, iteration, is_valid
3.  Subnormal BBA         — m(∅) > 0
4.  Conversions           — bel, pl, b, q, v, w
5.  Round-trip            — consistency checks
6.  Low-level API         — numpy arrays via conversions module
7.  Combination rules     — CRC, Dempster, DRC (distinct sources)
8.  DRC                   — disjunctive rule
9.  Cautious & Bold       — nondistinct sources
10. Correction mechanisms — discount, contextual_discount, CR, CdD, CN
11. Simple MFs            — DSVector.simple, DSVector.negative_simple
    Decombination         — decombine_crc, decombine_drc
12. Display formats       — ansi, plain, html, latex
13. display_all           — all representations in one table
14. Conditioning          — condition(m, A), decondition(m, A), C_A, D_A
15. BetP and PlP          — pignistic and plausibility probability transformations

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
#   A^β  (DSVector.simple)          — focal sets Ω (mass β) and A (mass 1−β)
#   A_β  (DSVector.negative_simple) — focal sets ∅ (mass β) and A (mass 1−β)

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

from evtools.display import repr_plain, repr_html, repr_latex

m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})

# ANSI (default __repr__)
print(f"{DIM}# repr_ansi — colored terminal (default){R}\n")
print(m)

# Plain text
print(f"\n{DIM}# repr_plain — no colors, for logs and files{R}\n")
print(repr_plain(m))

# Column header adapts to kind
print(f"\n{DIM}# Column header adapts to the kind{R}")
for kind_fn, label in [("to_bel","Belief"), ("to_pl","Plausibility"), ("to_q","Implicability")]:
    v = getattr(m, kind_fn)()
    print(f"\n{DIM}# {label} function — header shows '{v.kind.value}'{R}")
    print(repr_plain(v))

# LaTeX
print(f"\n{DIM}# repr_latex — ready to paste in a LaTeX paper{R}\n")
print(repr_latex(m))

# display() method
print(f"\n{DIM}# display(fmt) — explicit format selection{R}")
print(f"  Available: ansi, plain, html, latex\n")
print(m.display("plain"))

# Jupyter auto-display via _repr_html_
print(f"\n{DIM}# In Jupyter: DSVector renders as HTML table automatically{R}")
print(f"{DIM}# via _repr_html_() — no extra call needed{R}")
print(f"  HTML preview (first 3 lines):")
for line in repr_html(m).split("\n")[:3]:
    print(f"    {line}")

# ---------------------------------------------------------------------------
# 14. display_all — all representations in one table
# ---------------------------------------------------------------------------
# display_all(m) shows m, bel, pl, b, q in one table.
# Column v (disjunctive weights) is added automatically if m is subnormal
# (m(∅) > 0, so that b(A) > 0 for all A ⊆ Ω).
# Column w (conjunctive weights) is added automatically if m is non-dogmatic
# (m(Ω) > 0, so that q(A) > 0 for all A ⊂ Ω).

section("14. display_all — all representations in one table")

from evtools.display import display_all

# Normal, dogmatic BBA: m(∅)=0, m(Ω)=0 → only m, bel, pl, b, q
m_dog = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(f"{DIM}# Normal dogmatic BBA — columns: m, bel, pl, b, q{R}")
print(display_all(m_dog, "plain"))

# Non-dogmatic BBA: m(Ω)>0 → w column added
m_nd = DSVector.from_focal(frame, {"a": 0.3, "r": 0.3, "a,h,r": 0.4})
print(f"\n{DIM}# Non-dogmatic BBA — w column added (m(Ω)={m_nd[frozenset({'a','h','r'})]:.1f}){R}")
print(display_all(m_nd, "plain"))

# Subnormal BBA: m(∅)>0 → v column added
m_sub3 = DSVector.from_focal(frame, {"": 0.1, "a": 0.5, "r": 0.4}, complete=False)
print(f"\n{DIM}# Subnormal dogmatic BBA — v column added (m(∅)={m_sub3[frozenset()]:.1f}){R}")
print(display_all(m_sub3, "plain"))

# Subnormal non-dogmatic: both v and w
m_full = DSVector.from_focal(frame, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False)
print(f"\n{DIM}# Subnormal non-dogmatic — both v and w columns{R}")
print(display_all(m_full, "plain"))

# display_all via method
print(f"\n{DIM}# Via method: m.display_all('plain'){R}")
print(m_nd.display_all("plain"))

# LaTeX version
print(f"\n{DIM}# LaTeX output{R}")
print(display_all(m_nd, "latex"))

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
# BetP({x}) = Σ_{A∋x} m(A) / (|A| · (1 − m(∅)))   (Smets & Kennes 1994)
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
