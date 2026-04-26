"""
evtools — Tutorial
==================

This tutorial walks through the main features of evtools, using a simple
target recognition example: a sensor classifies aerial targets as
airplane (a), helicopter (h), or rocket (r).

Reference:
    D. Mercier, B. Quost, T. Denoeux. Refined modeling of sensor reliability
    in the belief function framework using contextual discounting.
    Information Fusion, 9(2), 246-258, 2008.
"""

import numpy as np
from evtools.dsvector import DSVector, Kind

B  = "\033[1m"
R  = "\033[0m"
DIM = "\033[2m"

def section(title):
    print(f"\n{B}{'─' * 60}{R}")
    print(f"{B}  {title}{R}")
    print(f"{B}{'─' * 60}{R}\n")


frame = ["a", "h", "r"]  # airplane, helicopter, rocket

# ---------------------------------------------------------------------------
# 1. Building a mass function
# ---------------------------------------------------------------------------

section("1. Building a mass function")

# A sensor hesitates between airplane and rocket
# Missing mass (0.0) is automatically assigned to Ω = {a, h, r}
print(f"{DIM}# from_focal — human-friendly string keys{R}")
m = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(m)

# From a dense numpy array
# Index order: ∅, {a}, {h}, {a,h}, {r}, {a,r}, {h,r}, {a,h,r}
print(f"\n{DIM}# from_dense — numpy array{R}")
array = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
m2 = DSVector.from_dense(frame, array)
print(m2)

# From a sparse dict of frozensets
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
print(f"m[{{h}}]    = {m[frozenset({'h'})]}  {DIM}(not a focal element){R}")
print(f"n_focal   = {m.n_focal}")
print(f"dense     = {m.dense}")

print(f"\n{DIM}# Iterating over focal elements:{R}")
for subset, value in m:
    label = "{" + ", ".join(sorted(subset)) + "}" if subset else "∅"
    print(f"  m({label}) = {value}")

# ---------------------------------------------------------------------------
# 3. Subnormal BBA  (m(∅) > 0)
# ---------------------------------------------------------------------------

section("3. Subnormal BBA  —  m(∅) > 0")

m_sub = DSVector.from_focal(
    frame,
    {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2},
    complete=False,
)
print(m_sub)

# ---------------------------------------------------------------------------
# 4. Conversions
# ---------------------------------------------------------------------------

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
# 5. Round-trip consistency
# ---------------------------------------------------------------------------

section("5. Round-trip consistency")

for kind in [Kind.BEL, Kind.PL, Kind.B, Kind.Q]:
    back = m.to(kind).to(Kind.M)
    ok = np.allclose(back.dense, m.dense, atol=1e-10)
    status = f"\033[32m✓ OK\033[0m" if ok else f"\033[31m✗ MISMATCH\033[0m"
    print(f"  m → {kind.value:>3} → m   {status}")

for kind in [Kind.V, Kind.W]:
    back = m_sub.to(kind).to(Kind.M)
    ok = np.allclose(back.dense, m_sub.dense, atol=1e-10)
    status = f"\033[32m✓ OK\033[0m" if ok else f"\033[31m✗ MISMATCH\033[0m"
    print(f"  m_sub → {kind.value} → m   {status}")

# ---------------------------------------------------------------------------
# 6. Low-level conversions API
# ---------------------------------------------------------------------------

section("6. Low-level conversions API  (numpy arrays)")

from evtools.conversions import mtobel, mtopl, mtob

m_array = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
print(f"bel : {mtobel(m_array)}")
print(f"pl  : {mtopl(m_array)}")
print(f"b   : {mtob(m_array)}")

# ---------------------------------------------------------------------------
# 7. Combination rules
# ---------------------------------------------------------------------------

section("7. Combination rules")

from evtools.combinations import crc, drc

# Two sensors observing the same target
# Sensor 1: hesitates between airplane and rocket
s1 = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})

# Sensor 2: hesitates between helicopter and rocket, with some ignorance
s2 = DSVector.from_focal(frame, {"h": 0.3, "r": 0.4, "a,h,r": 0.3})

print(f"{DIM}# Sensor 1{R}")
print(s1)

print(f"\n{DIM}# Sensor 2{R}")
print(s2)

# CRC — sources are distinct and reliable
print(f"\n{DIM}# CRC (sparse) — m1 & m2{R}")
m12 = crc(s1, s2)
print(m12)

print(f"\n{DIM}# CRC (dense / FMT) — same result{R}")
m12_dense = crc(s1, s2, method="dense")
print(m12_dense)

print(f"\n{DIM}# Operator shortcut: s1 & s2{R}")
print(s1 & s2)

# Verify commutativity
ok = np.allclose((s1 & s2).dense, (s2 & s1).dense, atol=1e-10)
GREEN = "\033[32m"
print(f"\n  Commutativity s1 & s2 == s2 & s1 :  {GREEN}✓ OK{R}" if ok else "  ✗ MISMATCH")

# Conflict: mass on ∅ indicates contradiction between sources
conflict = m12[frozenset()]
print(f"  Conflict m(∅) = {conflict:.4f}")

# Dempster's rule — normalized CRC
print(f"\n{DIM}# Dempster's rule (normalized CRC) — s1 @ s2{R}")
m12_dempster = s1 @ s2
print(m12_dempster)
print(f"  {DIM}(conflict absorbed, m(∅) = 0){R}")

# DRC
section("8. Disjunctive Rule of Combination (DRC)")

print(f"{DIM}# DRC — at least one source is reliable, we don't know which{R}")
print(f"{DIM}# Result is less committed (larger focal elements){R}\n")
m12_drc = drc(s1, s2)
print(f"{DIM}# DRC (sparse) — m1 | m2{R}")
print(m12_drc)

print(f"\n{DIM}# Operator shortcut: s1 | s2{R}")
print(s1 | s2)

section("9. Cautious and Bold rules (nondistinct sources)")

from evtools.combinations import cautious, bold

# Cautious: nondogmatic BBAs (m(Ω) > 0)
c1 = DSVector.from_focal(frame, {"a": 0.3, "h": 0.2, "a,h,r": 0.5})
c2 = DSVector.from_focal(frame, {"a": 0.4, "r": 0.1, "a,h,r": 0.5})

print(f"{DIM}# Cautious rule — sources reliable but possibly overlapping{R}")
print(f"{DIM}# Commutative, associative, idempotent{R}\n")
print(f"{DIM}# Source 1{R}")
print(c1)
print(f"\n{DIM}# Source 2{R}")
print(c2)
print(f"\n{DIM}# Cautious combination{R}")
print(cautious(c1, c2))

# Idempotence: combining with itself leaves unchanged
ok = np.allclose(cautious(c1, c1).dense, c1.dense, atol=1e-10)
GREEN = "\033[32m"
R2 = "\033[0m"
print(f"\n  Idempotence c1 ∧ c1 == c1 :  {GREEN}✓ OK{R2}" if ok else "  ✗ MISMATCH")

# Bold: subnormal BBAs (m(∅) > 0)
b1 = DSVector.from_focal(frame, {"": 0.1, "a": 0.4, "a,h,r": 0.5}, complete=False)
b2 = DSVector.from_focal(frame, {"": 0.2, "r": 0.3, "a,h,r": 0.5}, complete=False)

print(f"\n{DIM}# Bold disjunctive rule — sources possibly overlapping, ≥1 reliable{R}")
print(f"{DIM}# Requires subnormal BBAs (m(∅) > 0){R}\n")
print(bold(b1, b2))


# ---------------------------------------------------------------------------
# 10. Correction mechanisms
# ---------------------------------------------------------------------------

section("10. Correction mechanisms")

from evtools.corrections import (
    discount, contextual_discount, theta_contextual_discount,
    contextual_reinforce, contextual_dediscount, contextual_dereinforce,
    contextual_negate,
)

# Sensor hesitates between airplane and rocket
s = DSVector.from_focal(frame, {"a": 0.5, "r": 0.5})
print(f"{DIM}# Original BBA{R}")
print(s)

# --- Classical discounting ---
print(f"\n{DIM}# Classical discounting β=0.6 — source 60% reliable{R}")
print(discount(s, 0.4))

# --- Contextual discounting ---
print(f"\n{DIM}# Contextual discounting — unreliable only when airplane (β_a=0.6){R}")
betas_cd = {
    frozenset({"a"}): 0.6,
    frozenset({"h"}): 1.0,
    frozenset({"r"}): 1.0,
}
mcd = contextual_discount(s, betas_cd)
print(mcd)

# --- Θ-contextual discounting ---
print(f"\n{DIM}# Θ-contextual discounting — Θ={{a}},{{h,r}}{R}")
betas_theta = {frozenset({"a"}): 0.4, frozenset({"h","r"}): 0.9}
print(theta_contextual_discount(s, betas_theta))

# --- Contextual reinforcement ---
print(f"\n{DIM}# Contextual reinforcement — dual of discounting{R}")
betas_cr = {frozenset({"a"}): 0.6, frozenset({"h"}): 1.0, frozenset({"r"}): 1.0}
print(contextual_reinforce(s, betas_cr))

# --- CdD: inverse of CD ---
print(f"\n{DIM}# Contextual de-discounting — reverses the CD above{R}")
mdd = contextual_dediscount(mcd, betas_cd)
print(mdd)
ok = np.allclose(mdd.dense, s.dense, atol=1e-6)
print(f"  Recovers original: {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")

# --- CN: contextual negating ---
print(f"\n{DIM}# Contextual negating — source lies for some contexts{R}")
print(contextual_negate(s, {frozenset({"a"}): 0.7}))

# --- is_valid ---
print(f"\n{DIM}# is_valid — useful after decombination operations{R}")
print(f"  Original BBA is_valid : {s.is_valid}")
print(f"  Discounted   is_valid : {discount(s, 0.4).is_valid}")

# ---------------------------------------------------------------------------
# 11. Simple MFs and decombination
# ---------------------------------------------------------------------------

section("11. Simple MFs and decombination")

from evtools.combinations import decombine_crc, decombine_drc

# --- DSVector.simple and DSVector.negative_simple ---
print(f"{DIM}# Simple MF A^β — focal sets Ω and A{R}")
s = DSVector.simple(frame, frozenset({"a"}), beta=0.6)
print(s)

print(f"\n{DIM}# Negative simple MF θ^β — focal sets ∅ and θ{R}")
ns = DSVector.negative_simple(frame, frozenset({"a"}), beta=0.4)
print(ns)

# --- decombine_crc ---
print(f"\n{DIM}# decombine_crc: m1 6∩ m2 — removes m2 from a conjunctive combination{R}")
m1 = DSVector.from_focal(frame, {"a": 0.4, "a,h,r": 0.6})
m2 = DSVector.from_focal(frame, {"h": 0.3, "a,h,r": 0.7})
from evtools.combinations import crc
m12 = crc(m1, m2)
m1_recovered = decombine_crc(m12, m2)
ok = m1_recovered.is_valid and np.allclose(m1_recovered.dense, m1.dense, atol=1e-6)
print(f"  crc(m1, m2) then decombine_crc(m12, m2) recovers m1: {GREEN}✓ OK{R}" if ok else f"  {RED}✗ MISMATCH{R}")
print(m1_recovered)

# --- decombine_drc ---
print(f"\n{DIM}# decombine_drc: m1 6∪ m2 — removes m2 from a disjunctive combination{R}")
m1_sub = DSVector.from_focal(frame, {"": 0.1, "a": 0.4, "a,h,r": 0.5}, complete=False)
m2_sub = DSVector.from_focal(frame, {"": 0.2, "h": 0.3, "a,h,r": 0.5}, complete=False)
from evtools.combinations import drc
m12_d = drc(m1_sub, m2_sub)
m1_recovered_d = decombine_drc(m12_d, m2_sub)
ok_d = m1_recovered_d.is_valid and np.allclose(m1_recovered_d.dense, m1_sub.dense, atol=1e-6)
print(f"  drc(m1, m2) then decombine_drc(m12, m2) recovers m1: {GREEN}✓ OK{R}" if ok_d else f"  {RED}✗ MISMATCH{R}")
print(m1_recovered_d)
