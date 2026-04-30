"""
Tests for evtools.learning — fit_cd, fit_cr, fit_cn closed-form solutions.

The main numerical fixture is Pichon et al. (2016) Table 4 (source data,
sensors 1 & 2 with 4 objects each) → Table 6 (optimal β and E_pl values).
"""

import numpy as np
import pytest

from evtools.dsvector import DSVector
from evtools.corrections import (
    contextual_discount, contextual_reinforce, contextual_negate,
)
from evtools.metrics import pl_loss
from evtools.learning import fit_cd, fit_cr, fit_cn


FRAME = ["a", "h", "r"]


# ---------------------------------------------------------------------------
# Pichon 2016 Table 4 — two sensors, 4 objects each
# ---------------------------------------------------------------------------

# Sensor 1: 4 BBAs over {a, h, r}
SENSOR_1 = [
    DSVector.from_focal(FRAME, {"r": 0.5, "h,r": 0.3, "a,h,r": 0.2}),  # truth = a
    DSVector.from_focal(FRAME, {"h": 0.5, "r": 0.2, "a,h,r": 0.3}),    # truth = h
    DSVector.from_focal(FRAME, {"h": 0.4, "a,r": 0.6}),                # truth = a
    DSVector.from_focal(FRAME, {"a,r": 0.6, "h,r": 0.4}),              # truth = r
]
LABELS_1 = ["a", "h", "a", "r"]

# Sensor 2: 4 BBAs over {a, h, r}
SENSOR_2 = [
    DSVector.from_focal(FRAME, {"a,h": 0.7, "a,h,r": 0.3}),            # truth = a
    DSVector.from_focal(FRAME, {"a": 0.3, "a,h": 0.4, "a,h,r": 0.3}),  # truth = h
    DSVector.from_focal(FRAME, {"a": 0.2, "h,r": 0.6, "a,h,r": 0.2}),  # truth = a
    DSVector.from_focal(FRAME, {"h,r": 1.0}),                           # truth = r
]
LABELS_2 = ["a", "h", "a", "r"]


# Expected β and E_pl values from Pichon 2016 Table 6 (rounded to 2 decimals).
EXPECTED_S1 = {
    "cd": {"betas": (0.76, 1.00, 1.00), "epl": 3.39},
    "cr": {"betas": (0.94, 0.66, 0.38), "epl": 2.33},
    "cn": {"betas": (0.33, 1.00, 0.45), "epl": 2.59},
}
EXPECTED_S2 = {
    "cd": {"betas": (0.74, 1.00, 1.00), "epl": 4.81},
    "cr": {"betas": (0.65, 0.22, 0.55), "epl": 2.39},
    "cn": {"betas": (0.63, 0.06, 0.86), "epl": 2.25},
}


def _betas_singletons_to_tuple(betas: dict, frame: list) -> tuple:
    """Convert {frozenset({a}): β_a, ...} to (β_a, β_h, β_r) tuple (CD)."""
    return tuple(betas[frozenset({x})] for x in frame)


def _betas_complements_to_tuple(betas: dict, frame: list) -> tuple:
    """Convert {complement-of-{a}: β_a, ...} to (β_a, β_h, β_r) tuple (CR/CN)."""
    omega = frozenset(frame)
    return tuple(betas[omega - {x}] for x in frame)


# ---------------------------------------------------------------------------
# Pichon 2016 Table 6 reproduction (sensor 1)
# ---------------------------------------------------------------------------

class TestPichonTable6Sensor1:

    def test_cd_betas(self):
        betas = fit_cd(SENSOR_1, LABELS_1)
        np.testing.assert_allclose(_betas_singletons_to_tuple(betas, FRAME),
                                    EXPECTED_S1["cd"]["betas"], atol=0.01)

    def test_cd_epl(self):
        betas = fit_cd(SENSOR_1, LABELS_1)
        corrected = [contextual_discount(m, betas) for m in SENSOR_1]
        loss = pl_loss(corrected, LABELS_1)
        assert loss == pytest.approx(EXPECTED_S1["cd"]["epl"], abs=0.01)

    def test_cr_betas(self):
        betas = fit_cr(SENSOR_1, LABELS_1)
        np.testing.assert_allclose(_betas_complements_to_tuple(betas, FRAME),
                                    EXPECTED_S1["cr"]["betas"], atol=0.01)

    def test_cr_epl(self):
        betas = fit_cr(SENSOR_1, LABELS_1)
        corrected = [contextual_reinforce(m, betas) for m in SENSOR_1]
        loss = pl_loss(corrected, LABELS_1)
        assert loss == pytest.approx(EXPECTED_S1["cr"]["epl"], abs=0.01)

    def test_cn_betas(self):
        betas = fit_cn(SENSOR_1, LABELS_1)
        np.testing.assert_allclose(_betas_complements_to_tuple(betas, FRAME),
                                    EXPECTED_S1["cn"]["betas"], atol=0.01)

    def test_cn_epl(self):
        betas = fit_cn(SENSOR_1, LABELS_1)
        corrected = [contextual_negate(m, betas) for m in SENSOR_1]
        loss = pl_loss(corrected, LABELS_1)
        assert loss == pytest.approx(EXPECTED_S1["cn"]["epl"], abs=0.01)


# ---------------------------------------------------------------------------
# Pichon 2016 Table 6 reproduction (sensor 2)
# ---------------------------------------------------------------------------

class TestPichonTable6Sensor2:

    def test_cd_betas(self):
        betas = fit_cd(SENSOR_2, LABELS_2)
        np.testing.assert_allclose(_betas_singletons_to_tuple(betas, FRAME),
                                    EXPECTED_S2["cd"]["betas"], atol=0.01)

    def test_cd_epl(self):
        betas = fit_cd(SENSOR_2, LABELS_2)
        corrected = [contextual_discount(m, betas) for m in SENSOR_2]
        loss = pl_loss(corrected, LABELS_2)
        assert loss == pytest.approx(EXPECTED_S2["cd"]["epl"], abs=0.01)

    def test_cr_betas(self):
        betas = fit_cr(SENSOR_2, LABELS_2)
        np.testing.assert_allclose(_betas_complements_to_tuple(betas, FRAME),
                                    EXPECTED_S2["cr"]["betas"], atol=0.01)

    def test_cr_epl(self):
        betas = fit_cr(SENSOR_2, LABELS_2)
        corrected = [contextual_reinforce(m, betas) for m in SENSOR_2]
        loss = pl_loss(corrected, LABELS_2)
        assert loss == pytest.approx(EXPECTED_S2["cr"]["epl"], abs=0.01)

    def test_cn_betas(self):
        betas = fit_cn(SENSOR_2, LABELS_2)
        np.testing.assert_allclose(_betas_complements_to_tuple(betas, FRAME),
                                    EXPECTED_S2["cn"]["betas"], atol=0.01)

    def test_cn_epl(self):
        betas = fit_cn(SENSOR_2, LABELS_2)
        corrected = [contextual_negate(m, betas) for m in SENSOR_2]
        loss = pl_loss(corrected, LABELS_2)
        assert loss == pytest.approx(EXPECTED_S2["cn"]["epl"], abs=0.01)


# ---------------------------------------------------------------------------
# Sanity properties (any dataset, any fitter)
# ---------------------------------------------------------------------------

class TestSanity:

    def test_fit_cd_keys_are_singletons(self):
        betas = fit_cd(SENSOR_1, LABELS_1)
        assert set(betas.keys()) == {frozenset({a}) for a in FRAME}

    @pytest.mark.parametrize("fit_fn", [fit_cr, fit_cn])
    def test_fit_cr_cn_keys_are_complements(self, fit_fn):
        betas = fit_fn(SENSOR_1, LABELS_1)
        omega = frozenset(FRAME)
        assert set(betas.keys()) == {omega - {a} for a in FRAME}

    @pytest.mark.parametrize("fit_fn", [fit_cd, fit_cr, fit_cn])
    def test_betas_in_unit_interval(self, fit_fn):
        betas = fit_fn(SENSOR_1, LABELS_1)
        for b in betas.values():
            assert 0.0 <= b <= 1.0

    def test_perfect_predictions_fit_identity_cd(self):
        # If predictions are already categorical on the truth, β=1 (no change).
        preds = [DSVector.from_focal(FRAME, {y: 1.0}) for y in LABELS_1]
        betas = fit_cd(preds, LABELS_1)
        assert all(b == pytest.approx(1.0) for b in betas.values())

    def test_perfect_predictions_fit_identity_cr(self):
        preds = [DSVector.from_focal(FRAME, {y: 1.0}) for y in LABELS_1]
        betas = fit_cr(preds, LABELS_1)
        assert all(b == pytest.approx(1.0) for b in betas.values())

    def test_lengths_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            fit_cd(SENSOR_1, LABELS_1[:2])

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fit_cd([], [])

    def test_invalid_label_type_raises(self):
        with pytest.raises(TypeError):
            fit_cd(SENSOR_1, [123, "h", "a", "r"])

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="not in frame"):
            fit_cd(SENSOR_1, ["a", "h", "X", "r"])  # "X" not in frame

    def test_frame_mismatch_raises(self):
        m_other = DSVector.from_focal(["x", "y", "z"], {"x": 1.0})
        with pytest.raises(ValueError, match="frame"):
            fit_cd([SENSOR_1[0], m_other, SENSOR_1[2], SENSOR_1[3]], LABELS_1)


# ---------------------------------------------------------------------------
# Soft labels (Mutmainah 2021, Section 5.2 — Ẽ_pl)
# ---------------------------------------------------------------------------

class TestSoftLabels:

    def test_categorical_soft_label_equals_hard(self):
        # A soft label that is categorical on {y} must yield the same fit
        # as the hard label "y".
        preds  = SENSOR_1
        hards  = LABELS_1
        softs  = [DSVector.from_focal(FRAME, {y: 1.0}) for y in hards]
        # Compare value-by-value (independent of singleton-vs-complement keys).
        for fit_fn in (fit_cd, fit_cr, fit_cn):
            b_hard = sorted(fit_fn(preds, hards).values())
            b_soft = sorted(fit_fn(preds, softs).values())
            np.testing.assert_allclose(b_hard, b_soft, atol=1e-12)

    def test_mixed_hard_and_soft_runs(self):
        # Mixing hard and soft labels should work and produce β ∈ [0, 1].
        soft_a = DSVector.from_focal(FRAME, {"a": 0.7, "a,h": 0.3})
        labels_mixed = ["a", soft_a, "a", "r"]
        betas = fit_cd(SENSOR_1, labels_mixed)
        for b in betas.values():
            assert 0.0 <= b <= 1.0
