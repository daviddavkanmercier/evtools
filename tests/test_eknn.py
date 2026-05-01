"""
Tests for evtools.classifiers.eknn — EkNN classifier (Denoeux 1995,
Zouhal & Denoeux 1998).
"""

import numpy as np
import pytest

from evtools.classifiers import EkNN
from evtools.dsvector import DSVector


# ---------------------------------------------------------------------------
# Synthetic dataset: 3 well-separated 2D Gaussian classes
# ---------------------------------------------------------------------------

def _make_synth(seed: int = 0, n_per_class: int = 30):
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 3.0]])
    X = np.vstack([rng.normal(loc=mu, scale=0.5, size=(n_per_class, 2)) for mu in means])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


X_TRAIN, Y_TRAIN = _make_synth(seed=0, n_per_class=30)
X_TEST,  Y_TEST  = _make_synth(seed=1, n_per_class=10)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_default_construction(self):
        clf = EkNN()
        assert clf.k == 5
        assert clf.alpha == 0.95
        assert clf.lambda_ == 1.0
        assert clf.optimize is True
        assert clf.method == "trf"

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            EkNN(alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            EkNN(alpha=0.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k"):
            EkNN(k=0)


# ---------------------------------------------------------------------------
# fit / predict — without optimization
# ---------------------------------------------------------------------------

class TestFitNoOptim:

    def test_fit_sets_attributes(self):
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        assert hasattr(clf, "gamma_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_classes_")
        assert clf.n_classes_ == 3
        assert clf.gamma_.shape == (3,)
        assert (clf.gamma_ > 0).all()

    def test_init_gamma_matches_heuristic(self):
        # γ_q = 1 / sqrt(mean intra-class distance) — sanity check
        from scipy.spatial.distance import pdist
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        for q in range(3):
            mask = (Y_TRAIN == q)
            mean_d = pdist(X_TRAIN[mask]).mean()
            expected = 1.0 / np.sqrt(mean_d)
            assert abs(clf.gamma_[q] - expected) < 1e-10

    def test_high_accuracy_on_separable_data(self):
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        acc = (clf.predict(X_TEST) == Y_TEST).mean()
        assert acc > 0.9, f"expected high accuracy, got {acc}"

    def test_predict_returns_original_labels(self):
        # Use string labels — predict should return the same strings.
        y_str = np.array(["a", "b", "c"])[Y_TRAIN]
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, y_str)
        pred = clf.predict(X_TEST)
        assert pred.dtype.kind in ("U", "O")
        assert set(pred).issubset({"a", "b", "c"})


# ---------------------------------------------------------------------------
# fit / predict — with optimization
# ---------------------------------------------------------------------------

class TestFitOptim:

    def test_optim_trf_runs(self):
        clf = EkNN(k=5, optimize=True, method="trf").fit(X_TRAIN, Y_TRAIN)
        assert clf.gamma_.shape == (3,)
        assert (clf.gamma_ > 0).all()

    def test_optim_lbfgsb_runs(self):
        clf = EkNN(k=5, optimize=True, method="l-bfgs-b").fit(X_TRAIN, Y_TRAIN)
        assert clf.gamma_.shape == (3,)
        assert (clf.gamma_ > 0).all()

    def test_optim_reduces_training_loss(self):
        # The optimized γ should achieve a lower MSE than the initialization.
        from evtools.classifiers.eknn import _knn, _residuals_and_jacobian
        from scipy.spatial.distance import pdist

        # Compute training residuals before and after optimization
        knn_idx, knn_sqd = _knn(X_TRAIN, X_TRAIN, 5, exclude_self=True)
        T = np.zeros((3, len(Y_TRAIN)))
        T[Y_TRAIN, np.arange(len(Y_TRAIN))] = 1.0

        # Heuristic init
        clf_init = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        res_init, _ = _residuals_and_jacobian(
            clf_init.gamma_, clf_init.alpha, clf_init.lambda_,
            knn_idx, knn_sqd, Y_TRAIN, 3, T, return_jac=False)
        loss_init = 0.5 * (res_init ** 2).sum()

        # Optimized
        clf_opt = EkNN(k=5, optimize=True, method="trf").fit(X_TRAIN, Y_TRAIN)
        res_opt, _ = _residuals_and_jacobian(
            clf_opt.gamma_, clf_opt.alpha, clf_opt.lambda_,
            knn_idx, knn_sqd, Y_TRAIN, 3, T, return_jac=False)
        loss_opt = 0.5 * (res_opt ** 2).sum()

        assert loss_opt < loss_init - 1e-9

    def test_trf_and_lbfgsb_converge_close(self):
        clf_trf   = EkNN(k=5, optimize=True, method="trf").fit(X_TRAIN, Y_TRAIN)
        clf_lbfgs = EkNN(k=5, optimize=True, method="l-bfgs-b").fit(X_TRAIN, Y_TRAIN)
        # Both methods should converge to roughly the same minimum (cost-wise).
        # The γ values may differ on flat directions, so we compare predictions.
        pred_trf   = clf_trf.predict(X_TEST)
        pred_lbfgs = clf_lbfgs.predict(X_TEST)
        # On well-separated data, both should agree on most predictions.
        agreement = (pred_trf == pred_lbfgs).mean()
        assert agreement > 0.9

    def test_high_accuracy_with_optim(self):
        clf = EkNN(k=5, optimize=True).fit(X_TRAIN, Y_TRAIN)
        acc = (clf.predict(X_TEST) == Y_TEST).mean()
        assert acc > 0.9

    def test_unknown_method_raises(self):
        clf = EkNN(k=5, optimize=True, method="bogus")
        with pytest.raises(ValueError, match="method"):
            clf.fit(X_TRAIN, Y_TRAIN)


# ---------------------------------------------------------------------------
# predict_bba — DSVector outputs
# ---------------------------------------------------------------------------

class TestPredictBba:

    def test_returns_list_of_dsvectors(self):
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        bbas = clf.predict_bba(X_TEST)
        assert isinstance(bbas, list)
        assert len(bbas) == len(X_TEST)
        assert all(isinstance(m, DSVector) for m in bbas)

    def test_each_bba_is_valid(self):
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        for m in clf.predict_bba(X_TEST):
            assert m.is_valid, f"invalid BBA: {m.sparse}"

    def test_frame_matches_classes(self):
        y_str = np.array(["a", "b", "c"])[Y_TRAIN]
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, y_str)
        bbas = clf.predict_bba(X_TEST)
        for m in bbas:
            assert set(m.frame) == {"a", "b", "c"}

    def test_predict_consistent_with_predict_bba(self):
        # The label predicted by predict() should be the argmax over class
        # singletons in the BBA.
        clf = EkNN(k=5, optimize=False).fit(X_TRAIN, Y_TRAIN)
        bbas = clf.predict_bba(X_TEST)
        labels = clf.predict(X_TEST)
        for m, label in zip(bbas, labels):
            # Build singleton masses array, in class order
            mass_per_class = np.array([
                m[frozenset({str(c)})] for c in clf.classes_
            ])
            argmax_label = clf.classes_[np.argmax(mass_per_class)]
            assert argmax_label == label


# ---------------------------------------------------------------------------
# Pre-fit guards
# ---------------------------------------------------------------------------

class TestPreFitGuards:

    def test_predict_before_fit_raises(self):
        clf = EkNN()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(X_TEST)

    def test_predict_bba_before_fit_raises(self):
        clf = EkNN()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_bba(X_TEST)

    def test_too_few_classes_raises(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 0])
        with pytest.raises(ValueError, match="classes"):
            EkNN(k=1).fit(X, y)

    def test_k_too_large_raises(self):
        with pytest.raises(ValueError, match="k="):
            EkNN(k=100).fit(X_TRAIN, Y_TRAIN)


# ---------------------------------------------------------------------------
# λ values — different cost functions
# ---------------------------------------------------------------------------

class TestLambdaValues:

    @pytest.mark.parametrize("lambda_", [0.0, 1.0/3.0, 1.0])
    def test_three_lambda_values_run(self, lambda_):
        clf = EkNN(k=5, optimize=True, lambda_=lambda_).fit(X_TRAIN, Y_TRAIN)
        assert clf.gamma_.shape == (3,)
        # Should still classify reasonably on well-separated data
        acc = (clf.predict(X_TEST) == Y_TEST).mean()
        assert acc > 0.85

    def test_default_lambda_is_pl_loss(self):
        clf = EkNN()
        assert clf.lambda_ == 1.0


# ---------------------------------------------------------------------------
# Sanity numerical: gradient consistency between TRF and L-BFGS-B
# ---------------------------------------------------------------------------

def test_jacobian_matches_finite_difference():
    """
    Verify the analytical Jacobian matches a finite-difference estimate
    on a small dataset. This catches algebra mistakes in the gradient.
    """
    from evtools.classifiers.eknn import _knn, _residuals_and_jacobian

    X, y = _make_synth(seed=42, n_per_class=10)
    M = 3
    knn_idx, knn_sqd = _knn(X, X, K=4, exclude_self=True)
    T = np.zeros((M, len(y)))
    T[y, np.arange(len(y))] = 1.0

    rng = np.random.default_rng(42)
    gamma = rng.uniform(0.5, 2.0, size=M)

    res0, J = _residuals_and_jacobian(
        gamma, alpha=0.95, lambda_=1.0,
        knn_idx=knn_idx, knn_sqdist=knn_sqd,
        y_train_int=y, M=M, T=T, return_jac=True)

    # Finite differences
    eps = 1e-6
    J_fd = np.zeros_like(J)
    for q in range(M):
        gp = gamma.copy(); gp[q] += eps
        gm = gamma.copy(); gm[q] -= eps
        rp, _ = _residuals_and_jacobian(gp, 0.95, 1.0, knn_idx, knn_sqd, y, M, T, return_jac=False)
        rm, _ = _residuals_and_jacobian(gm, 0.95, 1.0, knn_idx, knn_sqd, y, M, T, return_jac=False)
        J_fd[:, q] = (rp - rm) / (2 * eps)

    # Tolerance scaled by magnitude
    np.testing.assert_allclose(J, J_fd, rtol=1e-4, atol=1e-7)
