"""
Evidential k-nearest neighbor (EkNN) classifier.

For a test instance x_s, each of its K nearest neighbors x_i in the training
set produces a simple basic belief assignment (BBA) on the frame of classes
Ω = {C_1, ..., C_M}:

    m_{s,i}({C_q}) = α · exp(-γ_q² · d_{s,i}²)
    m_{s,i}(Ω)    = 1 - m_{s,i}({C_q})

where C_q is the class of x_i and d_{s,i} is its (Euclidean) distance to x_s.
The K BBAs are combined with Dempster's rule, yielding a single BBA on Ω
that summarizes the evidence about the class of x_s.

The per-class parameters γ_q can be either:
- initialized heuristically (γ_q = 1 / sqrt(mean intra-class distance), Denoeux 1995); or
- optimized to minimize a pl-based discrepancy on the training set (Zouhal &
  Denoeux 1998). The cost is a generalized squared error

      L(γ) = ½ · mean_n( ‖T_λ(m_n) - one_hot(y_n)‖² )

  where T_λ is the parametric transform

      T_λ({C_c}) = m({C_c}) + λ · m(Ω)

  Since EkNN produces BBAs whose only focal elements are singletons and Ω,
  the well-known special cases are:
    - λ = 0    → T_0({c})   = m({c})           = Bel({c})  → Bel-loss
    - λ = 1/M  → T_{1/M}({c}) = m({c}) + m(Ω)/M = BetP({c}) → BetP-loss
                 (default in Zouhal & Denoeux 1998)
    - λ = 1    → T_1({c})   = m({c}) + m(Ω)    = Pl({c})   → Pl-loss
                 (default in evtools)

Optimization is delegated to SciPy:
    - method="trf"      : scipy.optimize.least_squares (Trust Region Reflective)
                          with analytical Jacobian — recommended.
    - method="l-bfgs-b" : scipy.optimize.minimize with L-BFGS-B and
                          analytical gradient — alternative.

References
----------
- Denoeux, T. (1995). A k-nearest neighbor classification rule based on
  Dempster-Shafer theory. IEEE Transactions on Systems, Man and Cybernetics,
  25(5), 804-813.
- Zouhal, L. M., Denoeux, T. (1998). An evidence-theoretic k-NN rule with
  parameter optimization. IEEE Transactions on Systems, Man and Cybernetics
  Part C, 28(2), 263-271.
"""

from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, least_squares

from ..dsvector import DSVector
from ..constants import ZERO_MASS


# ---------------------------------------------------------------------------
# Default constants (private — overridable via fit() kwargs)
# ---------------------------------------------------------------------------

_GAMMA_MIN_DEFAULT: float = 1e-4   # lower bound for γ_q during optimization
_ALPHA_DEFAULT:     float = 0.95   # sensible default for α (Denoeux 1995, p. 808)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_gamma(X: np.ndarray, y_int: np.ndarray, M: int) -> np.ndarray:
    """
    Heuristic initialization of γ_q for each class q (Denoeux 1995).

        γ_q = 1 / sqrt(mean_{i,j ∈ class q, i<j} ‖x_i - x_j‖)

    Returns
    -------
    np.ndarray, shape (M,)
        Initial γ values (one per class). Defaults to 1.0 if a class has
        fewer than 2 training samples.
    """
    gamma = np.ones(M)
    for q in range(M):
        mask = (y_int == q)
        if mask.sum() < 2:
            continue   # fallback: γ_q = 1.0
        D = pdist(X[mask])
        mean_d = D.mean()
        if mean_d > 0:
            gamma[q] = 1.0 / np.sqrt(mean_d)
    return gamma


def _knn(X_train: np.ndarray, X_query: np.ndarray, K: int,
         exclude_self: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the K nearest neighbors of each query in X_train.

    If ``exclude_self`` is True (typical for leave-one-out on training data),
    the query points are assumed to be the same as ``X_train``, the very
    first neighbor of each (= itself, distance 0) is dropped, and K+1
    neighbors are queried internally.

    Returns
    -------
    indices : np.ndarray, shape (N_query, K)
    sq_dist : np.ndarray, shape (N_query, K)
    """
    tree = KDTree(X_train)
    if exclude_self:
        dist, idx = tree.query(X_query, k=K + 1)
        return idx[:, 1:], dist[:, 1:] ** 2
    dist, idx = tree.query(X_query, k=K)
    return idx, dist ** 2


def _forward(
    gamma:        np.ndarray,
    alpha:        float,
    knn_idx:      np.ndarray,
    knn_sqdist:   np.ndarray,
    y_train_int:  np.ndarray,
    M:            int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Forward pass: for each query, build the unnormalized BBA mk by
    Dempster-combining the K simple BBAs from the neighbors.

    Returns
    -------
    mkn : (M+1, N)  normalized BBA — rows 0..M-1 = class masses, row M = m(Ω)
    mk  : (M+1, N)  unnormalized BBA before division by Kn
    Kn  : (N,)      normalization constants Σ_c mk + mk_Ω
    s   : (K, N)    s_{k,n} = α · exp(-γ_{q_k}² · d_{k,n}²)
    Tk_per_k : list of K arrays of shape (M, N)
        Tk[k][c, n] = 1 if the class of the k-th neighbor of query n is c.
    """
    K = knn_idx.shape[1]
    N = knn_idx.shape[0]

    mk_classes = np.zeros((M, N))
    mk_omega   = np.ones(N)

    s = np.zeros((K, N))
    Tk_per_k: list[np.ndarray] = []

    for k in range(K):
        nbr_classes = y_train_int[knn_idx[:, k]]                  # (N,)
        gamma_sq_k  = gamma[nbr_classes] ** 2                     # (N,)
        s_k         = alpha * np.exp(-gamma_sq_k * knn_sqdist[:, k])
        s[k, :]     = s_k

        Tk = np.zeros((M, N))
        Tk[nbr_classes, np.arange(N)] = 1.0
        Tk_per_k.append(Tk)

        m_k_classes = Tk * s_k                                    # (M, N)
        m_k_omega   = 1.0 - s_k                                   # (N,)

        # Dempster combination (no intermediate normalization, à la
        # gradientds.R, simpler for the gradient).
        new_mk_classes = (
            mk_classes * (m_k_classes + m_k_omega[None, :])
            + m_k_classes * mk_omega[None, :]
        )
        new_mk_omega = mk_omega * m_k_omega
        mk_classes, mk_omega = new_mk_classes, new_mk_omega

    mk = np.vstack([mk_classes, mk_omega[None, :]])
    Kn = mk.sum(axis=0)
    mkn = mk / Kn[None, :]
    return mkn, mk, Kn, s, Tk_per_k


def _residuals_and_jacobian(
    gamma:        np.ndarray,
    alpha:        float,
    lambda_:      float,
    knn_idx:      np.ndarray,
    knn_sqdist:   np.ndarray,
    y_train_int:  np.ndarray,
    M:            int,
    T:            np.ndarray,
    return_jac:   bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute residuals Q (flattened) and optionally the Jacobian dQ/dγ.

    The cost minimized by least_squares is ½ · ‖residuals‖², equivalent to
    ½ · Σ_n ‖Q[:, n]‖² where Q[c, n] = mkn[c, n] + λ·mkn[Ω, n] - T[c, n].

    Parameters
    ----------
    T : (M, N)  one-hot target (T[c, n] = 1 iff class of n is c).

    Returns
    -------
    residuals : (M·N,)
    jacobian  : (M·N, M)  dQ/dγ if ``return_jac`` else None.
    """
    mkn, mk, Kn, s, Tk_per_k = _forward(
        gamma, alpha, knn_idx, knn_sqdist, y_train_int, M)

    Q = mkn[:M, :] + lambda_ * mkn[M, :][None, :] - T            # (M, N)

    # residuals[n*M + c] = Q[c, n]
    residuals = Q.T.ravel()

    if not return_jac:
        return residuals, None

    K = knn_idx.shape[1]
    N = knn_idx.shape[0]

    # J3D[c, n, q] = dQ[c, n] / dγ_q
    J3D = np.zeros((M, N, M))

    # Avoid div-by-zero — γ ≥ gamma_min ensures s_k ≤ α < 1, so m_k_omega ≥ 1-α > 0.
    eps = ZERO_MASS

    for k in range(K):
        Tk = Tk_per_k[k]
        nbr_classes = y_train_int[knn_idx[:, k]]
        d_sq = knn_sqdist[:, k]
        s_k  = s[k, :]

        m_k_classes = Tk * s_k
        m_k_omega   = 1.0 - s_k

        # Rewind k-th BBA from mk: factor it out to expose ∂mk/∂s_k.
        m_k_omega_safe = np.where(m_k_omega > eps, m_k_omega, eps)
        mm_omega = mk[M, :] / m_k_omega_safe                    # (N,)
        denom = m_k_classes + m_k_omega[None, :]                # (M, N)
        denom_safe = np.where(denom > eps, denom, eps)
        mm_classes = (mk[:M, :] - mm_omega[None, :] * m_k_classes) / denom_safe

        v   = (mm_classes + mm_omega[None, :]) * Tk - mm_classes
        DsK = v.sum(axis=0) - mm_omega                          # (N,)

        Kn_safe = np.where(Kn > eps, Kn, eps)
        Kn_sq   = Kn_safe ** 2
        Dsm_classes = (Kn[None, :] * v - mk[:M, :] * DsK[None, :]) / Kn_sq[None, :]
        Dsm_omega   = (-Kn * mm_omega - mk[M, :] * DsK) / Kn_sq

        dQ_ds_k = Dsm_classes + lambda_ * Dsm_omega[None, :]    # (M, N)

        # ds_k/dγ_q = -2·γ_q·d²·s_k · 𝟙(q == nbr_classes[n])
        # → only γ[nbr_classes[n]] is affected for that n.
        gamma_per_n = gamma[nbr_classes]                        # (N,)
        update = dQ_ds_k * (-2.0 * d_sq * s_k * gamma_per_n)[None, :]   # (M, N)
        # Scatter into J3D along the q dimension.
        np.add.at(J3D, (slice(None), np.arange(N), nbr_classes), update)

    # J[n*M + c, q] = J3D[c, n, q]
    J = np.transpose(J3D, (1, 0, 2)).reshape(N * M, M)
    return residuals, J


def _optimize_gamma(
    gamma_init:   np.ndarray,
    alpha:        float,
    lambda_:      float,
    knn_idx:      np.ndarray,
    knn_sqdist:   np.ndarray,
    y_train_int:  np.ndarray,
    M:            int,
    T:            np.ndarray,
    method:       str,
    gamma_min:    float,
    max_iter:     int,
    verbose:      bool,
) -> np.ndarray:
    """Run the optimizer of choice and return the optimal γ vector."""
    args = (alpha, lambda_, knn_idx, knn_sqdist, y_train_int, M, T)

    if method == "trf":
        def fun(g):  return _residuals_and_jacobian(g, *args, return_jac=False)[0]
        def jac(g):  return _residuals_and_jacobian(g, *args, return_jac=True)[1]
        result = least_squares(
            fun=fun, x0=gamma_init, jac=jac,
            method="trf",
            bounds=([gamma_min] * M, [np.inf] * M),
            max_nfev=max_iter,
            verbose=2 if verbose else 0,
        )
        return result.x

    if method == "l-bfgs-b":
        def cost_and_grad(g):
            res, J = _residuals_and_jacobian(g, *args, return_jac=True)
            cost = 0.5 * float(res @ res)
            grad = J.T @ res
            return cost, grad
        result = minimize(
            fun=cost_and_grad, x0=gamma_init, jac=True,
            method="L-BFGS-B",
            bounds=[(gamma_min, None)] * M,
            options={"maxiter": max_iter, "disp": verbose},
        )
        return result.x

    raise ValueError(
        f"EkNN: unknown method '{method}'. Choose 'trf' or 'l-bfgs-b'."
    )


# ---------------------------------------------------------------------------
# EkNN class
# ---------------------------------------------------------------------------

class EkNN:
    """
    Evidential k-nearest neighbor classifier.

    Parameters
    ----------
    k : int, default 5
        Number of nearest neighbors used to compute the BBA for each query.
    alpha : float, default 0.95
        Maximum mass that any neighbor can allocate to its class. Must be
        in (0, 1) — α = 1 would lead to numerically unstable updates
        (m(Ω) = 0).
    lambda_ : float, default 1.0
        Coefficient of the m(Ω) term in the cost transform
            T_λ({c}) = m({c}) + λ · m(Ω).
        Special cases (when the BBA has only singletons and Ω as focals,
        which is the case for EkNN outputs):
            λ = 0    → T_0   = Bel(c),  cost = Bel-loss
            λ = 1/M  → T_1/M = BetP(c), cost = BetP-loss (Zouhal 1998 default)
            λ = 1    → T_1   = Pl(c),   cost = Pl-loss   (evtools default)
    optimize : bool, default True
        If True (default), γ_q are optimized on the training set
        (Zouhal & Denoeux 1998). If False, the heuristic initialization of
        Denoeux 1995 is kept.
    method : {"trf", "l-bfgs-b"}, default "trf"
        Optimization method (used only when ``optimize=True``):
            - "trf"      : scipy.optimize.least_squares (recommended,
                           exploits the least-squares structure).
            - "l-bfgs-b" : scipy.optimize.minimize with L-BFGS-B.
    max_iter : int, default 300
        Maximum number of optimizer iterations.
    gamma_min : float, default 1e-4
        Lower bound enforced on each γ_q during optimization.
    verbose : bool, default False
        If True, the optimizer prints progress.

    Attributes (set by ``fit``)
    ---------------------------
    classes_     : np.ndarray
        Unique class labels, in order they appear after factor-encoding.
    n_classes_   : int
        Number of classes M.
    X_train_     : np.ndarray
        Stored training features (used by ``predict`` to find neighbors).
    y_train_int_ : np.ndarray
        Training labels encoded as integers in [0, M).
    gamma_       : np.ndarray, shape (M,)
        Per-class γ values (after optional optimization).
    """

    def __init__(
        self,
        k:         int = 5,
        alpha:     float = _ALPHA_DEFAULT,
        lambda_:   float = 1.0,
        optimize:  bool = True,
        method:    Literal["trf", "l-bfgs-b"] = "trf",
        max_iter:  int = 300,
        gamma_min: float = _GAMMA_MIN_DEFAULT,
        verbose:   bool = False,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"EkNN: alpha must lie in (0, 1), got {alpha}.")
        if k < 1:
            raise ValueError(f"EkNN: k must be a positive integer, got {k}.")

        self.k         = k
        self.alpha     = alpha
        self.lambda_   = lambda_
        self.optimize  = optimize
        self.method    = method
        self.max_iter  = max_iter
        self.gamma_min = gamma_min
        self.verbose   = verbose

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Sequence) -> "EkNN":
        """Fit the classifier on training data ``(X, y)``."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.n_classes_      = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError(
                f"EkNN: need at least 2 classes, got {self.n_classes_}."
            )
        if self.k >= len(X):
            raise ValueError(
                f"EkNN: k={self.k} must be smaller than the training "
                f"size n={len(X)}."
            )

        self.X_train_     = X
        self.y_train_int_ = y_int

        gamma = _init_gamma(X, y_int, self.n_classes_)

        if self.optimize:
            knn_idx, knn_sqdist = _knn(X, X, self.k, exclude_self=True)
            T = np.zeros((self.n_classes_, len(X)))
            T[y_int, np.arange(len(X))] = 1.0
            gamma = _optimize_gamma(
                gamma_init  = gamma,
                alpha       = self.alpha,
                lambda_     = self.lambda_,
                knn_idx     = knn_idx,
                knn_sqdist  = knn_sqdist,
                y_train_int = y_int,
                M           = self.n_classes_,
                T           = T,
                method      = self.method,
                gamma_min   = self.gamma_min,
                max_iter    = self.max_iter,
                verbose     = self.verbose,
            )

        self.gamma_ = gamma
        return self

    def _bbas_dense(self, X_query: np.ndarray) -> np.ndarray:
        """Return the (M+1, N_query) array of normalized BBAs for the queries."""
        if not hasattr(self, "gamma_"):
            raise RuntimeError("EkNN: fit() must be called before predict().")
        X_query = np.asarray(X_query, dtype=float)
        knn_idx, knn_sqdist = _knn(self.X_train_, X_query, self.k, exclude_self=False)
        mkn, *_ = _forward(
            self.gamma_, self.alpha,
            knn_idx, knn_sqdist, self.y_train_int_, self.n_classes_,
        )
        return mkn

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels for the queries.

        Decision rule: for each query, pick the class with maximum mass on
        its singleton (equivalent to max BetP / max Pl on this BBA, since
        the only non-singleton focal element is Ω).
        """
        mkn = self._bbas_dense(X_query)
        idx = np.argmax(mkn[:self.n_classes_, :], axis=0)
        return self.classes_[idx]

    def predict_bba(self, X_query: np.ndarray) -> list[DSVector]:
        """
        Return one :class:`DSVector` BBA per query.

        Each BBA has the original class labels (cast to ``str``) as atom
        names. Focal elements are the singletons ``{C_q}`` (one per class
        with non-negligible mass) and ``Ω``.
        """
        mkn = self._bbas_dense(X_query)
        N_query = mkn.shape[1]
        frame = [str(c) for c in self.classes_]
        omega = frozenset(frame)

        bbas: list[DSVector] = []
        for n in range(N_query):
            sparse: dict[frozenset, float] = {}
            for q in range(self.n_classes_):
                if mkn[q, n] > ZERO_MASS:
                    sparse[frozenset({frame[q]})] = float(mkn[q, n])
            if mkn[self.n_classes_, n] > ZERO_MASS:
                sparse[omega] = float(mkn[self.n_classes_, n])
            bbas.append(DSVector.from_sparse(frame, sparse))
        return bbas
