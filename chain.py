"""Simulation library for the compositional GP prior chain studied in the paper.

Two chains are simulated:

  Scalar pairwise-distance chain  (used by the trajectory and stationary-law
  figures): tracks $L_i := \\log v_i^2$ for $v_i = \\|Z_{i,1} - Z_{i,2}\\|$ in
  $\\mathbb R^d$, with the analytic step

      L_{i+1} = log 2 + log(1 - exp(- e^{L_i} / (2 r^2))) + log X_i,
      X_i ~ chi^2_d.

  Position chain (used by the per-chain visualisation figures): tracks the
  full $\\bar Z_i \\in (\\mathbb R^d)^n$ via Cholesky factorisation of the
  $n \\times n$ RBF correlation matrix $R(\\bar Z_i)$ at each step.

Critical bandwidth: $r_c(d) = \\sqrt 2 \\, e^{\\psi(d/2)/2}$, with the chain
reparametrised as $r = \\lambda \\, r_c(d)$ throughout. Subcritical: $\\lambda
< 1$. Supercritical: $\\lambda > 1$.

Numerical care for the scalar chain: $1 - \\exp(-x)$ underflows to 0 for
moderate $x$, which would spuriously absorb the chain at $v = 0$. We work
in log-space ($L_i$) and use $\\log(-\\mathrm{expm1}(-x))$ for moderate $x$
plus the linearisation $\\log x$ once $x$ underflows.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma


# ---------------------------------------------------------------------------
# Threshold and helpers.
# ---------------------------------------------------------------------------


def r_c_of_d(d: int) -> float:
    """Critical radius $r_c(d) = \\sqrt{2}\\, e^{\\psi(d/2)/2}$."""
    return float(np.sqrt(2.0) * np.exp(digamma(d / 2.0) / 2.0))


def mu_of(d: int, r: float) -> float:
    """Asymptotic supercritical decay rate $\\mu_d(r) = 2\\log r - \\psi(d/2) - \\log 2$.

    Equals $2 \\log \\lambda$ at $r = \\lambda \\, r_c(d)$.
    """
    return float(2.0 * np.log(r) - digamma(d / 2.0) - np.log(2.0))


def jensen_L_star(r: float, d: int) -> float:
    """Jensen upper bound $L_*(r, d)$ on $\\mathbb E_\\pi[\\log v^2]$ in the
    subcritical regime ($r < r_c(d)$).
    """
    rc = r_c_of_d(d)
    rhs = (r / rc) ** 2
    if not 0 < rhs < 1:
        raise ValueError(f"jensen_L_star requires 0 < r < r_c(d); got r={r}, r_c={rc}")
    alpha = brentq(lambda a: (1.0 - np.exp(-a)) / a - rhs, 1e-12, 1e6)
    return float(np.log(2.0 * r**2 * alpha))


# ---------------------------------------------------------------------------
# Scalar pairwise-distance chain (log-space).
# ---------------------------------------------------------------------------


def _log_one_minus_exp_neg(x: np.ndarray) -> np.ndarray:
    """Stable $\\log(1 - e^{-x})$ for $x \\ge 0$, returning $-\\infty$ at $x = 0$."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, -np.inf)
    nz = x > 0
    out[nz] = np.log(-np.expm1(-x[nz]))
    return out


def step_log(
    L: np.ndarray, r: float, rng: np.random.Generator, d: int = 1
) -> np.ndarray:
    """Advance $L_i = \\log v_i^2$ by one step. Vectorised over $L$."""
    L = np.asarray(L, dtype=float)
    log_2r2 = np.log(2.0 * r**2)
    x = np.exp(L - log_2r2)
    log_1m_phi = _log_one_minus_exp_neg(x)
    underflowed = ~np.isfinite(log_1m_phi)
    log_1m_phi = np.where(underflowed, L - log_2r2, log_1m_phi)

    size = L.shape if L.ndim > 0 else ()
    if d == 1:
        g = rng.standard_normal(size=size)
        log_X = 2.0 * np.log(np.abs(g))
    else:
        X = rng.chisquare(d, size=size)
        log_X = np.log(X)
    return np.log(2.0) + log_1m_phi + log_X


def simulate_L_trajectories(
    r: float,
    n_steps: int,
    n_copies: int,
    v1: float,
    rng: np.random.Generator,
    d: int = 1,
) -> np.ndarray:
    """Return array of shape ``(n_steps + 1, n_copies)`` of $L_i = \\log v_i^2$ values."""
    L = np.full(n_copies, np.log(v1**2))
    out = np.empty((n_steps + 1, n_copies))
    out[0] = L
    for i in range(n_steps):
        L = step_log(L, r, rng, d=d)
        out[i + 1] = L
    return out


def simulate_L_long(
    r: float,
    n_steps: int,
    v1: float,
    rng: np.random.Generator,
    d: int = 1,
) -> np.ndarray:
    """Single long chain; return array of $L_i$ for $i = 0, \\dots, n_{\\text{steps}}$."""
    L = np.array(np.log(v1**2))
    out = np.empty(n_steps + 1)
    out[0] = L
    for i in range(n_steps):
        L = step_log(L.reshape(()), r, rng, d=d).reshape(())
        out[i + 1] = L
    return out


# ---------------------------------------------------------------------------
# Position chain $\bar Z_i \in (\mathbb R^d)^n$ (used for per-chain visualisations).
# ---------------------------------------------------------------------------


def simulate_Z_chain(
    r: float,
    n: int,
    d: int,
    n_steps: int,
    rng: np.random.Generator,
    Z0: np.ndarray | None = None,
    save_depths: tuple[int, ...] | None = None,
    jitter: float = 1e-8,
) -> dict[int, np.ndarray] | np.ndarray:
    """Run the position chain for ``n_steps`` depths.

    If ``save_depths`` is None, returns the full path of shape
    ``(n_steps + 1, n, d)``. If ``save_depths`` is a tuple of ints,
    returns ``{depth: Z}`` for each requested depth (memory-cheap for
    large ``n_steps``).
    """
    if Z0 is None:
        Z = rng.standard_normal(size=(n, d))
    else:
        Z = np.asarray(Z0, dtype=float).reshape(n, d)
    inv_two_r2 = 1.0 / (2.0 * r * r)
    eye_n = np.eye(n)

    if save_depths is None:
        out = np.empty((n_steps + 1, n, d))
        out[0] = Z
        for i in range(n_steps):
            sq = np.sum(Z * Z, axis=1)
            D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
            np.maximum(D2, 0.0, out=D2)
            R = np.exp(-D2 * inv_two_r2)
            R = 0.5 * (R + R.T) + jitter * eye_n
            L = np.linalg.cholesky(R)
            W = rng.standard_normal(size=(n, d))
            Z = L @ W
            out[i + 1] = Z
        return out

    save_set = set(save_depths)
    snaps: dict[int, np.ndarray] = {}
    if 0 in save_set:
        snaps[0] = Z.copy()
    for i in range(n_steps):
        sq = np.sum(Z * Z, axis=1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        R = np.exp(-D2 * inv_two_r2)
        R = 0.5 * (R + R.T) + jitter * eye_n
        L = np.linalg.cholesky(R)
        W = rng.standard_normal(size=(n, d))
        Z = L @ W
        if (i + 1) in save_set:
            snaps[i + 1] = Z.copy()
    return snaps
