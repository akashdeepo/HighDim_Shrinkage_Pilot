"""Data-generating processes for the high-dimensional shrinkage pilot.

Three return models, all with a common target population covariance `Sigma` so
comparisons across distributions are fair:
  - Gaussian
  - Multivariate Student-t (elliptical heavy tails), standardized to Cov = Sigma
  - GH skew-t (asymmetric heavy tails) via a normal mean-variance mixture

Conventions: returns are rows (each row = one period, length p = #assets).
"""
import numpy as np
from scipy import stats


def make_cov(p, model="ar1", rho=0.6, seed=None):
    """Return a p x p SPD covariance matrix.

    model='ar1' -> correlation rho^|i-j| (correlated assets => near-singular when p~n).
    Unit variances by default (a correlation matrix); good enough for the pilot.
    """
    if model == "ar1":
        idx = np.arange(p)
        C = rho ** np.abs(idx[:, None] - idx[None, :])
        return C
    if model == "identity":
        return np.eye(p)
    raise ValueError(f"unknown cov model: {model}")


def _chol(Sigma):
    # robust Cholesky (add tiny jitter if needed)
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eps = 1e-10 * np.trace(Sigma) / Sigma.shape[0]
        return np.linalg.cholesky(Sigma + eps * np.eye(Sigma.shape[0]))


def simulate_gaussian(n, Sigma, mu=None, rng=None):
    rng = np.random.default_rng(rng)
    p = Sigma.shape[0]
    mu = np.zeros(p) if mu is None else np.asarray(mu)
    L = _chol(Sigma)
    Z = rng.standard_normal((n, p))
    return mu + Z @ L.T


def simulate_t(n, Sigma, df, mu=None, rng=None, standardize=True):
    """Multivariate Student-t. If standardize, Cov(X) == Sigma (else scatter==Sigma).

    X = mu + Z / sqrt(g/df),  Z ~ N(0, S),  g ~ chi2(df).
    Cov(X) = S * df/(df-2). Setting S = Sigma*(df-2)/df gives Cov(X)=Sigma.
    """
    if df <= 2:
        raise ValueError("df>2 required for finite covariance")
    rng = np.random.default_rng(rng)
    p = Sigma.shape[0]
    mu = np.zeros(p) if mu is None else np.asarray(mu)
    S = Sigma * (df - 2) / df if standardize else Sigma
    L = _chol(S)
    Z = rng.standard_normal((n, p)) @ L.T
    g = rng.chisquare(df, size=n)
    return mu + Z / np.sqrt(g / df)[:, None]


def simulate_skew_t(n, Sigma, df, gamma, mu=None, rng=None):
    """GH skew-t via normal mean-variance mixture (Aas & Haff 2006 style).

    W ~ InvGamma(df/2, scale=df/2);  X = mu + gamma*W + sqrt(W) * Z,  Z ~ N(0, Sigma_base).
    gamma is a length-p skewness vector (0 -> symmetric t). Sigma_base set so that the
    mixing scale E[W]=df/(df-2) leaves the symmetric part comparable to `Sigma`.

    NOTE: for skew-t the exact population covariance also picks up Var(W)*gamma gamma';
    standardization here is APPROXIMATE (symmetric part matched). Refine for M2 if the
    gap interpretation needs an exact-covariance control. df>4 recommended.
    """
    rng = np.random.default_rng(rng)
    p = Sigma.shape[0]
    mu = np.zeros(p) if mu is None else np.asarray(mu)
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (p,))
    EW = df / (df - 2)
    Sigma_base = Sigma / EW  # so E[W]*Sigma_base = Sigma for the symmetric part
    L = _chol(Sigma_base)
    W = stats.invgamma.rvs(a=df / 2, scale=df / 2, size=n, random_state=rng)
    Z = rng.standard_normal((n, p)) @ L.T
    return mu + np.outer(W, gamma) + np.sqrt(W)[:, None] * Z
