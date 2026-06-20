"""Portfolio-weight estimators for the pilot.

GMV (global minimum variance) weights from a covariance estimate:
    w = Sigma^{-1} 1 / (1' Sigma^{-1} 1).

Estimators compared:
  - gmv_pinv         : sample covariance + Moore-Penrose pseudo-inverse (the "naive"
                       baseline; ties directly to Bodnar's pseudo-inverse theme; works p>n)
  - gmv_ledoit_wolf  : Ledoit-Wolf LINEAR shrinkage covariance, then GMV (correct, standard)
  - gmv_oracle       : GMV from the TRUE Sigma (lower-bound benchmark)
  - gmv_bdps19       : Bodnar-Parolya-Schmid (2018) GMV weight shrinkage -- TODO/VERIFY
"""
import numpy as np
from sklearn.covariance import LedoitWolf


def gmv_from_cov(Sigma, use_pinv=False):
    p = Sigma.shape[0]
    ones = np.ones(p)
    if use_pinv:
        Sinv_ones = np.linalg.pinv(Sigma) @ ones
    else:
        Sinv_ones = np.linalg.solve(Sigma, ones)
    return Sinv_ones / (ones @ Sinv_ones)


def sample_cov(R):
    # R: n x p, rows = observations
    return np.cov(R, rowvar=False)


def gmv_pinv(R):
    """Naive GMV: sample covariance, Moore-Penrose inverse (handles singular p>=n)."""
    return gmv_from_cov(sample_cov(R), use_pinv=True)


def gmv_ledoit_wolf(R):
    """Ledoit-Wolf linear-shrinkage covariance, then GMV. Standard, well-validated."""
    lw = LedoitWolf().fit(R)
    return gmv_from_cov(lw.covariance_, use_pinv=False)


def gmv_oracle(Sigma_true):
    """GMV from the known population covariance -- the benchmark everyone is chasing."""
    return gmv_from_cov(Sigma_true, use_pinv=False)


def gmv_bdps19(R, target=None):
    """Bodnar-Parolya-Schmid (2018, EJOR) GMV shrinkage estimator.

    Form:  w_hat = alpha * w_sample + (1 - alpha) * target,   1'target = 1.
    The optimal shrinkage intensity alpha* is an RMT result (function of c=p/n and the
    sample GMV relative loss). DO NOT hardcode from memory -- implement directly from the
    paper or cross-check against HDShOP::new_GMV_portfolio_weights_BDPS19 before trusting.
    """
    raise NotImplementedError(
        "BDPS19 intensity must be implemented/verified from the paper or HDShOP. "
        "Until then, use gmv_ledoit_wolf as the shrinkage representative (Gate A)."
    )
