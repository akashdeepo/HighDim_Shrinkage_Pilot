"""CVaR-optimal portfolio via the Rockafellar-Uryasev LP (scipy.linprog, no extra deps).

Minimize CVaR_beta of the portfolio LOSS (loss = -return) subject to budget 1'w = 1.
Shorts allowed by default, matching the (sign-unconstrained) GMV comparison.

RU formulation. Variables x = [w (p), zeta (1), u (N)]:
    min  zeta + 1/(N(1-beta)) * sum_j u_j
    s.t. u_j >= (-R_j . w) - zeta,   u_j >= 0,   1'w = 1
where R_j is training scenario j (a length-p return vector).
"""
import numpy as np
from scipy.optimize import linprog


def cvar_optimal_weights(R_train, beta=0.95, allow_short=True):
    R = np.asarray(R_train, dtype=float)
    N, p = R.shape
    nvar = p + 1 + N

    # objective: zeta + 1/(N(1-beta)) sum u
    c = np.zeros(nvar)
    c[p] = 1.0
    c[p + 1:] = 1.0 / (N * (1.0 - beta))

    # inequality: (-R_j).w - zeta - u_j <= 0
    A_ub = np.zeros((N, nvar))
    A_ub[:, :p] = -R
    A_ub[:, p] = -1.0
    A_ub[:, p + 1:] = -np.eye(N)
    b_ub = np.zeros(N)

    # equality: sum(w) = 1
    A_eq = np.zeros((1, nvar))
    A_eq[0, :p] = 1.0
    b_eq = np.array([1.0])

    w_bound = (None, None) if allow_short else (0.0, None)
    bounds = [w_bound] * p + [(None, None)] + [(0.0, None)] * N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    return res.x[:p]


if __name__ == "__main__":
    # quick self-check: on symmetric Gaussian data CVaR-opt should roughly match GMV
    rng = np.random.default_rng(0)
    R = rng.standard_normal((400, 8)) * 0.02
    w = cvar_optimal_weights(R, beta=0.95)
    print("sum(w)=", round(float(w.sum()), 6), " weights:", np.round(w, 3))
