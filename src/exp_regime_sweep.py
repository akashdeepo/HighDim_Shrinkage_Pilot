"""Experiment M2b -- IN-REGIME sweep (the fix to the low-dimensional pilot).

The original money shot ran at p=50, n=500 (c=0.1): low-dimensional, covariance
well-conditioned, generalized inverse never appears -- NOT the regime the project is
about. This rerun sweeps c = p/n across the high-dimensional and singular cases
(crossing p = n into p > n), where the generalized inverse genuinely appears.

What we learned forcing the rerun: with short-selling allowed, empirical CVaR
optimization becomes ILL-POSED (unbounded -- in-sample pseudo-arbitrage) once p is
large. So both portfolios are constrained long-only (no short sales, w>=0, 1'w=1),
the standard textbook setting, which bounds the CVaR program and gives a fair
apples-to-apples comparison.

n fixed at 300; p in {150, 270, 300, 360} -> c in {0.5, 0.9, 1.0, 1.2}.
DGP: skewed heavy-tailed (skew-t, df=5, skew gamma=-0.2).
variance-optimal = long-only GMV on the Ledoit-Wolf shrinkage covariance (regularized;
                   the shrinkage target stays invertible even at p>n).
CVaR-optimal     = long-only Rockafellar-Uryasev LP at 95% (naive / empirical).

Report per c: in-sample and out-of-sample 95% ES ratio (variance-opt / CVaR-opt).
Ratio > 1 => variance-opt worse; < 1 => CVaR-opt worse. No pre-judged direction.

Run:  python src/exp_regime_sweep.py
Outputs: figures/m2b_regime_sweep.png, results/m2b_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgp import make_cov, simulate_skew_t            # noqa: E402
from cvar_opt import cvar_optimal_weights            # noqa: E402
from metrics import var_es, oos_variance             # noqa: E402

N_TRAIN = 300
P_LIST = [150, 270, 300, 360]      # c = 0.5, 0.9, 1.0, 1.2
N_TEST = 10000
REPS = 20
DF = 5.0
GAMMA = -0.2
RHO = 0.6
BETA = 0.95
BASE_SEED = 20260620


def es95(w, R):
    return var_es(R @ w, level=0.95)[1]


def long_only_gmv(R):
    """Min-variance over the simplex on the Ledoit-Wolf shrinkage covariance (QP)."""
    Sigma = LedoitWolf().fit(R).covariance_
    p = Sigma.shape[0]
    ones = np.ones(p)
    res = minimize(lambda w: w @ Sigma @ w, ones / p, jac=lambda w: 2 * Sigma @ w,
                   bounds=[(0.0, None)] * p,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                 "jac": lambda w: ones}],
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
    return res.x


def run():
    rng = np.random.default_rng(BASE_SEED)
    print(f"seed={BASE_SEED}  n={N_TRAIN}  df={DF}  gamma={GAMMA}  reps={REPS}  "
          f"beta={BETA}  (long-only both)")
    rows = []
    for p in P_LIST:
        c = p / N_TRAIN
        Sigma = make_cov(p, model="ar1", rho=RHO)
        acc = {k: [] for k in ["es_v_in", "es_c_in", "es_v_out", "es_c_out",
                                "var_v_out", "var_c_out"]}
        for _ in range(REPS):
            train = simulate_skew_t(N_TRAIN, Sigma, df=DF, gamma=GAMMA, rng=rng)
            test = simulate_skew_t(N_TEST, Sigma, df=DF, gamma=GAMMA, rng=rng)
            w_var = long_only_gmv(train)                          # regularized variance-opt
            w_cv = cvar_optimal_weights(train, beta=BETA, allow_short=False)  # naive CVaR
            acc["es_v_in"].append(es95(w_var, train))
            acc["es_c_in"].append(es95(w_cv, train))
            acc["es_v_out"].append(es95(w_var, test))
            acc["es_c_out"].append(es95(w_cv, test))
            acc["var_v_out"].append(oos_variance(w_var, test))
            acc["var_c_out"].append(oos_variance(w_cv, test))
        m = {k: float(np.nanmean(v)) for k, v in acc.items()}
        in_ratio = m["es_v_in"] / m["es_c_in"]
        out_ratio = m["es_v_out"] / m["es_c_out"]
        rows.append({"c": c, "p": p, "n": N_TRAIN, **m,
                     "es_ratio_in": in_ratio, "es_ratio_out": out_ratio})
        print(f"c={c:>4} p={p:>4}  ES95 in: var={m['es_v_in']:.3f} cvar={m['es_c_in']:.3f} "
              f"(ratio {in_ratio:.2f})  out: var={m['es_v_out']:.3f} cvar={m['es_c_out']:.3f} "
              f"(ratio {out_ratio:.2f})")

    df = pd.DataFrame(rows)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    df.to_csv(os.path.join(here, "results", "m2b_summary.csv"), index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(df.c, df.es_ratio_in, "o-", label="in-sample (knows the tail)")
    plt.plot(df.c, df.es_ratio_out, "s-", label="out-of-sample (must generalize)")
    plt.axhline(1.0, color="gray", lw=0.8, ls=":")
    plt.axvline(1.0, color="crimson", lw=0.8, ls=":", label="p = n (singular)")
    plt.xlabel("concentration ratio  c = p / n")
    plt.ylabel("ES(95%) ratio:  variance-opt / CVaR-opt")
    plt.title("In-regime sweep (long-only): CVaR-opt vs regularized variance-opt\n"
              "(ratio>1: variance-opt worse; <1: CVaR-opt worse) -- skew-t, df=5")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(here, "figures", "m2b_regime_sweep.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure -> {fig_path}")


if __name__ == "__main__":
    run()
