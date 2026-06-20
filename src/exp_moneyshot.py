"""Experiment M2 -- THE MONEY SHOT (with in-sample vs out-of-sample diagnostic).

Original hypothesis: under skewed heavy tails, a CVaR-optimal portfolio should beat the
variance-optimal (shrinkage GMV) portfolio on realized tail risk (ES).

Reality check built in: we measure ES BOTH in-sample (on the training scenarios the CVaR
LP optimized over) AND out-of-sample (fresh test set). This separates two effects:
  - objective mismatch (does targeting CVaR help when you know the distribution?)  -> in-sample
  - estimation error / overfitting of the empirical tail                            -> in vs out gap

Design: p=50, n=500 (c=0.1). DGP GH skew-t, sweep df at symmetric vs left-skewed.
variance-optimal = Ledoit-Wolf shrinkage GMV ; CVaR-optimal = Rockafellar-Uryasev LP @95%.

Run:  python src/exp_moneyshot.py
Outputs: figures/m2_overfit.png, results/m2_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgp import make_cov, simulate_skew_t            # noqa: E402
from estimators import gmv_ledoit_wolf               # noqa: E402
from cvar_opt import cvar_optimal_weights            # noqa: E402
from metrics import oos_variance, var_es             # noqa: E402

P = 50
N_TRAIN = 500
N_TEST = 20000
REPS = 40
RHO = 0.6
DFS = [8.0, 6.0, 5.0, 4.5]
GAMMAS = {"symmetric": 0.0, "left-skewed": -0.2}
BETA = 0.95
BASE_SEED = 20260619


def es99(w, R):
    return var_es(R @ w, level=0.99)[1]


def es95(w, R):
    return var_es(R @ w, level=0.95)[1]


def run():
    rng = np.random.default_rng(BASE_SEED)
    Sigma = make_cov(P, model="ar1", rho=RHO)
    rows = []
    for label, g in GAMMAS.items():
        for df in DFS:
            acc = {k: [] for k in ["es_v_in", "es_c_in", "es_v_out", "es_c_out",
                                    "es95_v_in", "es95_c_in", "es95_v_out", "es95_c_out",
                                    "var_v", "var_c"]}
            for _ in range(REPS):
                train = simulate_skew_t(N_TRAIN, Sigma, df=df, gamma=g, rng=rng)
                test = simulate_skew_t(N_TEST, Sigma, df=df, gamma=g, rng=rng)
                w_var = gmv_ledoit_wolf(train)
                w_cv = cvar_optimal_weights(train, beta=BETA)
                acc["es_v_in"].append(es99(w_var, train))
                acc["es_c_in"].append(es99(w_cv, train))
                acc["es_v_out"].append(es99(w_var, test))
                acc["es_c_out"].append(es99(w_cv, test))
                acc["es95_v_in"].append(es95(w_var, train))
                acc["es95_c_in"].append(es95(w_cv, train))
                acc["es95_v_out"].append(es95(w_var, test))
                acc["es95_c_out"].append(es95(w_cv, test))
                acc["var_v"].append(oos_variance(w_var, test))
                acc["var_c"].append(oos_variance(w_cv, test))
            m = {k: float(np.nanmean(v)) for k, v in acc.items()}
            rows.append({"skew": label, "df": df, **m,
                         "es_ratio_in": m["es_v_in"] / m["es_c_in"],
                         "es_ratio_out": m["es_v_out"] / m["es_c_out"],
                         "es95_ratio_in": m["es95_v_in"] / m["es95_c_in"],
                         "es95_ratio_out": m["es95_v_out"] / m["es95_c_out"]})
            print(f"{label:12s} df={df:>4}  "
                  f"ES95(matched) in-ratio={m['es95_v_in']/m['es95_c_in']:.2f} "
                  f"out-ratio={m['es95_v_out']/m['es95_c_out']:.2f}   |   "
                  f"ES99 in-ratio={m['es_v_in']/m['es_c_in']:.2f} "
                  f"out-ratio={m['es_v_out']/m['es_c_out']:.2f}")

    df_out = pd.DataFrame(rows)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    df_out.to_csv(os.path.join(here, "results", "m2_summary.csv"), index=False)

    # Figure: in-sample vs out-of-sample ES99 ratio (varopt / cvaropt), left-skewed.
    # ratio < 1 => CVaR-opt has higher ES (is worse). The in->out flip = overfitting.
    sub = df_out[df_out["skew"] == "left-skewed"].sort_values("df")
    plt.figure(figsize=(7, 4.5))
    plt.plot(sub.df, sub.es_ratio_in, "o-", label="in-sample (knows the tail)")
    plt.plot(sub.df, sub.es_ratio_out, "s-", label="out-of-sample (must generalize)")
    plt.axhline(1.0, color="gray", lw=0.8, ls=":")
    plt.gca().invert_xaxis()
    plt.xlabel("degrees of freedom  (smaller = heavier tails $\\rightarrow$)")
    plt.ylabel("ES(99%) ratio:  variance-opt / CVaR-opt")
    plt.title("CVaR optimization wins in-sample but OVERFITS out-of-sample\n"
              "(ratio<1: CVaR-opt worse; >1: variance-opt worse) -- left-skewed heavy tails")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(here, "figures", "m2_overfit.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure -> {fig_path}")


if __name__ == "__main__":
    run()
