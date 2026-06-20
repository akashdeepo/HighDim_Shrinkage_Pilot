"""Experiment 1 (M1) -- SANITY / FOUNDATION.

Question: across the concentration ratio c = p/n (from low-dim to singular p>n),
does shrinkage stabilize the GMV portfolio where the naive pseudo-inverse falls apart?

Gate A passes if, as c -> 1 and beyond:
  - naive pseudo-inverse GMV out-of-sample variance blows up,
  - Ledoit-Wolf shrinkage GMV stays close to the oracle (true-Sigma) GMV.

Run:  python src/exp1_sanity.py
Outputs: figures/exp1_oos_variance.png, results/exp1_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgp import make_cov, simulate_gaussian          # noqa: E402
from estimators import gmv_pinv, gmv_ledoit_wolf, gmv_oracle  # noqa: E402
from metrics import oos_variance                      # noqa: E402

# ---- config ----
N_TRAIN = 250
RATIOS = [0.3, 0.5, 0.7, 0.9, 1.1, 1.5]
REPS = 100
N_TEST = 20000
RHO = 0.6
BASE_SEED = 20260619

METHODS = {
    "naive_pinv": gmv_pinv,
    "ledoit_wolf": gmv_ledoit_wolf,
}


def run():
    rng = np.random.default_rng(BASE_SEED)
    rows = []
    for c in RATIOS:
        p = max(2, int(round(c * N_TRAIN)))
        Sigma = make_cov(p, model="ar1", rho=RHO)
        acc = {m: [] for m in METHODS}
        acc["oracle"] = []
        for _ in range(REPS):
            train = simulate_gaussian(N_TRAIN, Sigma, rng=rng)
            test = simulate_gaussian(N_TEST, Sigma, rng=rng)
            for name, fn in METHODS.items():
                try:
                    w = fn(train)
                    acc[name].append(oos_variance(w, test))
                except Exception:
                    acc[name].append(np.nan)
            acc["oracle"].append(oos_variance(gmv_oracle(Sigma), test))
        row = {"c_ratio": c, "p": p, "n": N_TRAIN}
        for k, v in acc.items():
            row[f"{k}_oos_var"] = float(np.nanmean(v))
        rows.append(row)
        print(f"c={c:>4}  p={p:>4}  "
              f"naive={row['naive_pinv_oos_var']:.4g}  "
              f"LW={row['ledoit_wolf_oos_var']:.4g}  "
              f"oracle={row['oracle_oos_var']:.4g}")

    df = pd.DataFrame(rows)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    os.makedirs(os.path.join(here, "figures"), exist_ok=True)
    df.to_csv(os.path.join(here, "results", "exp1_summary.csv"), index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(df.c_ratio, df.naive_pinv_oos_var, "o-", label="Naive pseudo-inverse GMV")
    plt.plot(df.c_ratio, df.ledoit_wolf_oos_var, "s-", label="Ledoit-Wolf shrinkage GMV")
    plt.plot(df.c_ratio, df.oracle_oos_var, "k--", label="Oracle (true $\\Sigma$) GMV")
    plt.axvline(1.0, color="gray", lw=0.8, ls=":")
    plt.xlabel("concentration ratio  c = p / n")
    plt.ylabel("out-of-sample portfolio variance")
    plt.yscale("log")
    plt.title("Exp 1 (sanity): shrinkage vs naive GMV across p/n")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(here, "figures", "exp1_oos_variance.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure -> {fig_path}")
    print("Gate A check: naive should explode as c->1+, LW should track the oracle.")


if __name__ == "__main__":
    run()
