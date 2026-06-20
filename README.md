# High-Dimensional Shrinkage Portfolios under Heavy Tails — Simulation Pilot

A small, reproducible simulation study of regularized portfolio estimation when the
number of assets is comparable to the sample size (`p ≈ n`) and returns are heavy-tailed
and skewed. The focus is the gap between **variance-based** and **downside-risk** (CVaR /
expected shortfall) portfolio objectives, and the role of **shrinkage / regularization**
in making either one work out-of-sample.

## Background

When `p ≈ n`, the sample covariance matrix is ill-conditioned or singular, so its inverse
— the object every optimal portfolio rule depends on — amplifies estimation error. Two
established remedies are the **generalized (Moore–Penrose / ridge-type) inverse** and
**linear / nonlinear shrinkage** of the covariance or the portfolio weights (see Bodnar &
Parolya, *Reviving Pseudo-Inverses*, Annals of Statistics 54(2):1053–1079, 2026; Ledoit &
Wolf). This pilot asks how those ideas carry over to **heavy-tailed, downside-risk** settings.

## Experiments

**Exp 1 — foundation (`src/exp1_sanity.py`).** Out-of-sample GMV portfolio variance across
the concentration ratio `c = p/n` (0.3 → 1.5), Gaussian returns, comparing the naive
pseudo-inverse GMV, Ledoit–Wolf shrinkage GMV, and the oracle (true-Σ) GMV.
*Result:* shrinkage tracks the oracle throughout, while the naive pseudo-inverse degrades
sharply and peaks at the interpolation threshold `c ≈ 1` — the pseudo-inverse pathology
that motivates regularization. See `figures/exp1_oos_variance.png`.

**Exp 2 — variance vs. downside-risk objectives (`src/exp_moneyshot.py`).** Under GH
skew-t returns (`p=50, n=500`, covariance well-estimated), compares a variance-optimal
portfolio (Ledoit–Wolf shrinkage GMV) with a CVaR-optimal portfolio (Rockafellar–Uryasev
LP), measuring expected shortfall **both in-sample and out-of-sample**.
*Result:* CVaR optimization lowers in-sample 95% ES by 23–38% — and the advantage grows as
tails fatten and skew increases — but **naive empirical CVaR optimization overfits the tail
and reverses out-of-sample**, underperforming the regularized variance-optimal portfolio by
~20% (confirmed at the optimizer's own 95% level). This suggests that the estimation-error
problem solved by shrinkage for *variance* objectives has not yet been addressed for
*downside-risk* objectives — a natural direction for regularized CVaR / Rachev-ratio
high-dimensional portfolio estimation. See `figures/m2_overfit.png`.

## Repository structure

```
src/
  dgp.py            # return models: Gaussian, Student-t, GH skew-t (common target Σ)
  estimators.py     # GMV via pseudo-inverse, Ledoit–Wolf shrinkage, oracle
  cvar_opt.py       # CVaR-optimal weights via the Rockafellar–Uryasev LP (scipy.linprog)
  metrics.py        # OOS variance, VaR, CVaR/ES, Rachev ratio
  exp1_sanity.py    # Exp 1
  exp_moneyshot.py  # Exp 2
figures/            # generated plots
results/            # generated summary CSVs
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
python src/exp1_sanity.py
python src/exp_moneyshot.py
```

Seeds are fixed (`numpy` Generator, base seed 20260619) for reproducibility.

## Limitations

Exploratory pilot, not a finished study: a single heavy-tailed DGP family (GH skew-t),
moderate dimension, simulated data only, and the CVaR-optimal portfolio is the
*un-regularized* baseline (regularizing it is the proposed direction, not implemented here).

## Author

Akash Deep — exploratory work in high-dimensional portfolio statistics and heavy-tailed
risk. Built with reference to the shrinkage / random-matrix program of T. Bodnar, N. Parolya
and collaborators, and the heavy-tailed (Normal Tempered-Stable) econometrics developed with
S. T. Rachev and F. J. Fabozzi.
