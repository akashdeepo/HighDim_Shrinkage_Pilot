# High-Dimensional Shrinkage Portfolios under Heavy Tails — Simulation Pilot

A small, reproducible simulation study of regularized portfolio estimation when the
number of assets is comparable to or larger than the sample size (`p ≈ n`, `p > n`) and
returns are heavy-tailed and skewed. The focus is the gap between **variance-based** and
**downside-risk** (CVaR / expected shortfall) portfolio objectives, and the role of
**shrinkage / regularization** in making either one work out-of-sample.

## Background

When `p ≈ n`, the sample covariance matrix is ill-conditioned or singular, so its inverse
— the object every optimal portfolio rule depends on — amplifies estimation error. Two
established remedies are the **generalized (Moore–Penrose / ridge-type) inverse** and
**linear / nonlinear shrinkage** of the covariance or the portfolio weights (Bodnar &
Parolya, *Reviving Pseudo-Inverses*, Annals of Statistics 54(2):1053–1079, 2026,
doi:10.1214/25-AOS2602; Ledoit & Wolf). This pilot asks how those ideas carry over to
**heavy-tailed, downside-risk** settings.

## Experiments

**Exp 1 — foundation (`src/exp1_sanity.py`).** Out-of-sample GMV portfolio variance across
the concentration ratio `c = p/n` (0.3 → 1.5), Gaussian returns: naive pseudo-inverse GMV
vs. Ledoit–Wolf shrinkage GMV vs. oracle. *Result:* shrinkage tracks the oracle throughout,
while the naive pseudo-inverse degrades and **peaks at the interpolation threshold `c ≈ 1`** —
the pseudo-inverse pathology that motivates regularization. See `figures/exp1_oos_variance.png`.

**Exp 2 — variance vs. downside-risk, low-dimensional first pass (`src/exp_moneyshot.py`).**
At `p=50, n=500` (`c=0.1`, covariance well-estimated), compares a variance-optimal portfolio
(Ledoit–Wolf shrinkage GMV) with a CVaR-optimal portfolio (Rockafellar–Uryasev), in-sample
vs. out-of-sample, under GH skew-t returns. *Result:* CVaR optimization wins in-sample but
overfits and reverses out-of-sample. **Caveat: `c=0.1` is the low-dimensional regime** —
useful as an isolating first check, but not the singular-covariance regime the project is
about. Exp 2b fixes that. See `figures/m2_overfit.png`.

**Exp 2b — in-regime sweep (`src/exp_regime_sweep.py`).** The on-regime version: sweeps
`c = p/n ∈ {0.5, 0.9, 1.0, 1.2}` (n=300), crossing `p = n` into `p > n`, skew-t returns.
Variance-optimal = long-only Ledoit–Wolf shrinkage GMV; CVaR-optimal = long-only
Rockafellar–Uryasev LP. *Results, reported as they came:*
1. **Unconstrained, the empirical CVaR program is ill-posed in high dimensions** (unbounded LP —
   in-sample pseudo-arbitrage). This is why both portfolios are constrained long-only.
2. **Long-only:** CVaR-opt lowers in-sample 95% ES by ~20%, but the advantage does **not**
   survive out-of-sample, where it is a few percent *worse* than the regularized variance
   portfolio.
3. The out-of-sample gap is **flat across `c`** — it does not grow with dimension.

This is consistent with the known fragility of CVaR optimization (Lim, Shanthikumar & Vahn,
*Operations Research Letters* 39(3):163–171, 2011), now seen in the high-dimensional setting.
It frames an open question rather than a finished result: can shrinkage / generalized-inverse
regularization stabilize downside-risk objectives the way it stabilized variance objectives?
See `figures/m2b_regime_sweep.png`.

## Repository structure

```
src/
  dgp.py              # return models: Gaussian, Student-t, GH skew-t (common target Σ)
  estimators.py       # GMV via pseudo-inverse, Ledoit–Wolf shrinkage, oracle
  cvar_opt.py         # CVaR-optimal weights via the Rockafellar–Uryasev LP (scipy.linprog)
  metrics.py          # OOS variance, VaR, CVaR/ES, Rachev ratio
  exp1_sanity.py      # Exp 1
  exp_moneyshot.py    # Exp 2  (low-dimensional first pass)
  exp_regime_sweep.py # Exp 2b (in-regime sweep, long-only)
figures/              # generated plots
results/              # generated summary CSVs
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
python src/exp1_sanity.py
python src/exp_moneyshot.py
python src/exp_regime_sweep.py
```

Seeds are fixed (`numpy` Generator) for reproducibility.

## Limitations

Exploratory pilot, not a finished study: a single heavy-tailed DGP family (GH skew-t),
moderate dimension, simulated data only, and the CVaR-optimal portfolio is the
*un-regularized* baseline (regularizing it is the proposed direction, not implemented here).

## Author

Akash Deep — exploratory work in high-dimensional portfolio statistics and heavy-tailed
risk. Built with reference to the shrinkage / random-matrix program of T. Bodnar, N. Parolya
and collaborators, and the heavy-tailed (Normal Tempered-Stable) econometrics developed with
S. T. Rachev and F. J. Fabozzi.
