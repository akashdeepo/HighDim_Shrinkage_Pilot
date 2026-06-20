"""Out-of-sample portfolio risk metrics. Returns are simple period returns."""
import numpy as np


def portfolio_returns(w, R):
    return R @ w


def oos_variance(w, R):
    return float(np.var(portfolio_returns(w, R), ddof=1))


def oos_std(w, R):
    return float(np.std(portfolio_returns(w, R), ddof=1))


def var_es(returns, level=0.95):
    """Historical VaR and CVaR/Expected Shortfall on LOSSES (loss = -return).

    level=0.95 -> the 5% worst tail. Returns (VaR, ES) as positive loss numbers.
    """
    losses = -np.asarray(returns)
    q = np.quantile(losses, level)
    tail = losses[losses >= q]
    es = tail.mean() if tail.size else q
    return float(q), float(es)


def cvar(returns, level=0.95):
    return var_es(returns, level)[1]


def rachev_ratio(returns, alpha=0.05, beta=0.05):
    """Rachev ratio = ETL of the upside (best beta) / ETL of the downside (worst alpha).

    Higher is better (more reward in the right tail per unit of left-tail risk).
    """
    r = np.asarray(returns)
    lo = np.quantile(r, alpha)        # worst alpha threshold
    hi = np.quantile(r, 1 - beta)     # best beta threshold
    downside = -r[r <= lo].mean()     # expected loss in the left tail (positive)
    upside = r[r >= hi].mean()        # expected gain in the right tail
    if downside <= 0:
        return np.nan
    return float(upside / downside)
