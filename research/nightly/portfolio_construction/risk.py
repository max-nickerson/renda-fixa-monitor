"""Risk models for sparse, stale debenture marks + default-scenario simulation.

All estimates at decision grid position p use only R[:p] (R[t] = t -> t+1 move, realised at the t+1 mark <= p).

Two covariance models (annualised, excess-over-CDI rate-hedged returns):
  'dts'  structural:  sigma_i = kappa * DTS_i  (DTS = duration x CDI+ spread, Ben Dor et al. 2007) and a block
         correlation (same issuer / same sector / market) whose three levels are estimated on trailing data.
  'emp'  de-smoothed empirical: weekly returns over the trailing 104 weeks, aggregated to overlapping 4-week sums
         (variance-ratio de-smoothing: captures the autocorrelation that stale marks induce up to lag 3);
         own vol if >= 26 weekly obs else the DTS-predicted vol; correlation = 50/50 shrink of the pairwise sample
         correlation towards the block structure; eigenvalue-clipped to PSD.

Default scenarios (CVaR): issuer-level defaults from a one-factor + sector Gaussian copula, PD over horizon h from
the CDI+ spread (hazard = s / (1 - R), R = 40%) times a real-world / risk-neutral ratio psi, loss on default =
1 - 0.40 / ratio (price as % of par), plus a normal spread-MTM component with the 'dts' covariance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from research.nightly import harness as H

_W = {}


def weekly_returns():
    """(week_end_pos array, W matrix n_weeks x NB of compounded weekly returns, OBS mask: bond had a grid row)."""
    if "W" in _W:
        return _W["W"]
    C = H._core()
    R = H._Rmat("base")
    TD = C["TD"]
    ND = R.shape[0]
    blocks = np.arange(ND) // 5
    nb = blocks.max() + 1
    LR = np.log1p(np.clip(R, -0.99, None))
    Wl = np.zeros((nb, R.shape[1]))
    np.add.at(Wl, blocks, LR)
    obs = np.zeros((nb, R.shape[1]), dtype=bool)
    np.logical_or.at(obs, blocks, (TD >= 0) | (R != 0))   # a gap move is booked on a non-row day
    end_pos = np.array([np.nonzero(blocks == k)[0].max() for k in range(nb)])   # last R row in block
    Wm = np.where(obs, np.expm1(Wl), np.nan)
    _W["W"] = (end_pos, Wm)
    return _W["W"]


def _window(p: int, n_weeks: int = 104):
    """Weekly returns fully realised by close p: block's last R row t satisfies t + 1 <= p."""
    end_pos, Wm = weekly_returns()
    k = np.nonzero(end_pos + 1 <= p)[0]
    k = k[-n_weeks:]
    return Wm[k]


def _sum4(X: np.ndarray) -> np.ndarray:
    """Overlapping 4-week sums (NaN if any NaN in the window)."""
    if X.shape[0] < 4:
        return X[:0]
    return X[3:] + X[2:-1] + X[1:-2] + X[:-3]


def block_params(p: int, univ_b: np.ndarray, sec: np.ndarray, iss: np.ndarray, max_n: int = 450, seed: int = 0):
    """Average pairwise correlation (4-week sums) for same-issuer / same-sector-diff-issuer / other pairs, and
    kappa = vol / DTS scale (fitted later). Estimated on universe bonds with >= 26 weekly obs."""
    key = ("bp", p)
    if key in _W:
        return _W[key]
    X = _window(p)
    S4 = _sum4(X[:, univ_b])
    nobs = np.isfinite(S4).sum(axis=0)
    ok = nobs >= 26
    idx = np.nonzero(ok)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, max_n, replace=False))
    Cm = pd.DataFrame(S4[:, idx]).corr(min_periods=20).to_numpy()
    si, ii = sec[idx], iss[idx]
    same_i = ii[:, None] == ii[None, :]
    same_s = (si[:, None] == si[None, :]) & ~same_i
    off = ~np.eye(len(idx), dtype=bool)
    fin = np.isfinite(Cm)

    def avg(m):
        v = Cm[m & off & fin]
        return float(np.clip(np.nanmean(v), 0, 0.95)) if len(v) > 10 else np.nan
    r_i, r_s, r_m = avg(same_i), avg(same_s), avg(~same_i & ~same_s)
    r_m = 0.1 if not np.isfinite(r_m) else r_m
    r_s = max(r_m, r_s if np.isfinite(r_s) else r_m + 0.1)
    r_i = max(r_s, r_i if np.isfinite(r_i) else 0.7)
    out = {"rho_iss": r_i, "rho_sec": r_s, "rho_mkt": r_m}
    _W[key] = out
    return out


def block_corr(sec, iss, bp):
    sec, iss = np.asarray(sec), np.asarray(iss)
    same_i = iss[:, None] == iss[None, :]
    same_s = (sec[:, None] == sec[None, :]) & ~same_i
    C = np.full((len(sec), len(sec)), bp["rho_mkt"])
    C[same_s] = bp["rho_sec"]
    C[same_i] = bp["rho_iss"]
    np.fill_diagonal(C, 1.0)
    return C


def _psd(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, 1e-4, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / d[:, None] / d[None, :]


def emp_vol(p: int, b: np.ndarray):
    """De-smoothed annual vol from overlapping 4-week sums: sqrt(var(sum4) * 13). NaN if < 26 obs."""
    X = _window(p)
    S4 = _sum4(X[:, b])
    n = np.isfinite(S4).sum(axis=0)
    v = np.nanstd(S4, axis=0, ddof=1) * np.sqrt(13)
    v[n < 26] = np.nan
    return v, S4


def cov(p: int, x: pd.DataFrame, kind: str = "dts", bp=None):
    """Covariance (annual) for the rows of x (needs b, dur, cdi_bps, sector, cnpj8). Returns (Sigma, vol)."""
    b = x["b"].to_numpy()
    sec = x["sector"].to_numpy().astype(str)
    iss = x["cnpj8"].to_numpy().astype(str)
    dts = x["dur"].clip(0.25, 15).to_numpy() * np.clip(x["cdi_bps"].to_numpy(), 30, 3000) / 1e4
    ev, S4 = emp_vol(p, b)
    ok = np.isfinite(ev) & (dts > 0)
    # kappa: median ratio emp vol / DTS (robust); fallback 0.35
    kappa = float(np.nanmedian(ev[ok] / dts[ok])) if ok.sum() >= 10 else 0.35
    vd = kappa * dts
    C0 = block_corr(sec, iss, bp)
    if kind == "dts":
        # stale marks bias measured correlations towards 0 (same-issuer pairs measure only ~0.2, an Epps-like
        # effect); the structural model floors them at economic priors (issuer 0.8 / sector 0.35 / market 0.2)
        bps = {"rho_iss": max(bp["rho_iss"], 0.8), "rho_sec": max(bp["rho_sec"], 0.35),
               "rho_mkt": max(bp["rho_mkt"], 0.2)}
        vol, C = vd, block_corr(sec, iss, bps)
    else:
        vol = np.where(ok, 0.5 * ev + 0.5 * vd, vd)          # shrink own vol half-way to the DTS prediction
        Cs = pd.DataFrame(S4).corr(min_periods=26).to_numpy()
        Cs = np.where(np.isfinite(Cs), Cs, C0)
        C = _psd(0.5 * Cs + 0.5 * C0)
    Sig = C * vol[:, None] * vol[None, :]
    return Sig, vol


def default_scenarios(x: pd.DataFrame, bp: dict, n: int = 3000, h: float = 0.5, psi: float = 0.5,
                      rec: float = 0.40, rho_d_mkt: float = 0.15, rho_d_sec: float = 0.30, Sig=None, seed: int = 7):
    """Scenario returns (n x N) over horizon h: carry s*h - default loss (issuer-level Gaussian copula) + MTM noise.
    Returns (scen, pd_h) where pd_h = real-world horizon PD per row."""
    rng = np.random.default_rng(seed)
    s = np.clip(x["cdi_bps"].to_numpy(), 0, 3000) / 1e4
    ratio = np.clip(x["ratio"].to_numpy(), 0.5, 1.2)
    lgd = np.clip(1 - rec / ratio, 0, 1)
    lam = s / (1 - rec) * psi
    pdh = 1 - np.exp(-lam * h)
    thr = norm.ppf(np.clip(pdh, 1e-8, 0.999))
    iss = x["cnpj8"].astype(str).to_numpy()
    sec = x["sector"].astype(str).to_numpy()
    ui, ii = np.unique(iss, return_inverse=True)
    us, si = np.unique(sec, return_inverse=True)
    M = rng.standard_normal((n, 1))
    Sf = rng.standard_normal((n, len(us)))
    E = rng.standard_normal((n, len(ui)))
    # issuer latent: sqrt(rm) M + sqrt(rs - rm) S_sec + sqrt(1 - rs) e
    sec_of_iss = pd.Series(si).groupby(ii).first().reindex(range(len(ui))).to_numpy()
    Z = np.sqrt(rho_d_mkt) * M + np.sqrt(rho_d_sec - rho_d_mkt) * Sf[:, sec_of_iss] + np.sqrt(1 - rho_d_sec) * E
    D = Z[:, ii] < thr[None, :]                      # all bonds of an issuer default together (issuer latent)
    scen = s[None, :] * h - D * lgd[None, :]
    if Sig is not None:
        L = np.linalg.cholesky(Sig * h + 1e-10 * np.eye(len(s)))
        scen = scen + rng.standard_normal((n, len(s))) @ L.T
    return scen, pdh


def cvar(losses: np.ndarray, a: float = 0.99) -> float:
    q = np.quantile(losses, a)
    return float(losses[losses >= q].mean())
