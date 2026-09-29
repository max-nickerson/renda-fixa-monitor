"""Weight builders over a candidate set x (rows of one decision cross-section).

Every builder returns a weight vector (sum 1, >= 0) aligned with x. Constraints (dict `cons`):
  issuer_cap  max weight per issuer (cnpj8)
  sector_cap  max weight per sector (None = off)
  dur_band    (lo, hi) portfolio duration band in years (None = off)
  wmax        per-row max weight array (liquidity), None = off
  min_lot     drop weights below this after solving and renormalise (then re-apply issuer cap)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp

from research.nightly import harness as H

SOLVER = "CLARABEL"


def _groups(labels):
    u, inv = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    A = np.zeros((len(u), len(inv)))
    A[inv, np.arange(len(inv))] = 1.0
    return A


def _cons_list(w, x, cons, homog=None):
    """cvxpy constraints. homog: a cvxpy scalar s (sum of y) for homogeneous (max-div) formulations."""
    tot = 1.0 if homog is None else homog
    cl = [w >= 0]
    if homog is None:
        cl.append(cp.sum(w) == 1)
    if cons.get("issuer_cap"):
        cl.append(_groups(x["cnpj8"]) @ w <= cons["issuer_cap"] * tot)
    if cons.get("sector_cap"):
        sec = x["sector"].astype(str).to_numpy()
        A = _groups(sec)
        cl.append(A @ w <= cons["sector_cap"] * tot)
    if cons.get("dur_band"):
        lo, hi = cons["dur_band"]
        d = x["dur"].to_numpy()
        cl += [d @ w >= lo * tot, d @ w <= hi * tot]
    if cons.get("wmax") is not None:
        cl.append(w <= np.asarray(cons["wmax"]) * tot)
    return cl


def _solve(prob, w):
    try:
        prob.solve(solver=SOLVER)
    except Exception:
        try:
            prob.solve(solver="SCS")
        except Exception:
            return None
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    v = np.clip(np.asarray(w.value).ravel(), 0, None)
    return v / v.sum() if v.sum() > 0 else None


def _finish(v, x, cons):
    if v is None:
        return None
    ml = cons.get("min_lot")
    if ml:
        for _ in range(3):
            keep = v >= ml
            if keep.sum() < 5:
                break
            v = np.where(keep, v, 0.0)
            v = v / v.sum()
            if (v[v > 0] >= ml).all():
                break
    return v


def simple(cons):
    return not (cons.get("sector_cap") or cons.get("dur_band") or cons.get("wmax") is not None)


def project(x, w0, cons):
    """Closest feasible weights to target w0 (least squares, relative to 1/N scale)."""
    w0 = np.asarray(w0, float)
    w0 = w0 / w0.sum()
    if simple(cons):
        return _finish(H.cap_weights(x["cnpj8"].to_numpy(), cons.get("issuer_cap") or 1.0, w0), x, cons)
    n = len(x)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w0) * n), _cons_list(w, x, cons))
    v = _solve(prob, w)
    if v is None:        # infeasible (e.g. duration band / liquidity) -> relax duration band
        c2 = dict(cons, dur_band=None)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w0) * n), _cons_list(w, x, c2))
        v = _solve(prob, w)
    return _finish(v, x, cons)


def minvar(x, Sig, cons):
    n = len(x)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sig))), _cons_list(w, x, cons))
    return _finish(_solve(prob, w), x, cons)


def meanvar(x, Sig, mu, gamma, cons, w_prev=None, tc=0.0):
    n = len(x)
    w = cp.Variable(n)
    obj = mu @ w - gamma / 2 * cp.quad_form(w, cp.psd_wrap(Sig))
    if w_prev is not None and tc > 0:
        obj = obj - tc * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Maximize(obj), _cons_list(w, x, cons))
    return _finish(_solve(prob, w), x, cons)


def maxdiv(x, Sig, vol, cons):
    """max w'sigma / sqrt(w'Sig w): min y'Sig y s.t. sigma'y = 1, y >= 0, homogeneous constraints; w = y/sum(y)."""
    n = len(x)
    y = cp.Variable(n)
    s = cp.sum(y)
    cl = _cons_list(y, x, cons, homog=s) + [vol @ y == 1]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sig))), cl)
    return _finish(_solve(prob, y), x, cons)


def erc_issuer(x, Sig, cons):
    """Equal risk contribution across issuers (bonds equal-weighted inside each issuer), Spinu log-barrier form,
    then the constraint projection (issuer / sector caps etc.)."""
    iss = x["cnpj8"].astype(str).to_numpy()
    A = _groups(iss)                                   # n_iss x n
    cnt = A.sum(axis=1)
    B = A / cnt[:, None]                               # issuer portfolio = EW of its bonds
    SI = B @ Sig @ B.T
    m = len(cnt)
    yv = cp.Variable(m)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(yv, cp.psd_wrap(SI)) - cp.sum(cp.log(yv)) / m), [yv >= 1e-8])
    try:
        prob.solve(solver=SOLVER)
        y = np.clip(np.asarray(yv.value).ravel(), 1e-12, None)
    except Exception:
        y = 1 / np.sqrt(np.diag(SI))
    y = y / y.sum()
    w0 = B.T @ y
    return project(x, w0, cons)


def cvar_opt(x, scen, lam, cons, a=0.975):
    """max mean(scen) w - lam * CVaR_a(-scen w)  (Rockafellar-Uryasev LP)."""
    n_s, n = scen.shape
    w = cp.Variable(n)
    al = cp.Variable()
    u = cp.Variable(n_s)
    loss = -scen @ w
    cv = al + cp.sum(u) / ((1 - a) * n_s)
    prob = cp.Problem(cp.Maximize(scen.mean(axis=0) @ w - lam * cv), _cons_list(w, x, cons) + [u >= 0, u >= loss - al])
    return _finish(_solve(prob, w), x, cons)
