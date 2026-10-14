"""From-scratch implementations used by stages 13-18 (no R available).

PERMANOVA (sequential SS via Gower-centred projection), classical MDS, correspondence analysis,
detrended CA (detrending by segments + Hill-style rescaling, simplified), spectral seriation,
Holland delta score, normalised compression distance, Jensen-Shannon distance, leaf permutation.
Each method has a known-answer gate in s13_gates.py.
"""
import bz2, zlib
import numpy as np
from scipy.spatial.distance import jensenshannon, pdist, squareform


# ------------------------------------------------------------------ distances / ordination
def js_matrix(P):
    return squareform(pdist(P, lambda a, b: jensenshannon(a, b, base=2)))


def profiles(M, wmask, eps=1e-9):
    """Combine word-block and line-block counts with equal weight (eo777-style rebuild)."""
    P = np.zeros_like(M, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        P[:, wmask] = M[:, wmask] / M[:, wmask].sum(1, keepdims=True)
        P[:, ~wmask] = M[:, ~wmask] / M[:, ~wmask].sum(1, keepdims=True)
    P = np.nan_to_num(P) + eps
    return P / P.sum(1, keepdims=True)


def cmds(D, k=3):
    n = D.shape[0]; J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    share = w[:k] / w[w > 0].sum()
    return V[:, :k] * np.sqrt(np.clip(w[:k], 0, None)), share


# ------------------------------------------------------------------ PERMANOVA
def _proj(X):
    """Orthonormal basis (rank-revealing) of the column space of X."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    r = int((s > 1e-10 * s.max()).sum())
    return U[:, :r]


def dummies(labels):
    labels = np.asarray(labels).astype(str)
    lv = np.unique(labels)
    return (labels[:, None] == lv[None, :]).astype(float)


def gower(D):
    n = D.shape[0]; J = np.eye(n) - 1.0 / n
    return -0.5 * J @ (D ** 2) @ J


def permanova_seq(G, terms):
    """Sequential SS for a list of (name, label_array). Intercept implicit (G is centred).
    Returns dict name -> (SS, df), plus 'resid' and 'total'."""
    n = G.shape[0]
    X = np.ones((n, 1)); prevQ = _proj(X); prev_ss = np.trace(prevQ.T @ G @ prevQ); prev_r = prevQ.shape[1]
    out = {}
    for name, lab in terms:
        X = np.hstack([X, dummies(lab)])
        Q = _proj(X); ss = np.trace(Q.T @ G @ Q)
        out[name] = (ss - prev_ss, Q.shape[1] - prev_r)
        prev_ss, prev_r = ss, Q.shape[1]
    total = np.trace(G)
    out["resid"] = (total - prev_ss, n - prev_r)
    out["total"] = (total, n - 1)
    return out


# ------------------------------------------------------------------ correspondence analysis
def ca(N):
    N = np.asarray(N, float); n = N.sum(); P = N / n
    r = P.sum(1); c = P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    rows = U / np.sqrt(r)[:, None] * s[None, :]
    cols = Vt.T / np.sqrt(c)[:, None] * s[None, :]
    return rows, cols, s ** 2


def _detrend(x, ref, nseg=26):
    """Detrending by segments: subtract smoothed segment means of x along ref (Hill & Gauch 1980)."""
    edges = np.linspace(ref.min(), ref.max() + 1e-12, nseg + 1)
    seg = np.clip(np.digitize(ref, edges) - 1, 0, nseg - 1)
    means = np.array([x[seg == k].mean() if (seg == k).any() else np.nan for k in range(nseg)])
    # fill empty segments by interpolation, then 1-2-1 smoothing (3 passes)
    idx = np.arange(nseg); ok = ~np.isnan(means)
    means = np.interp(idx, idx[ok], means[ok])
    for _ in range(3):
        means = np.convolve(np.r_[means[0], means, means[-1]], [0.25, 0.5, 0.25], "valid")
    return x - means[seg]


def _rescale(site, N):
    """Hill-style rescaling so the mean within-site SD of species scores is 1 (simplified, global)."""
    rs = N.sum(1); cs = N.sum(0)
    sp = (N.T @ site) / np.where(cs > 0, cs, 1)
    var = np.array([np.sum(N[i] * (sp - site[i]) ** 2) / rs[i] for i in range(N.shape[0])])
    sd = np.sqrt(np.average(var, weights=rs))
    return site / sd


def dca(N, iters=200, nseg=26, seed=0):
    """Returns site scores for 2 axes, eigenvalues, axis-1 gradient length (SD units)."""
    N = np.asarray(N, float); rs = N.sum(1); cs = N.sum(0)
    rng = np.random.default_rng(seed)
    axes, lams = [], []
    for k in range(2):
        x = rng.normal(size=N.shape[0])
        lam = 0.0
        for _ in range(iters):
            sp = (N.T @ x) / cs                      # species scores
            xn = (N @ sp) / rs                        # site scores (reciprocal averaging)
            xn = xn - np.average(xn, weights=rs)
            for a in axes:                            # orthogonalise / detrend against previous axes
                xn = _detrend(xn, a, nseg)
                xn = xn - np.average(xn, weights=rs)
            num = np.sqrt(np.average(xn ** 2, weights=rs)); den = np.sqrt(np.average(x ** 2, weights=rs))
            lam = num / den if den > 0 else 0.0
            x = xn / num
        axes.append(x.copy()); lams.append(lam ** 1)
    # eigenvalue in RA convention: lambda = (dispersion ratio)^... the converged ratio IS lambda for RA
    ax1 = _rescale(axes[0] - axes[0].min(), N)
    ax2 = _rescale(axes[1] - axes[1].min(), N)
    return np.c_[ax1, ax2], np.array(lams), float(ax1.max() - ax1.min())


def quad_r2(x, y):
    A = np.c_[np.ones_like(x), x, x ** 2]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - (y - A @ b).var() / y.var()


def spectral_order(D):
    """Fiedler-vector seriation on similarity 1 - D/max(D)."""
    S = 1 - D / D.max(); np.fill_diagonal(S, 0)
    L = np.diag(S.sum(1)) - S
    w, V = np.linalg.eigh(L)
    return V[:, 1]


# ------------------------------------------------------------------ tree-likeness
def delta_mean(D, quartets=None):
    n = D.shape[0]
    if quartets is None:
        from itertools import combinations
        quartets = np.array(list(combinations(range(n), 4)))
    a, b, c, d = quartets.T
    s = np.c_[D[a, b] + D[c, d], D[a, c] + D[b, d], D[a, d] + D[b, c]]
    s.sort(1)
    m3, m2, m1 = s[:, 0], s[:, 1], s[:, 2]
    den = m1 - m3
    delta = np.where(den > 1e-15, (m1 - m2) / np.where(den > 1e-15, den, 1), 0.0)
    return float(delta.mean())


# ------------------------------------------------------------------ compression distance
def _c(b, kind):
    return len(zlib.compress(b, 9)) if kind == "zlib" else len(bz2.compress(b, 9))


def ncd_matrix(strings, kind="zlib"):
    bs = [s.encode() for s in strings]; n = len(bs)
    C = np.array([_c(b, kind) for b in bs], float)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cxy = (_c(bs[i] + b"\n" + bs[j], kind) + _c(bs[j] + b"\n" + bs[i], kind)) / 2
            D[i, j] = D[j, i] = (cxy - min(C[i], C[j])) / max(C[i], C[j])
    return D


# ------------------------------------------------------------------ restricted permutation
def leaf_permute(sheet, leaf_id, stratum, rng):
    """Shuffle sheet labels among LEAVES within strata; every page of a leaf moves together."""
    sheet = np.asarray(sheet); leaf_id = np.asarray(leaf_id); stratum = np.asarray(stratum)
    leaves = np.unique(leaf_id)
    leaf_sheet = {l: sheet[leaf_id == l][0] for l in leaves}
    leaf_str = {l: stratum[leaf_id == l][0] for l in leaves}
    new = {}
    for s in set(leaf_str.values()):
        ls = [l for l in leaves if leaf_str[l] == s]
        perm = rng.permutation(len(ls))
        for l, p in zip(ls, perm):
            new[l] = leaf_sheet[ls[p]]
    return np.array([new[l] for l in leaf_id])


# ------------------------------------------------------------------ MRM (regression on distance matrices)
def mrm_fit(y, X):
    """OLS of pairwise response on pairwise predictors; returns standardised betas (y and X z-scored) and R2."""
    ys = (y - y.mean()) / y.std()
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    A = np.c_[np.ones(len(ys)), Xs]
    b, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r2 = 1 - ((ys - A @ b) ** 2).sum() / (ys ** 2).sum()
    return b[1:], r2


# ------------------------------------------------------------------ variance components (REML)
def fit_vc(y, comps, reml=True):
    """Crossed random intercepts via statsmodels MixedLM with one dummy group.
    comps: dict name -> label array. Returns dict name -> variance, resid variance, llf."""
    import pandas as pd, warnings
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"y": y, "g": 1})
    vc = {}
    for k, v in comps.items():
        df[k] = np.asarray(v).astype(str)
        vc[k] = f"0 + C({k})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y ~ 1", df, groups="g", re_formula="0", vc_formula=vc)
        r = m.fit(reml=reml, method=["lbfgs", "powell"], maxiter=2000)
    names = list(comps.keys())
    # statsmodels orders vcomp by sorted vc names
    vcomp = dict(zip(sorted(names), r.vcomp))
    return {k: float(vcomp[k]) for k in names}, float(r.scale), float(r.llf)
