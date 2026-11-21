"""Glyph splitter library: word box + EVA string -> per-character horizontal bands.
Pure functions; no file access. Frozen copy is hashed before the held-out run.
"""
import numpy as np

CHARS = list("acdefghiklmnopqrstvxyz")

def fit_width_table(words, widths, scales, intercept=True):
    """Non-negative least squares: width/scale = sum_c count_c * w_c (+ pad). Returns (dict char->w, pad)."""
    idx = {c: i for i, c in enumerate(CHARS)}
    A = np.zeros((len(words), len(CHARS) + (1 if intercept else 0)))
    for r, w in enumerate(words):
        for ch in w:
            if ch in idx: A[r, idx[ch]] += 1
        if intercept: A[r, -1] = 1
    b = np.asarray(widths, float) / np.asarray(scales, float)
    try:
        from scipy.optimize import nnls
        x, _ = nnls(A, b)
    except Exception:
        x = np.full(A.shape[1], 0.3); L = np.linalg.norm(A, 2) ** 2
        for _ in range(3000): x = np.maximum(0, x - (A.T @ (A @ x - b)) / L)
    tab = {c: float(x[i]) for c, i in idx.items()}
    return tab, (float(x[-1]) if intercept else 0.0)

def trim_extent(profile, tau):
    """ink extent [a, b) in profile columns; tau=None -> whole box."""
    n = len(profile)
    if tau is None or n < 3 or profile.max() <= 0: return 0, n
    on = np.where(profile > tau * profile.max())[0]
    return (int(on[0]), int(on[-1]) + 1) if len(on) else (0, n)

def prior_cuts(word, a, b, table=None, pad=0.0):
    """cut positions (len(word)+1 edges) in columns between a and b. table=None -> equal split."""
    n = len(word)
    if n == 0: return np.array([a, b], float)
    w = np.array([1.0 if table is None else max(table.get(ch, 0.0), 1e-3) for ch in word], float)
    tot = w.sum() + (pad if table is not None else 0.0)
    left = (pad / 2 if table is not None else 0.0) / tot
    edges = left + np.concatenate([[0.0], np.cumsum(w)]) / tot
    edges[0], edges[-1] = 0.0, 1.0          # outer bands absorb the padding
    return a + edges * (b - a)

def snap_cuts(profile, edges, lam, min_frac=0.35):
    """Move interior cuts to ink minima near the priors (dynamic programming, monotone)."""
    n = len(edges) - 1
    if n <= 1: return edges.copy()
    a, b = edges[0], edges[-1]; W = len(profile)
    k = np.ones(3) / 3; sm = np.convolve(profile, k, mode="same")
    ref = np.median(sm[sm > 0]) if (sm > 0).any() else 1.0
    ink = sm / max(ref, 1e-9)
    widths = np.diff(edges); s = max(float(np.mean(widths)), 1.0)
    cols = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
    if len(cols) < n + 1: return edges.copy()
    cols = np.clip(cols, 0, W - 1)
    best = None; back = []
    for i in range(1, n):
        cost = ink[cols] + lam * ((cols - edges[i]) / s) ** 2
        if best is None:
            ok = cols - a >= min_frac * widths[0]
            cur = np.where(ok, cost, np.inf); back.append(np.full(len(cols), -1))
        else:
            gap = min_frac * widths[i - 1]
            cur = np.full(len(cols), np.inf); bk = np.full(len(cols), -1)
            run_best, run_idx, j = np.inf, -1, 0
            for t in range(len(cols)):
                while j < len(cols) and cols[j] <= cols[t] - gap:
                    if best[j] < run_best: run_best, run_idx = best[j], j
                    j += 1
                if run_idx >= 0: cur[t] = run_best + cost[t]; bk[t] = run_idx
            back.append(bk)
        best = cur
    ok = b - cols >= min_frac * widths[-1]
    best = np.where(ok, best, np.inf)
    if not np.isfinite(best).any(): return edges.copy()
    t = int(np.argmin(best)); cuts = []
    for i in range(n - 1, 0, -1):
        cuts.append(float(cols[t])); t = int(back[i - 1][t]) if i > 1 else t
    out = edges.copy(); out[1:-1] = cuts[::-1]
    return out

def split(word, profile, mode="equal", table=None, pad=0.0, tau=None, lam=2.0):
    """Return edges (len(word)+1) in profile columns. mode: equal | table | snap."""
    a, b = trim_extent(profile, tau)
    edges = prior_cuts(word, a, b, table if mode in ("table", "snap") else None, pad)
    if mode == "snap": edges = snap_cuts(profile, edges, lam)
    return edges
