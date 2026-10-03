"""MG2: second attempt (after MG1 failed its pre-registration). Changes vs MG1, fixed before any MG2 run:
 (a) coined types use the SAME count features as vocabulary (log use count, use==1, use==2) -> no separate reuse weight;
 (b) section cache: log1p(count of the token on EARLIER folios of the same section in the stream) with its own weight;
 (c) maxiter 600; (d) runaway replicates counted in scoring.
Pre-registered in RUNNING_RESULTS_lock1_20261003.md (MG2 section).

Stages (each atomic: temp file + fsync + os.replace via joint_model.atomic; skipped if output exists):
  python3 mg1.py events F     -> joint_run/MG1/events_fold{F}.pkl   (inner 7-way cross-fitted event tables)
  python3 mg1.py fit F        -> joint_run/MG1/params_fold{F}.pkl   (L-BFGS, fixed L2)
  python3 mg1.py gen F R0 R1  -> joint_run/MG1/gen_fold{F}_rep{R}.pkl
  python3 mg1.py score        -> ../mg1_results.json
Reuses GPT's bundle: joint_model.load_lines / atoms / AtomModel (training-only fitted, from model_fold{F}.pkl),
run_joint.diagnostics unchanged. Raw-character Levenshtein ED via GPT's native library.
"""
import collections as C, ctypes as ct, json, math, pickle, sys, time
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import joint_model as J
from joint_model import A, AI, atoms, norm, fnum

OUTM = J.OUT / 'MG2'; OUTM.mkdir(exist_ok=True)
SECS = ['HERBAL', 'ASTRO', 'BIO', 'PHARMA', 'RECIPES']; CURS = ['A', 'B', 'UNK']
NSC = len(SECS) * len(CURS)
NONE = A  # "no atom" index for prev-opener / line-break final
LAM_M, LAM_S = 1.0, 0.01
LINE_BANDS = [(1, 1), (2, 2), (3, 6), (7, 10**9)]; PAGE_BANDS = [(1, 5), (6, 12), (13, 32), (33, 64), (65, 10**9)]
NREC = len(LINE_BANDS) + len(PAGE_BANDS)

def sc_index(sec, cur): return SECS.index(sec) * len(CURS) + CURS.index(cur if cur in CURS else 'UNK')
def lenb(t): return min(7, len(atoms(t))) - 1  # 0..6
def posbin(i, n): return min(4, int(5 * i / max(1, n)))

class Dist:
    """Raw-character ED buckets from one source to a fixed list of strings (native)."""
    def __init__(self, strs):
        self.n = len(strs); self.arr = (ct.c_char_p * self.n)(*[s.encode() for s in strs]); self.cache = {}
    def __call__(self, src):
        if src not in self.cache:
            out = np.empty(self.n, np.uint8); J.LIB.dist_many(src.encode(), self.arr, self.n, out.ctypes.data)
            self.cache[src] = np.minimum(out, 4)
        return self.cache[src]

def fit_counts(lines):
    """Counts, vocab and ED routes from a set of training lines (GPT-style K=4 routing)."""
    vocab = sorted({t for r in lines for t in r['tokens']}); vi = {t: i for i, t in enumerate(vocab)}; V = len(vocab)
    cg = np.zeros(V); cs = np.zeros((len(SECS), V)); csc = np.zeros((NSC, V)); cf = np.zeros((len(SECS), V))
    nxt = C.defaultdict(lambda: np.zeros(5)); root = np.full(5, .5)
    for r in lines:
        s = SECS.index(r['section']); k = sc_index(r['section'], r['currier']); ts = r['tokens']
        for i, t in enumerate(ts):
            v = vi[t]; cg[v] += 1; cs[s, v] += 1; csc[k, v] += 1
            if i == 0: cf[s, v] += 1
        for a, b in zip(ts, ts[1:]):
            d = min(4, J.ed(a, b)); nxt[a][d] += 1; root[d] += 1
    root = norm(root); elig = [t for t in vocab if nxt[t].sum() >= 5]
    X = np.array([norm(nxt[t] + 20 * root) for t in elig]); w = np.array([nxt[t].sum() for t in elig])
    km = KMeans(n_clusters=4, random_state=20261003, n_init=20).fit(X, sample_weight=w)
    cen = km.cluster_centers_
    route = {t: int(np.argmin(((norm(nxt.get(t, np.zeros(5)) + 20 * root) - cen) ** 2).sum(1))) for t in vocab}
    return dict(vocab=vocab, vi=vi, cg=cg, cs=cs, csc=csc, cf=cf, route=route)

def cand_static(strs):
    return (np.array([AI[atoms(t)[0]] for t in strs], np.int16), np.array([AI[atoms(t)[-1]] for t in strs], np.int16),
            np.array([lenb(t) for t in strs], np.int16))

def count_feats(cnt, k_sc, ncoin):
    """U x 6 count-feature matrix for a section x currier key; coined rows are zero."""
    s = k_sc // len(CURS); cg = cnt['cg']
    F = np.stack([np.log(cg), np.log1p(cnt['cs'][s]), np.log1p(cnt['csc'][k_sc]), (cg == 1) * 1., (cg == 2) * 1.,
                  np.log1p(cnt['cf'][s])], 1)
    return np.vstack([F, np.zeros((ncoin, 6))]).astype(np.float32)

def walk(lines):
    """Yield per-position context dicts in reading order with real (teacher-forced) history."""
    pageh = {}; prevby = {}
    for r in lines:
        f = r['folio']; ph = pageh.setdefault(f, []); pr = prevby.get(f); ts = r['tokens']; n = len(ts)
        carry = bool(pr and r['para'] is not None and pr['para'] == r['para'] and r['line'] == pr['line'] + 1)
        regime = 0 if not carry else (1 if pr['pl'] == 0 else 2)
        popen = AI[atoms(pr['tokens'][0])[0]] if carry else NONE
        lbf = AI[atoms(pr['tokens'][-1])[-1]] if (pr and r['line'] == pr['line'] + 1) else NONE
        sk = 1 if carry else (2 if r['para'] is None else 0)
        lineh = []; prevED = 5
        for i, t in enumerate(ts):
            ctx = dict(folio=f, i=i, n=n, sec=r['section'], k_sc=sc_index(r['section'], r['currier']), pb=posbin(i, n), last=int(i == n - 1),
                       lineh=list(lineh), pageh=list(ph), regime=regime, popen=popen, lbf=lbf, sk=sk, prevED=prevED,
                       prev=lineh[-1] if i else None, target=t)
            yield ctx
            if i >= 1: prevED = min(4, J.ed(lineh[-1], t))
            lineh.append(t); ph.append(t)
        prevby[f] = dict(para=r['para'], line=r['line'], tokens=ts, pl=(pr['pl'] + 1 if carry else 0))

def recency(ctx, col):
    """[(col, feat)] for candidates in line/page history. Line band if in current line else page band."""
    out = {}; lh = ctx['lineh']; ph = ctx['pageh']; L = len(lh)
    for lag in range(1, L + 1):
        t = lh[-lag]
        if t in col and col[t] not in out:
            out[col[t]] = next(b for b, (lo, hi) in enumerate(LINE_BANDS) if lo <= lag <= hi)
    P = len(ph)
    for idx in range(P - L - 1, -1, -1):
        t = ph[idx]; lag = P - idx
        if t in col and col[t] not in out:
            out[col[t]] = len(LINE_BANDS) + next(b for b, (lo, hi) in enumerate(PAGE_BANDS) if lo <= lag <= hi)
    return list(out.items())

def nov_x(ctx):
    x = np.zeros(1 + 5 + 1 + 5 + 6 + 1); x[0] = 1; x[1 + SECS.index(ctx['sec'])] = 1; x[6] = ctx['i'] == 0
    x[7 + ctx['pb']] = 1; x[12 + (ctx['prevED'] if ctx['i'] else 5)] = 1; x[18] = ctx['last']; return x

# ---------------------------------------------------------------- stage: events
def stage_events(f):
    p = OUTM / f'events_fold{f}.pkl'
    if p.exists(): return
    rs = J.load_lines(__import__('os').environ.get('MG_LAYER', 'ZLZI')); train = [r for r in rs if r['fold'] != f]; splits = []
    for k in range(7):
        itr = [r for r in train if fnum(r['folio']) % 7 != k]; iva = [r for r in train if fnum(r['folio']) % 7 == k]
        cnt = fit_counts(itr); V = len(cnt['vocab'])
        coined = []; seen_new = set()
        for r in iva:
            for t in r['tokens']:
                if t not in cnt['vi'] and t not in seen_new: seen_new.add(t); coined.append(t)
        U = cnt['vocab'] + coined; col = {t: i for i, t in enumerate(U)}; head, fin, lb = cand_static(U)
        dist = Dist(U); use = C.Counter(); avail = 0
        ev = C.defaultdict(list); recs = []; extras = []; Drows = []; nx = []; ny = []
        secc = C.defaultdict(C.Counter); curf = None; curtoks = []; snaps = []
        for e, ctx in enumerate(walk(iva)):
            if ctx['folio'] != curf:
                if curf is not None: secc[cursec].update(curtoks)
                curf = ctx['folio']; cursec = ctx['sec']; curtoks = []
                sv = np.zeros(len(U), np.float32)
                for u, m in secc[cursec].items(): sv[col[u]] = math.log1p(m)
                snaps.append(sv)
            curtoks.append(ctx['target'])
            t = ctx['target']; isnov = t not in cnt['vi'] and use[t] == 0
            nx.append(nov_x(ctx)); ny.append(int(isnov))
            if not isnov:
                ei = len(ev['tgt'])
                ev['tgt'].append(col[t]); ev['kind'].append(int(ctx['i'] == 0)); ev['ksc'].append(ctx['k_sc'])
                ev['pb'].append(ctx['pb']); ev['last'].append(ctx['last']); ev['avail'].append(avail)
                ev['prevED'].append(ctx['prevED']); ev['regime'].append(ctx['regime']); ev['popen'].append(ctx['popen'])
                ev['lbf'].append(ctx['lbf']); ev['sk'].append(ctx['sk']); ev['snap'].append(len(snaps) - 1)
                if ctx['i']:
                    pv = ctx['prev']; ev['fp'].append(AI[atoms(pv)[-1]]); ev['route'].append(cnt['route'].get(pv, 4)); Drows.append(dist(pv))
                else:
                    ev['fp'].append(0); ev['route'].append(4); Drows.append(None)
                for c_, ft in recency(ctx, {u: col[u] for u in set(ctx['lineh'] + ctx['pageh']) if u in col and (col[u] < V or col[u] < V + avail)}):
                    recs.append((ei, c_, ft))
                for u, m in use.items():
                    if m > 1 and u in col and col[u] >= V: extras.append((ei, col[u], math.log(m), float(m == 2)))
            if t not in cnt['vi']:
                if use[t] == 0: avail += 1
                use[t] += 1
        E = {k_: np.array(v) for k_, v in ev.items()}
        isT = E['kind'] == 0
        D = np.zeros((int(isT.sum()), len(U)), np.uint8); ti = np.flatnonzero(isT)
        for j, e in enumerate(ti): D[j] = Drows[e]
        Fc = {k_sc: count_feats(cnt, k_sc, len(coined)) for k_sc in set(E['ksc'].tolist())}
        splits.append(dict(V=V, U=len(U), head=head, fin=fin, lb=lb, E=E, Tidx=ti, D=D, Fc=Fc,
                           rec=np.array(recs, float).reshape(-1, 3), extra=np.array(extras, float).reshape(-1, 4), SNAP=np.array(snaps),
                           nx=np.array(nx), ny=np.array(ny)))
        print(json.dumps(dict(stage='events', fold=f, split=k, V=V, coined=len(coined), events=len(E['tgt']), novel=int(sum(ny)))), flush=True)
    J.atomic(splits, p)

# ---------------------------------------------------------------- parameters
SHAPES = [('wT', (5,)), ('wS', (6,)), ('J', (A, A)), ('E', (6, 5)), ('R', (5, 5)), ('P', (5, A)), ('L', (A,)), ('Lb', (7,)),
          ('rT', (NREC,)), ('rS', (NREC,)), ('coinT', (1,)), ('coinS', (1,)), ('cu', (1,)),
          ('O', (3 * (A + 1), A)), ('SH', (3, A)), ('LB', (A + 1, A)), ('cT', (1,)), ('cS', (1,))]
MATRIX = {'J', 'E', 'R', 'P', 'L', 'O', 'SH', 'LB'}
def unpack(x):
    out = {}; i = 0
    for k, s in SHAPES:
        n = int(np.prod(s)); out[k] = x[i:i + n].reshape(s); i += n
    return out
def pack(d): return np.concatenate([d[k].ravel() for k, _ in SHAPES])
NPAR = sum(int(np.prod(s)) for _, s in SHAPES)
def init():
    x = np.zeros(NPAR); d = unpack(x); d['wT'][0] = 1; d['wS'][0] = 1; return pack(d)

def scores(sp, W, ev, kind):
    """Score matrix (B x U) for a chunk of events ev (indices into sp['E']) of one kind (0 T, 1 S)."""
    E = sp['E']; V = sp['V']; U = sp['U']; head = sp['head']; B = len(ev)
    cs = np.zeros((B, U), np.float32)
    for k_sc in np.unique(E['ksc'][ev]):
        m = E['ksc'][ev] == k_sc; F = sp['Fc'][k_sc]
        cs[m] = (F[:, :5] @ W['wT']) if kind == 0 else (F @ W['wS'])
    cs += W['Lb'][sp['lb']][None, :]
    cs += W['L'][sp['fin']][None, :] * E['last'][ev][:, None]
    if kind == 0:
        cs += W['J'][:, head][E['fp'][ev]] + W['P'][:, head][E['pb'][ev]]
        rows = np.searchsorted(sp['Tidx'], ev); D = sp['D'][rows]
        ER = (W['E'][E['prevED'][ev]] + W['R'][E['route'][ev]]).astype(np.float32)
        cs += ER.ravel()[(np.arange(B, dtype=np.int64)[:, None] * 5 + D)]
    else:
        cs += W['O'][:, head][E['regime'][ev] * (A + 1) + E['popen'][ev]] + W['SH'][:, head][E['sk'][ev]] + W['LB'][:, head][E['lbf'][ev]]
    colidx = np.arange(U)[None, :]; av = E['avail'][ev][:, None]
    coin = (colidx >= V) & (colidx < V + av)
    w_ = W['wT'] if kind == 0 else W['wS']
    cs += coin * ((W['coinT'][0] if kind == 0 else W['coinS'][0]) + w_[3])
    cs += (W['cT'][0] if kind == 0 else W['cS'][0]) * sp['SNAP'][E['snap'][ev]]
    cs[colidx >= V + av] = -np.inf
    return cs

def split_obj(sp, W, chunk=384):
    G = {k: np.zeros_like(v) for k, v in W.items()}; nll = 0.
    if True:
        E = sp['E']; V = sp['V']; head = sp['head']
        if '_grp' not in sp:
            g_ = {}
            for nm, arr in (('h', head), ('f', sp['fin'])):
                o = np.argsort(arr, kind='stable'); u_, st_ = np.unique(arr[o], return_index=True); g_[nm] = (o, u_, st_)
            sp['_grp'] = g_
        LOH = np.zeros((sp['U'], 7)); LOH[np.arange(sp['U']), sp['lb']] = 1
        def grp(R, nm):
            o, u_, st_ = sp['_grp'][nm]; out = np.zeros((R.shape[0], A)); out[:, u_] = np.add.reduceat(R[:, o], st_, axis=1); return out
        rec = sp['rec']; ext = sp['extra']
        for kind in (0, 1):
            idx = np.flatnonzero(E['kind'] == kind)
            for c0 in range(0, len(idx), chunk):
                ev = idx[c0:c0 + chunk]; B = len(ev); pos = {e: j for j, e in enumerate(ev)}
                S = scores(sp, W, ev, kind)
                rw = W['rT'] if kind == 0 else W['rS']
                ck = ('_rc', kind, c0)
                if ck not in sp:
                    rm = np.isin(rec[:, 0], ev) if len(rec) else np.zeros(0, bool); rr_ = rec[rm]
                    rj_ = np.array([pos[int(e)] for e in rr_[:, 0]], int) if len(rr_) else np.zeros(0, int)
                    xm_ = np.isin(ext[:, 0], ev) if len(ext) else np.zeros(0, bool); xx_ = ext[xm_]
                    xj_ = np.array([pos[int(e)] for e in xx_[:, 0]], int) if len(xx_) else np.zeros(0, int)
                    sp[ck] = (rr_, rj_, xx_, xj_)
                rr, rj, xx, xj = sp[ck]
                if len(rr): np.add.at(S, (rj, rr[:, 1].astype(int)), rw[rr[:, 2].astype(int)])
                w_ = W['wT'] if kind == 0 else W['wS']
                if len(xx): np.add.at(S, (xj, xx[:, 1].astype(int)), w_[0] * xx[:, 2] + w_[4] * xx[:, 3] - w_[3])
                S = S.astype(np.float64); mx = S.max(1, keepdims=True); Pm = np.exp(S - mx); Z = Pm.sum(1, keepdims=True); Pm /= Z
                tgt = E['tgt'][ev]; nll -= float((S[np.arange(B), tgt] - mx[:, 0] - np.log(Z[:, 0])).sum())
                R = -Pm; R[np.arange(B), tgt] += 1.  # observed - expected
                for k_sc in np.unique(E['ksc'][ev]):
                    m = E['ksc'][ev] == k_sc; F = sp['Fc'][k_sc]; g = R[m].sum(0) @ F
                    if kind == 0: G['wT'] -= g[:5]
                    else: G['wS'] -= g
                RS = R.sum(0); G['Lb'] -= RS @ LOH
                RF = grp(R, 'f'); G['L'] -= (RF * E['last'][ev][:, None]).sum(0)
                RH = grp(R, 'h')
                if kind == 0:
                    np.subtract.at(G['J'], E['fp'][ev], RH); np.subtract.at(G['P'], E['pb'][ev], RH)
                    rows = np.searchsorted(sp['Tidx'], ev); D = sp['D'][rows]
                    SD = np.bincount((np.arange(B, dtype=np.int64)[:, None] * 5 + D).ravel(), weights=R.ravel(), minlength=5 * B).reshape(B, 5)
                    np.subtract.at(G['E'], E['prevED'][ev], SD); np.subtract.at(G['R'], E['route'][ev], SD)
                else:
                    np.subtract.at(G['O'], E['regime'][ev] * (A + 1) + E['popen'][ev], RH)
                    np.subtract.at(G['SH'], E['sk'][ev], RH); np.subtract.at(G['LB'], E['lbf'][ev], RH)
                cs = R[:, V:]; av = E['avail'][ev][:, None]; cm = np.arange(cs.shape[1])[None, :] < av
                gc = (cs * cm).sum()
                gw = G['wT'] if kind == 0 else G['wS']
                if kind == 0: G['coinT'] -= gc
                else: G['coinS'] -= gc
                gw[3] -= gc
                if len(xx):
                    Rx = R[xj, xx[:, 1].astype(int)]; gw[0] -= (Rx * xx[:, 2]).sum(); gw[4] -= (Rx * xx[:, 3]).sum(); gw[3] += Rx.sum()
                gcache = (R * sp['SNAP'][E['snap'][ev]]).sum()
                if kind == 0: G['cT'] -= gcache
                else: G['cS'] -= gcache
                if len(rr):
                    gr = np.bincount(rr[:, 2].astype(int), weights=R[rj, rr[:, 1].astype(int)], minlength=NREC)
                    if kind == 0: G['rT'] -= gr
                    else: G['rS'] -= gr
    return nll, G

def penalty(W, nll, G):
    for k, v in W.items():
        lam = LAM_M if k in MATRIX else LAM_S
        if k in ('wT', 'wS'):  # penalise departure from the log-count prior (weight 1 on global log count)
            prior = np.zeros_like(v); prior[0] = 1; nll += .5 * lam * ((v - prior) ** 2).sum(); G[k] += lam * (v - prior)
        else:
            nll += .5 * lam * (v ** 2).sum(); G[k] += lam * v
    return nll, pack(G)

def objective(x, splits):
    W = unpack(x); nll = 0.; G = {k: np.zeros_like(v) for k, v in W.items()}
    for sp in splits:
        n_, g_ = split_obj(sp, W); nll += n_
        for k in G: G[k] += g_[k]
    return penalty(W, nll, G)

_SPLITS = None
def _winit(f):
    global _SPLITS; _SPLITS = pickle.load(open(OUTM / f'events_fold{f}.pkl', 'rb'))
def _work(a):
    k, x = a; n_, g_ = split_obj(_SPLITS[k], unpack(x)); return n_, pack(g_)
def objective_par(x, pool, nsplit):
    W = unpack(x); nll = 0.; g = np.zeros_like(x)
    for n_, g_ in pool.map(_work, [(k, x) for k in range(nsplit)], chunksize=1): nll += n_; g += g_
    return penalty(W, nll, unpack(g))

def stage_fit(f, maxiter=600):
    p = OUTM / f'params_fold{f}.pkl'
    if p.exists(): return
    import os; maxiter = int(os.environ.get('MG1_MAXITER', maxiter))
    splits = pickle.load(open(OUTM / f'events_fold{f}.pkl', 'rb'))
    x0 = init()
    t = time.time(); it = [0]
    def cb(xk):
        it[0] += 1
        if it[0] % 10 == 0: print(json.dumps(dict(stage='fit', fold=f, iter=it[0], secs=round(time.time() - t))), flush=True)
    import os, multiprocessing as mp
    nw = int(os.environ.get('MG1_WORKERS', '0'))
    if nw > 1:
        with mp.get_context('fork').Pool(nw, initializer=_winit, initargs=(f,)) as pool:
            res = minimize(objective_par, x0, args=(pool, len(splits)), jac=True, method='L-BFGS-B', callback=cb, options=dict(maxiter=maxiter, maxcor=20))
    else:
        res = minimize(objective, x0, args=(splits,), jac=True, method='L-BFGS-B', callback=cb, options=dict(maxiter=maxiter, maxcor=20))
    nx = np.vstack([s['nx'] for s in splits]); ny = np.concatenate([s['ny'] for s in splits])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(nx, ny)
    nev = sum(len(s['E']['tgt']) for s in splits)
    out = dict(x=res.x, nll=float(res.fun), nll_per_event=float(res.fun) / nev, converged=bool(res.success), message=str(res.message),
               nit=int(res.nit), nov_coef=lr.coef_[0], nov_int=float(lr.intercept_[0]), nov_rate_fit=float(ny.mean()), seconds=time.time() - t)
    J.atomic(out, p); print(json.dumps({k: v for k, v in out.items() if k not in ('x', 'nov_coef')}), flush=True)

# ---------------------------------------------------------------- stage: generation
class Gen:
    def __init__(self, f):
        rs = J.load_lines(__import__('os').environ.get('MG_LAYER', 'ZLZI')); self.train = [r for r in rs if r['fold'] != f]; self.test = [dict(r) for r in rs if r['fold'] == f]
        pr = pickle.load(open(OUTM / f'params_fold{f}.pkl', 'rb')); self.W = unpack(pr['x']); self.nc = pr['nov_coef']; self.ni = pr['nov_int']
        self.cnt = fit_counts(self.train); self.V = len(self.cnt['vocab'])
        mp_ = J.OUT / f'model_fold{f}.pkl'
        self.atom = pickle.load(open(mp_, 'rb')).atom if mp_.exists() else J.AtomModel(self.train)  # verified identical (max diff 0.0)
        self.head, self.fin, self.lb = cand_static(self.cnt['vocab']); self.dist = Dist(self.cnt['vocab'])
        W = self.W; self.Fc = {k: count_feats(self.cnt, k, 0) for k in range(NSC)}
        self.csT = {k: F[:, :5] @ W['wT'] for k, F in self.Fc.items()}; self.csS = {k: F @ W['wS'] for k, F in self.Fc.items()}
        self.static = W['Lb'][self.lb]; self.Lf = W['L'][self.fin]
    def run(self, seed):
        rng = np.random.default_rng(seed); W = self.W; V = self.V; vocab = self.cnt['vocab']; vi = self.cnt['vi']
        coined = []; cvi = {}; chead = []; cfin = []; clb = []; cuse = []; out = []
        test = self.test; gen_lines = []
        pageh = {}; prevby = {}; secc = C.defaultdict(C.Counter); curf = None; snapv = {}; snapc = {}
        for r in test:
            f_ = r['folio']
            if f_ != curf:
                if curf is not None: secc[cursec].update(pageh[curf])
                curf = f_; cursec = r['section']; snapv = {}; snapc = {}
                for u, m in secc[cursec].items():
                    if u in vi: snapv[vi[u]] = math.log1p(m)
                    else: snapc[u] = math.log1p(m)
                snap = np.zeros(V);
                for j_, v_ in snapv.items(): snap[j_] = v_
            ph = pageh.setdefault(f_, []); pr = prevby.get(f_); n = len(r['tokens'])
            carry = bool(pr and r['para'] is not None and pr['para'] == r['para'] and r['line'] == pr['line'] + 1)
            regime = 0 if not carry else (1 if pr['pl'] == 0 else 2); popen = AI[atoms(pr['tokens'][0])[0]] if carry else NONE
            lbf = AI[atoms(pr['tokens'][-1])[-1]] if (pr and r['line'] == pr['line'] + 1) else NONE
            sk = 1 if carry else (2 if r['para'] is None else 0); k_sc = sc_index(r['section'], r['currier'])
            lineh = []; prevED = 5
            for i in range(n):
                ctx = dict(i=i, sec=r['section'], pb=posbin(i, n), last=int(i == n - 1), prevED=prevED, lineh=lineh, pageh=ph)
                nC = len(coined); hd = np.concatenate([self.head, np.array(chead, np.int16)]); fn = np.concatenate([self.fin, np.array(cfin, np.int16)])
                if i == 0:
                    hw = W['O'][regime * (A + 1) + popen] + W['SH'][sk] + W['LB'][lbf]
                    s = np.concatenate([self.csS[k_sc], self.coin_scores(W['wS'], W['coinS'][0], cuse)]); wc = W['cS'][0]
                    rw = W['rS']
                else:
                    pv = lineh[-1]; fp = AI[atoms(pv)[-1]]; hw = W['J'][fp] + W['P'][ctx['pb']]
                    s = np.concatenate([self.csT[k_sc], self.coin_scores(W['wT'], W['coinT'][0], cuse)]); wc = W['cT'][0]
                    route = self.cnt['route'].get(pv, 4); er = W['E'][prevED] + W['R'][route]
                    d = self.dist(pv)
                    if nC: d = np.concatenate([d, np.array([min(4, J.ed(pv, c)) for c in coined], np.uint8)])
                    s = s + er[d]
                    rw = W['rT']
                s = s + hw[hd] + np.concatenate([self.static, W['Lb'][np.array(clb, np.int16)]]) + (self.Lf_all(fn) if ctx['last'] else 0)
                s[:V] += wc * snap
                if nC: s[V:] += wc * np.array([snapc.get(c, 0.) for c in coined])
                col = {}
                for u in set(lineh) | set(ph):
                    if u in vi: col[u] = vi[u]
                    elif u in cvi: col[u] = V + cvi[u]
                for c_, ft in recency(dict(lineh=lineh, pageh=ph), col): s[c_] += rw[ft]
                pn = 1 / (1 + math.exp(-(self.ni + nov_x(dict(ctx, i=i)) @ self.nc)))
                tok = None
                if rng.random() < pn:
                    base_h = self.atom.P(r['section'])[J.INIT, :A]
                    iw = np.exp(hw - hw.max())
                    for _ in range(200):
                        cand = self.atom.draw(r['section'], rng, initial_weights=iw)
                        if cand and cand not in vi and cand not in cvi: tok = cand; break
                if tok is None:
                    s = s - s.max(); p = np.exp(s); p /= p.sum(); j = int(rng.choice(len(p), p=p))
                    tok = vocab[j] if j < V else coined[j - V]
                if tok not in vi:
                    if tok not in cvi:
                        cvi[tok] = len(coined); coined.append(tok); a_ = atoms(tok); chead.append(AI[a_[0]]); cfin.append(AI[a_[-1]]); clb.append(lenb(tok)); cuse.append(0)
                    cuse[cvi[tok]] += 1
                if i >= 1: prevED = min(4, J.ed(lineh[-1], tok))
                lineh.append(tok); ph.append(tok)
            nr = {k: v for k, v in r.items() if not k.startswith('_')}; nr['tokens'] = list(lineh); gen_lines.append(nr)
            prevby[f_] = dict(para=r['para'], line=r['line'], tokens=list(lineh), pl=(pr['pl'] + 1 if carry else 0))
        return gen_lines, dict(coined=len(coined))
    def Lf_all(self, fn): return self.W['L'][fn]
    @staticmethod
    def coin_scores(w, cb, cuse):
        u = np.array(cuse, float)
        if not len(u): return np.zeros(0)
        return cb + w[3] + (u >= 2) * (w[0] * np.log(np.maximum(u, 1)) + w[4] * (u == 2) - w[3])

def stage_gen(f, r0, r1):
    g = None
    for rep in range(r0, r1):
        p = OUTM / f'gen_fold{f}_rep{rep:03d}.pkl'
        if p.exists(): continue
        if g is None: g = Gen(f)
        t = time.time(); lines, st = g.run(502610030 + rep * 100 + f); J.atomic(lines, p)
        print(json.dumps(dict(stage='gen', fold=f, rep=rep, secs=round(time.time() - t, 1), **st)), flush=True)

def stage_score(nrep):
    from run_joint import diagnostics
    rs = J.load_lines(__import__('os').environ.get('MG_LAYER', 'ZLZI')); obs = diagnostics(rs); order = {(r['folio'], r['line']): i for i, r in enumerate(rs)}
    sims = []; runaway = []
    for rep in range(nrep):
        lines = []
        for f in range(5):
            gl = pickle.load(open(OUTM / f'gen_fold{f}_rep{rep:03d}.pkl', 'rb')); lines += gl
            tk = [t for r in gl for t in r['tokens']]
            if C.Counter(tk).most_common(1)[0][1] > 0.06 * len(tk): runaway.append((rep, f))
        lines.sort(key=lambda r: order[r['folio'], r['line']]); sims.append(diagnostics(lines))
    keys = [k for k in obs if k != 'hard_zero_violation_rate']
    X = np.array([[s[k] for k in keys] for s in sims]); mu = X.mean(0); sd = X.std(0, ddof=1); y = np.array([obs[k] for k in keys])
    z = np.divide(y - mu, sd, out=np.zeros_like(y), where=sd > 0); simz = np.divide(X - mu, sd, out=np.zeros_like(X), where=sd > 0)  # zero-variance stats reported, z set 0
    null95 = float(np.quantile(np.abs(simz).max(1), .95))
    res = dict(n_rep=nrep, observed=obs, sim_mean=dict(zip(keys, mu.tolist())), sim_sd=dict(zip(keys, sd.tolist())), z=dict(zip(keys, z.tolist())),
               max_abs_z=float(np.abs(z).max()), argmax=keys[int(np.abs(z).argmax())], null95=null95,
               hard_zero_sim=[s['hard_zero_violation_rate'] for s in sims], runaway_reps=runaway)
    J.atomic(res, OUTM / 'score.pkl'); json.dump(res, open(J.ROOT.parent / 'mg2_results.json', 'w'), indent=1)
    for k in sorted(keys, key=lambda k: -abs(res['z'][k])):
        print(f"{k:28s} obs {obs[k]:10.4f} sim {res['sim_mean'][k]:10.4f} sd {res['sim_sd'][k]:9.5f} z {res['z'][k]:7.2f}")
    print('max|z|', res['max_abs_z'], res['argmax'], 'null95', null95, 'runaway', runaway)

if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'events': stage_events(int(sys.argv[2]))
    elif st == 'fit': stage_fit(int(sys.argv[2]))
    elif st == 'gen': stage_gen(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif st == 'score': stage_score(int(sys.argv[2]))