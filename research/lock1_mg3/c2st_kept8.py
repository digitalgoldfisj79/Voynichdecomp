"""Secondary (not a pre-registered criterion): C2ST on the kept-8 feature set, real held-out text vs generated text.
Kept-8 = {wl_mean, wl_std, wl_autocorr, ttr, hapax, H1, H2, chardist_max}; 120-token non-overlapping chunks;
StandardScaler + LogisticRegression; 5-fold stratified CV. Feature definitions are MY conventions (see memory note:
exact reproduction of the published C2ST column is NOT established).
usage: python3 c2st_kept8.py ARM NREP     (ARM in MG1 | A2 | primary)
"""
import collections as C, json, math, pickle, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joint_model as J

def feats(toks):
    wl = np.array([len(t) for t in toks], float); c = C.Counter(toks); s = ''.join(toks); cc = C.Counter(s); n = len(s)
    p1 = np.array(list(cc.values())) / n; H1 = -(p1 * np.log2(p1)).sum()
    bg = C.Counter(zip(s, s[1:])); p2 = np.array(list(bg.values())) / max(1, n - 1); H2 = -(p2 * np.log2(p2)).sum() - H1
    ac = np.corrcoef(wl[:-1], wl[1:])[0, 1] if wl.std() > 0 else 0.
    return [wl.mean(), wl.std(), ac, len(c) / len(toks), sum(v == 1 for v in c.values()) / len(toks), H1, H2, p1.max()]

def chunks(lines, size=120):
    ts = [t for r in lines for t in r['tokens']]; return [feats(ts[i:i + size]) for i in range(0, len(ts) - size + 1, size)]

def auc(Xa, Xb, seed=0):
    X = np.array(Xa + Xb); y = np.array([0] * len(Xa) + [1] * len(Xb))
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    s = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='roc_auc'); return float(s.mean()), float(s.std())

if __name__ == '__main__':
    arm, nrep = sys.argv[1], int(sys.argv[2])
    rs = J.load_lines(__import__('os').environ.get('MG_LAYER', 'ZLZI')); order = {(r['folio'], r['line']): i for i, r in enumerate(rs)}
    real = chunks(rs)
    ev = [r for r in rs if J.fnum(r['folio']) % 2 == 0]; od = [r for r in rs if J.fnum(r['folio']) % 2 == 1]
    gate = auc(chunks(ev), chunks(od))
    sub = '' if arm == 'primary' else arm; out = []
    for rep in range(nrep):
        lines = []
        for f in range(5): lines += pickle.load(open(J.OUT / sub / f'gen_fold{f}_rep{rep:03d}.pkl', 'rb'))
        lines.sort(key=lambda r: order[r['folio'], r['line']]); out.append(auc(real, chunks(lines), seed=rep))
    res = dict(arm=arm, real_vs_real_gate=gate, auc=[a for a, _ in out], auc_mean=float(np.mean([a for a, _ in out])),
               auc_sd_across_reps=float(np.std([a for a, _ in out], ddof=1)) if nrep > 1 else None, n_real_chunks=len(real))
    print(json.dumps(res)); json.dump(res, open(J.ROOT.parent / f'c2st_{arm}.json', 'w'), indent=1)