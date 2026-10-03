"""MG3 Lock-1 test: MG2 + one static empirical-Bayes folio palette draw.

The MG2 fitted model is unchanged. The only new generative module is a training-only,
folio-static score adjustment over ORIGINAL training-vocabulary candidates. No coined
or novel token is directly reweighted by the palette. See MG3_PREREG_20261003.md.
"""
from __future__ import annotations
import collections as C, json, math, os, pickle, sys, time
import numpy as np
import joint_model as J
import mg2 as M
from joint_model import A, AI, atoms, fnum

OUTM = J.OUT / 'MG3'; OUTM.mkdir(exist_ok=True)
M.OUTM = OUTM  # exact MG2 events/fits, isolated under MG3

class PaletteGen(M.Gen):
    def __init__(self, f):
        super().__init__(f)
        # Build outer-training folio bags aligned to the MG2 training vocabulary.
        pages = {}
        meta = {}
        for r in self.train:
            pages.setdefault(r['folio'], C.Counter()).update(r['tokens'])
            meta[r['folio']] = (r['section'], r['currier'])
        self._page_counts = {}
        self._page_n = {}
        for fol, cc in pages.items():
            v = np.zeros(self.V, np.float64)
            for t, n in cc.items():
                j = self.cnt['vi'].get(t)
                if j is not None: v[j] = n
            self._page_counts[fol] = v
            self._page_n[fol] = int(v.sum())
        self._groups_sc = C.defaultdict(list); self._groups_s = C.defaultdict(list)
        for fol, (s, c) in meta.items():
            self._groups_sc[(s, c)].append(fol); self._groups_s[s].append(fol)
        self._all_donors = sorted(pages)
        self._pg = self.cnt['cg'].astype(np.float64)
        self._pg /= self._pg.sum()
        self._test_folios = {r['folio'] for r in self.test}
        assert not (set(self._all_donors) & self._test_folios), 'outer-fold donor leakage'

    def palette(self, r, rng):
        pool = list(self._groups_sc.get((r['section'], r['currier']), ()))
        scope = 'section_currier'
        if len(pool) < 4:
            pool = list(self._groups_s.get(r['section'], ())); scope = 'section'
        if len(pool) < 4:
            pool = list(self._all_donors); scope = 'all'
        assert pool and not (set(pool) & self._test_folios)
        donor = pool[int(rng.integers(len(pool)))]
        lens = np.array([self._page_n[x] for x in pool], np.float64)
        m = max(1.0, float(np.median(lens)))
        cG = np.zeros(self.V, np.float64)
        NG = 0.0
        for x in pool:
            cG += self._page_counts[x]; NG += self._page_n[x]
        p0 = (cG + m * self._pg) / (NG + m)
        cD = self._page_counts[donor]; ND = float(self._page_n[donor])
        pD = (cD + m * p0) / (ND + m)
        # strictly positive because p0 > 0 for every training-vocabulary type
        adj = np.log(pD) - np.log(p0)
        assert np.isfinite(adj).all()
        return adj.astype(np.float32), donor, scope, m

    def run(self, seed):
        rng = np.random.default_rng(seed); W = self.W; V = self.V; vocab = self.cnt['vocab']; vi = self.cnt['vi']
        coined = []; cvi = {}; chead = []; cfin = []; clb = []; cuse = []; gen_lines = []
        test = self.test
        pageh = {}; prevby = {}; secc = C.defaultdict(C.Counter); curf = None; snapc = {}
        pal = np.zeros(V, np.float32); donors = []
        for r in test:
            f_ = r['folio']
            if f_ != curf:
                if curf is not None: secc[cursec].update(pageh[curf])
                curf = f_; cursec = r['section']; snapc = {}
                snapv = {}
                for u, m0 in secc[cursec].items():
                    if u in vi: snapv[vi[u]] = math.log1p(m0)
                    else: snapc[u] = math.log1p(m0)
                snap = np.zeros(V)
                for j_, v_ in snapv.items(): snap[j_] = v_
                pal, donor, scope, medn = self.palette(r, rng)
                donors.append((f_, donor, scope, medn))
            ph = pageh.setdefault(f_, []); pr = prevby.get(f_); n = len(r['tokens'])
            carry = bool(pr and r['para'] is not None and pr['para'] == r['para'] and r['line'] == pr['line'] + 1)
            regime = 0 if not carry else (1 if pr['pl'] == 0 else 2); popen = AI[atoms(pr['tokens'][0])[0]] if carry else M.NONE
            lbf = AI[atoms(pr['tokens'][-1])[-1]] if (pr and r['line'] == pr['line'] + 1) else M.NONE
            sk = 1 if carry else (2 if r['para'] is None else 0); k_sc = M.sc_index(r['section'], r['currier'])
            lineh = []; prevED = 5
            for i in range(n):
                ctx = dict(i=i, sec=r['section'], pb=M.posbin(i, n), last=int(i == n - 1), prevED=prevED, lineh=lineh, pageh=ph)
                nC = len(coined); hd = np.concatenate([self.head, np.array(chead, np.int16)]); fn = np.concatenate([self.fin, np.array(cfin, np.int16)])
                if i == 0:
                    hw = W['O'][regime * (A + 1) + popen] + W['SH'][sk] + W['LB'][lbf]
                    s = np.concatenate([self.csS[k_sc], self.coin_scores(W['wS'], W['coinS'][0], cuse)]); wc = W['cS'][0]; rw = W['rS']
                else:
                    pv = lineh[-1]; fp = AI[atoms(pv)[-1]]; hw = W['J'][fp] + W['P'][ctx['pb']]
                    s = np.concatenate([self.csT[k_sc], self.coin_scores(W['wT'], W['coinT'][0], cuse)]); wc = W['cT'][0]
                    route = self.cnt['route'].get(pv, 4); er = W['E'][prevED] + W['R'][route]
                    d = self.dist(pv)
                    if nC: d = np.concatenate([d, np.array([min(4, J.ed(pv, c)) for c in coined], np.uint8)])
                    s = s + er[d]; rw = W['rT']
                s = s + hw[hd] + np.concatenate([self.static, W['Lb'][np.array(clb, np.int16)]]) + (self.Lf_all(fn) if ctx['last'] else 0)
                s[:V] += wc * snap
                if nC: s[V:] += wc * np.array([snapc.get(c, 0.) for c in coined])
                # MG3's only new term: one static, training-only folio palette over original vocabulary.
                s[:V] += pal
                col = {}
                for u in set(lineh) | set(ph):
                    if u in vi: col[u] = vi[u]
                    elif u in cvi: col[u] = V + cvi[u]
                for c_, ft in M.recency(dict(lineh=lineh, pageh=ph), col): s[c_] += rw[ft]
                pn = 1 / (1 + math.exp(-(self.ni + M.nov_x(dict(ctx, i=i)) @ self.nc)))
                tok = None
                if rng.random() < pn:
                    iw = np.exp(hw - hw.max())
                    for _ in range(200):
                        cand = self.atom.draw(r['section'], rng, initial_weights=iw)
                        if cand and cand not in vi and cand not in cvi: tok = cand; break
                if tok is None:
                    s = s - s.max(); p = np.exp(s); p /= p.sum(); j = int(rng.choice(len(p), p=p))
                    tok = vocab[j] if j < V else coined[j - V]
                if tok not in vi:
                    if tok not in cvi:
                        cvi[tok] = len(coined); coined.append(tok); a_ = atoms(tok); chead.append(AI[a_[0]]); cfin.append(AI[a_[-1]]); clb.append(M.lenb(tok)); cuse.append(0)
                    cuse[cvi[tok]] += 1
                if i >= 1: prevED = min(4, J.ed(lineh[-1], tok))
                lineh.append(tok); ph.append(tok)
            nr = {k: v for k, v in r.items() if not k.startswith('_')}; nr['tokens'] = list(lineh); gen_lines.append(nr)
            prevby[f_] = dict(para=r['para'], line=r['line'], tokens=list(lineh), pl=(pr['pl'] + 1 if carry else 0))
        return gen_lines, dict(coined=len(coined), donor_n=len(donors), donor_unique=len({x[1] for x in donors}), donors=donors)

def stage_gen(f, r0, r1):
    g = None
    for rep in range(r0, r1):
        p = OUTM / f'gen_fold{f}_rep{rep:03d}.pkl'
        if p.exists(): continue
        if g is None: g = PaletteGen(f)
        t = time.time(); lines, st = g.run(503610030 + rep * 100 + f); J.atomic(lines, p)
        # keep logs compact; donor mapping is saved separately and not printed in full
        donors = st.pop('donors'); J.atomic(donors, OUTM / f'donors_fold{f}_rep{rep:03d}.pkl')
        print(json.dumps(dict(stage='gen', fold=f, rep=rep, secs=round(time.time()-t,1), **st)), flush=True)

def folio_seen_metrics(lines, real_rs):
    """Pooled fold-held-out metrics using real other-four-fold vocab as the seen reference.
    Mplus searches only earlier non-identical TRAINING-SEEN tokens on that folio.
    TTR is micro-averaged distinct train-seen types per folio / train-seen occurrences.
    """
    byfold_train = []
    for f in range(5):
        v = sorted({t for r in real_rs if r['fold'] != f for t in r['tokens']}); byfold_train.append((v, {t:i for i,t in enumerate(v)}))
    bypage = C.defaultdict(list)
    for r in lines: bypage[(r['fold'], r['folio'])].extend(r['tokens'])
    sM = 0.0; nM = 0; uniq_sum = 0; seen_n = 0; per_fold = {}
    for f in range(5):
        vocab, vi = byfold_train[f]; dist = M.Dist(vocab); fs=0.0; fn=0; fu=0; fseen=0
        for (ff, fol), toks in bypage.items():
            if ff != f: continue
            prior = set(); fol_seen = []
            for t in toks:
                j = vi.get(t)
                if j is None: continue
                fol_seen.append(t)
                if prior:
                    d = dist(t)
                    vals = [int(d[k]) for k in prior if k != j]
                    m = min(vals) if vals else 4
                else: m = 4
                fs += m; fn += 1; prior.add(j)
            fu += len(set(fol_seen)); fseen += len(fol_seen)
        per_fold[str(f)] = dict(Mplus=fs/max(fn,1), TTR=fu/max(fseen,1), n=fn)
        sM += fs; nM += fn; uniq_sum += fu; seen_n += fseen
    return dict(folio_seen_Mplus=sM/max(nM,1), folio_seen_TTR=uniq_sum/max(seen_n,1), per_fold=per_fold, n_seen=nM)

def stage_score(nrep):
    from run_joint import diagnostics
    rs = J.load_lines(os.environ.get('MG_LAYER','ZLZI')); obs = diagnostics(rs); obs_extra = folio_seen_metrics(rs, rs)
    order = {(r['folio'], r['line']): i for i,r in enumerate(rs)}
    sims=[]; extras=[]; runaway=[]
    for rep in range(nrep):
        lines=[]
        for f in range(5):
            gl=pickle.load(open(OUTM / f'gen_fold{f}_rep{rep:03d}.pkl','rb')); lines += gl
            tk=[t for r in gl for t in r['tokens']]
            if C.Counter(tk).most_common(1)[0][1] > 0.06*len(tk): runaway.append((rep,f))
        lines.sort(key=lambda r: order[r['folio'],r['line']]); sims.append(diagnostics(lines)); extras.append(folio_seen_metrics(lines,rs))
    keys=[k for k in obs if k!='hard_zero_violation_rate']; X=np.array([[s[k] for k in keys] for s in sims]); mu=X.mean(0); sd=X.std(0,ddof=1); y=np.array([obs[k] for k in keys])
    z=np.divide(y-mu,sd,out=np.zeros_like(y),where=sd>0); simz=np.divide(X-mu,sd,out=np.zeros_like(X),where=sd>0); null95=float(np.quantile(np.abs(simz).max(1),.95))
    ek=['folio_seen_Mplus','folio_seen_TTR']; EX=np.array([[e[k] for k in ek] for e in extras]); emu=EX.mean(0); esd=EX.std(0,ddof=1); ey=np.array([obs_extra[k] for k in ek]); ez=np.divide(ey-emu,esd,out=np.zeros_like(ey),where=esd>0)
    zd=dict(zip(keys,z.tolist())); ezd=dict(zip(ek,ez.tolist()))
    untargeted=['n_types','hapax_type_fraction','hapax_token_fraction','token_length_mean','atom_length_mean','q_kt_tail_CMI','q_kt_opportunities_per_token','irun_terminator_CMI','irun_opportunities_per_token']
    reuse=[k for k in keys if k.startswith('page_reuse_')]
    core_close=(float(np.abs(z).max())<=null95 and all(abs(zd[k])<=3 for k in untargeted) and len(runaway)==0 and all(abs(zd[k])<=3 for k in reuse) and all(abs(ezd[k])<=3 for k in ek))
    pre='FALSIFIED' if (float(np.abs(z).max())>10 or len(runaway)>=3) else ('CORE_CLOSE_PENDING_C2ST' if core_close else 'NOT_CLOSED_PARTIAL')
    res=dict(n_rep=nrep,observed=obs,sim_mean=dict(zip(keys,mu.tolist())),sim_sd=dict(zip(keys,sd.tolist())),z=zd,max_abs_z=float(np.abs(z).max()),argmax=keys[int(np.abs(z).argmax())],null95=null95,hard_zero_sim=[s['hard_zero_violation_rate'] for s in sims],runaway_reps=runaway,
             observed_extra={k:obs_extra[k] for k in ek},sim_extra_mean=dict(zip(ek,emu.tolist())),sim_extra_sd=dict(zip(ek,esd.tolist())),z_extra=ezd,real_extra_per_fold=obs_extra['per_fold'],core_close=core_close,pre_c2st_verdict=pre)
    J.atomic(res,OUTM/'score.pkl'); json.dump(res,open(J.ROOT.parent/'mg3_results.json','w'),indent=1)
    for k in sorted(keys,key=lambda k:-abs(zd[k])): print(f"{k:28s} obs {obs[k]:10.4f} sim {res['sim_mean'][k]:10.4f} sd {res['sim_sd'][k]:9.5f} z {zd[k]:7.2f}")
    for k in ek: print(f"{k:28s} obs {obs_extra[k]:10.5f} sim {res['sim_extra_mean'][k]:10.5f} sd {res['sim_extra_sd'][k]:9.6f} z {ezd[k]:7.2f}")
    print('MG3_PRE_C2ST',json.dumps(dict(verdict=pre,max_abs_z=res['max_abs_z'],argmax=res['argmax'],null95=null95,runaway=len(runaway),extra_z=ezd)))

if __name__=='__main__':
    st=sys.argv[1]
    if st=='events': M.stage_events(int(sys.argv[2]))
    elif st=='fit': M.stage_fit(int(sys.argv[2]))
    elif st=='gen': stage_gen(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))
    elif st=='score': stage_score(int(sys.argv[2]))
