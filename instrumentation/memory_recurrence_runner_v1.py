#!/usr/bin/env python3
import argparse, collections, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

EXPECTED_MANIFEST_SHA = '62655854117793168d46bfb05b548d5993f28dd613301d859fd666504a3b51c0'
SOURCE_LAM = {0:.03,1:.03,2:.03,3:.02,4:.03}
SOURCE_CODE = {
    'working_set_only':1,'r64_exact':2,'refractory_exact':3,'line_reset_r64':4,'family_persistence':5,
    'gross_page_shuffle':11,'gross_family_shuffle':12,'line_opener_unknown':13,
}
SPLIT_NS = {'development':410,'calibration':411,'validation':412,'control':413,'power':414}
FEATURE_NAMES = [
    'exact_lag1_excess','exact_lag2_5_excess','exact_lag6_16_excess','exact_lag17_64_excess',
    'family_lag1_excess','family_lag2_5_excess','family_lag6_16_excess','family_lag17_64_excess',
    'ed1_lag1_excess','ed1_lag2_5_excess','ed1_lag6_16_excess','ed1_lag17_64_excess',
    'exact_cross_minus_interior_excess','ed1_cross_minus_interior_excess',
    'adjacent_page_exact_jaccard','adjacent_page_family_jaccard','bif_minus_cross_exact_jaccard',
    'page_newtype_slope','gain_region','gain_order'
]
BUCKETS = [(1,1),(2,5),(6,16),(17,64)]


def load_t0b():
    p=os.environ.get('T0B_RUNNER','/workspace/t0b/t0b_runner.py')
    spec=importlib.util.spec_from_file_location('t0b_runner',p)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def verify_manifest(path):
    b=Path(path).read_bytes(); h=hashlib.sha256(b).hexdigest()
    if h != EXPECTED_MANIFEST_SHA:
        raise SystemExit(f'MANIFEST_HASH_MISMATCH {h} != {EXPECTED_MANIFEST_SHA}')
    return h


def seed(split,source,trial,fold,slot=0):
    ns=SPLIT_NS[split]; sc=SOURCE_CODE.get(source,99)
    return int(ns*1_000_000 + sc*100_000 + trial*1_000 + fold*100 + slot)


def collapse_repeats(t):
    if not t:return t
    out=[t[0]]
    for c in t[1:]:
        if c != out[-1]: out.append(c)
    return ''.join(out)


def transform_rows(rows, rep, s2):
    if rep=='R0_canonical_tokens':
        return rows
    if rep!='R1_repeated_glyph_collapse': raise ValueError(rep)
    out=[]
    for r in rows:
        q=dict(r); q['token']=collapse_repeats(r['token']); q['family']=s2.famof(q['token']); out.append(q)
    return out


def make_vocab(rows, s2):
    toks=sorted({r['token'] for r in rows}); tid={t:i for i,t in enumerate(toks)}
    fams=sorted({s2.famof(t) for t in toks}); fid={f:i for i,f in enumerate(fams)}
    fam_of=np.array([fid[s2.famof(t)] for t in toks],dtype=np.int32)
    return toks,tid,fam_of


def ed1_edge_codes(toks):
    V=len(toks); idx={t:i for i,t in enumerate(toks)}; edges=set()
    groups=collections.defaultdict(list)
    for i,t in enumerate(toks):
        for k in range(len(t)):
            groups[(len(t),k,t[:k]+t[k+1:])].append(i)
    for ids in groups.values():
        if len(ids)>1:
            for a in ids:
                for b in ids:
                    if a!=b: edges.add(a*V+b)
    for j,t in enumerate(toks):
        if len(t)<=1: continue
        seen=set()
        for k in range(len(t)):
            s=t[:k]+t[k+1:]
            if s in seen: continue
            seen.add(s); i=idx.get(s)
            if i is not None and i!=j:
                edges.add(i*V+j); edges.add(j*V+i)
    return np.array(sorted(edges),dtype=np.int64)


def ed1_hits(a,b,V,edges):
    if edges.size==0:return 0
    codes=(a.astype(np.int64)*V+b.astype(np.int64)).ravel()
    k=np.searchsorted(edges,codes)
    ok=(k<edges.size)
    if np.any(ok):
        kk=k[ok]; cc=codes[ok]; ok2=np.zeros_like(ok); ok2[np.flatnonzero(ok)]=edges[kk]==cc; return int(ok2.sum())
    return 0


def rel_counts(a,b,fam_of,V,edges):
    return (int(np.sum(a==b)), int(np.sum(fam_of[a]==fam_of[b])), ed1_hits(a,b,V,edges), int(a.size))


def smoothed_rate(h,n): return (float(h)+0.5)/(float(n)+1.0)
def log_excess(h,n,nh,nn): return math.log2(smoothed_rate(h,n)/smoothed_rate(nh,nn))


def page_sequences(rows, tid):
    by=collections.OrderedDict()
    for r in rows:
        by.setdefault(r['page'], {'tok':[],'line':[],'bif':r['bifolium'],'first_eid':r['eid']})
        by[r['page']]['tok'].append(tid[r['token']]); by[r['page']]['line'].append(r['line'])
        by[r['page']]['first_eid']=min(by[r['page']]['first_eid'],r['eid'])
    for v in by.values():
        v['tok']=np.asarray(v['tok'],dtype=np.int32); v['line']=np.asarray(v['line'])
    return by


def recurrence_features(rows, split, source, trial, fold, rep, s2):
    toks,tid,fam_of=make_vocab(rows,s2); V=len(toks); edges=ed1_edge_codes(toks); pages=page_sequences(rows,tid)
    obs=np.zeros((4,4),float)
    nul=np.zeros((4,4),float)
    bo=np.zeros((2,3),float); bn=np.zeros((2,3),float)
    for pi,(p,v) in enumerate(pages.items()):
        x=v['tok']; ln=v['line']; n=len(x)
        if n<2: continue
        for bi,(lo,hi) in enumerate(BUCKETS):
            for d in range(lo,min(hi,n-1)+1):
                e,f,ed,op=rel_counts(x[:-d],x[d:],fam_of,V,edges); obs[bi]+=e,f,ed,op
        same=ln[:-1]==ln[1:]
        for ci,mask in enumerate([same,~same]):
            if np.any(mask):
                e,f,ed,op=rel_counts(x[:-1][mask],x[1:][mask],fam_of,V,edges); bo[ci]+=e,ed,op
        rng=np.random.default_rng(seed(split,source,trial,fold,50+pi))
        P=np.stack([rng.permutation(n) for _ in range(64)],axis=0)
        X=x[P]
        for bi,(lo,hi) in enumerate(BUCKETS):
            for d in range(lo,min(hi,n-1)+1):
                a=X[:,:-d]; b=X[:,d:]
                e=int(np.sum(a==b)); f=int(np.sum(fam_of[a]==fam_of[b])); ed=ed1_hits(a,b,V,edges); op=int(a.size)
                nul[bi]+=e,f,ed,op
        for ci,mask in enumerate([same,~same]):
            if np.any(mask):
                a=X[:,:-1][:,mask]; b=X[:,1:][:,mask]
                e=int(np.sum(a==b)); ed=ed1_hits(a,b,V,edges); op=int(a.size); bn[ci]+=e,ed,op
    feats=[]
    for channel in range(3):
        for bi in range(4):
            feats.append(log_excess(obs[bi,channel],obs[bi,3],nul[bi,channel]/64.0,nul[bi,3]/64.0))
    exact_int=log_excess(bo[0,0],bo[0,2],bn[0,0]/64.0,bn[0,2]/64.0)
    exact_cross=log_excess(bo[1,0],bo[1,2],bn[1,0]/64.0,bn[1,2]/64.0)
    ed_int=log_excess(bo[0,1],bo[0,2],bn[0,1]/64.0,bn[0,2]/64.0)
    ed_cross=log_excess(bo[1,1],bo[1,2],bn[1,1]/64.0,bn[1,2]/64.0)
    feats += [exact_cross-exact_int, ed_cross-ed_int]
    pitems=sorted(pages.items(), key=lambda kv: kv[1]['first_eid'])
    sets_tok={p:set(v['tok'].tolist()) for p,v in pitems}; sets_fam={p:set(fam_of[list(sets_tok[p])].tolist()) if sets_tok[p] else set() for p,v in pitems}
    def jac(a,b):
        u=a|b; return len(a&b)/len(u) if u else 0.0
    adj_t=[];adj_f=[]
    full_pages=[]; seen=set()
    for r in s2.ROWS:
        if r['page'] not in seen: seen.add(r['page']); full_pages.append(r['page'])
    rank={p:i for i,p in enumerate(full_pages)}
    for (p,_),(q,__) in zip(pitems,pitems[1:]):
        if abs(rank.get(p,-999)-rank.get(q,-999))==1:
            adj_t.append(jac(sets_tok[p],sets_tok[q]));adj_f.append(jac(sets_fam[p],sets_fam[q]))
    feats += [float(np.mean(adj_t)) if adj_t else 0.0,float(np.mean(adj_f)) if adj_f else 0.0]
    within=[]
    byb=collections.defaultdict(list)
    for p,v in pitems: byb[v['bif']].append(p)
    for ps in byb.values():
        for i in range(len(ps)):
            for j in range(i+1,len(ps)): within.append(jac(sets_tok[ps[i]],sets_tok[ps[j]]))
    cross=[]
    for p,v in pitems:
        cand=[q for q,wv in pitems if q!=p and wv['bif']!=v['bif']]
        if cand:
            q=min(cand,key=lambda z:(abs(rank[z]-rank[p]),rank[z]))
            cross.append(jac(sets_tok[p],sets_tok[q]))
    feats.append((float(np.mean(within)) if within else 0.0)-(float(np.mean(cross)) if cross else 0.0))
    num=np.zeros(5); den=np.zeros(5)
    for _,v in pages.items():
        seq=[toks[i] for i in v['tok']]; seen=set(); n=len(seq)
        for i,t in enumerate(seq):
            b=min(4,int(5*i/max(n,1))); den[b]+=1; num[b]+=int(t not in seen); seen.add(t)
    rates=np.divide(num,den,out=np.zeros(5),where=den>0)
    feats.append(float(np.polyfit(np.linspace(0,1,5),rates,1)[0]))
    return feats


def generate_source(t0b, source, orig_tr, orig_te, fold, split, trial, strength=None):
    if source=='working_set_only': src=t0b.RegionalModel(orig_tr,fold,lam=0.0)
    elif source=='r64_exact': src=t0b.RegionalModel(orig_tr,fold,lam=(SOURCE_LAM[fold] if strength is None else float(strength)))
    else: src=t0b.RegionalModel(orig_tr,fold,lam=SOURCE_LAM[fold])
    def one(sk,slot):
        rng=np.random.default_rng(seed(split,source,trial,fold,slot))
        if source in ('working_set_only','r64_exact'):
            return src.generate(sk,seed(split,source,trial,fold,slot),'full')
        if source=='refractory_exact':
            old=t0b.REFRACTORY_RHO; t0b.REFRACTORY_RHO=0.04 if strength is None else float(strength)
            try:return t0b.generate_refractory(src,sk,rng)
            finally:t0b.REFRACTORY_RHO=old
        if source=='line_reset_r64': return t0b.generate_line_reset(src,sk,rng)
        if source=='family_persistence':
            old=t0b.FAMILY_PERSIST_RHO; t0b.FAMILY_PERSIST_RHO=0.04 if strength is None else float(strength)
            try:return t0b.generate_family_persistence(src,sk,rng)
            finally:t0b.FAMILY_PERSIST_RHO=old
        if source=='line_opener_unknown': return t0b.generate_opener(src,orig_tr,sk,rng)
        if source in ('gross_page_shuffle','gross_family_shuffle'):
            g=src.generate(sk,seed(split,source,trial,fold,slot),'full'); return t0b.shuffle_rows(g,source,rng)
        raise ValueError(source)
    return one(orig_tr,11),one(orig_te,29)


def run_fold(t0b, source, split, trial, fold, strength=None):
    orig_tr,orig_te=t0b.w.fold_data(fold)
    syn_tr,syn_te=generate_source(t0b,source,orig_tr,orig_te,fold,split,trial,strength)
    lam,_=t0b.choose_lambda(syn_tr,fold); fit=t0b.RegionalModel(syn_tr,fold,lam=lam)
    gain_region=float(fit.score(syn_te,'noregion')-fit.score(syn_te,'full'))
    gain_order=float(fit.score(syn_te,'noorder')-fit.score(syn_te,'full'))
    reps={}
    for rep in ('R0_canonical_tokens','R1_repeated_glyph_collapse'):
        rr=transform_rows(syn_te,rep,t0b.s2)
        f=recurrence_features(rr,split,source,trial,fold,rep,t0b.s2)+[gain_region,gain_order]
        assert len(f)==20 and all(np.isfinite(f)), (source,trial,fold,rep,f)
        reps[rep]=[float(x) for x in f]
    return {'fold':fold,'selected_lambda':float(lam),'features':reps}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('split',choices=list(SPLIT_NS))
    ap.add_argument('source',choices=list(SOURCE_CODE))
    ap.add_argument('start',type=int);ap.add_argument('stop',type=int)
    ap.add_argument('--strength',type=float,default=None)
    ap.add_argument('--manifest',default='/workspace/repo/instrumentation/manifests/memory_recurrence_source_id_v1.json')
    args=ap.parse_args()
    if args.split!='control' and args.source.startswith('gross_'): raise SystemExit('gross controls only allowed in control split')
    if args.split=='control' and args.source not in ('gross_page_shuffle','gross_family_shuffle','line_opener_unknown'): raise SystemExit('control split requires unknown control')
    if args.split=='power' and args.source not in ('r64_exact','refractory_exact','family_persistence'): raise SystemExit('power split source invalid')
    mh=verify_manifest(args.manifest); t0b=load_t0b()
    assert len(t0b.w.ROWS)==34087 and sorted(set(t0b.w.FOLDS.values()))==[0,1,2,3,4]
    print('P1QA='+json.dumps({'manifest_sha256':mh,'rows':len(t0b.w.ROWS),'features':FEATURE_NAMES,'source':args.source,'split':args.split,'strength':args.strength,'target_sealed':True},separators=(',',':')),flush=True)
    for trial in range(args.start,args.stop):
        st=time.time(); folds=[run_fold(t0b,args.source,args.split,trial,f,args.strength) for f in range(5)]
        out={'assay_id':'memory_recurrence_source_id_v1','split':args.split,'source':args.source,'trial':trial,'strength':args.strength,'feature_names':FEATURE_NAMES,'folds':folds,'elapsed_sec':time.time()-st,'target_opened':False}
        print('P1TRIAL='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__': main()
