# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3"]
# ///
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request
import numpy as np

NS='VBMV13EOPERATOR20260902'
Q0NS='VBMJOACHIMEXACTV9Q0'
DATA_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
ATOMS=('ckh','cth','cph','cfh','ch','sh','qo')
UA={'User-Agent':'VBMV13EOperator/2026-09-02'}
SMOOTH=.5
TOPK=32


def seed(tag):
    return int(hashlib.sha256(f'{NS}::{tag}'.encode()).hexdigest()[:16],16)%(2**32)

def rng(tag): return np.random.default_rng(seed(tag))

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)

def split_folio(fid):
    h=hashlib.sha256(f'{Q0NS}::{fid}'.encode()).hexdigest()[:8]
    return 'HOLD' if int(h,16)%5==0 else 'TRAIN'

def left_half(w):
    for a in ATOMS:
        if len(w)>=len(a)+1 and w.startswith(a): return a
    return w[0]

def parse_token(w):
    if not re.fullmatch(r'[a-z]+',w): return None
    if len(w)==1:return (w,'',w)
    L=left_half(w);R=w[-1]
    if len(w)<len(L)+1:return None
    return (L,w[len(L):-1],R)

def parse_segments(text):
    out=[];seg=[]
    def flush():
        nonlocal seg
        if seg: out.append(seg);seg=[]
    for w in text.split():
        if parse_token(w) is None:flush()
        else:seg.append(w)
    flush();return out

def esk(n): return re.sub(r'e+','E',n)

def js(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float);m=.5*(p+q)
    return .5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m))

def norm_counts(c):
    x=np.asarray(c,float)+SMOOTH
    return x/x.sum()

def apply_op(p,w):
    z=np.asarray(p,float)*np.exp(np.asarray(w,float));return z/z.sum()

def op_from_ratios(ratios, signs=None):
    if not ratios: return None
    A=np.stack(ratios)
    if signs is not None:A=A*np.asarray(signs,float)[:,None]
    w=np.median(A,axis=0);w=w-w.mean();return np.clip(w,-2,2)

def delta_pair(src,tgt,w):
    return float(js(src,tgt)-js(apply_op(src,w),tgt))

def skeleton_median(edge_deltas):
    by=collections.defaultdict(list)
    for sk,d in edge_deltas:by[sk].append(float(d))
    vals=[np.mean(v) for v in by.values()]
    return float(np.median(vals)) if vals else float('nan'), {k:float(np.mean(v)) for k,v in by.items()}

# ---------------- Synthetic calibration ----------------

def synthetic_rep(mode,rep,nnull=1000):
    R=rng(f'SYN::{mode}::{rep}'); S=40;D=34
    bases_pre=R.dirichlet(np.full(D,.7),size=S);bases_fol=R.dirichlet(np.full(D,.7),size=S)
    if mode=='SHARED':
        wp=R.normal(0,.35,D);wf=R.normal(0,.35,D);wp-=wp.mean();wf-=wf.mean()
        Wp=np.tile(wp,(S,1));Wf=np.tile(wf,(S,1))
    else:
        Wp=R.normal(0,.35,(S,D));Wf=R.normal(0,.35,(S,D));Wp-=Wp.mean(1,keepdims=True);Wf-=Wf.mean(1,keepdims=True)
    edges=[]
    for s in range(S):
        p0p=bases_pre[s];p1p=apply_op(p0p,Wp[s]);p0f=bases_fol[s];p1f=apply_op(p0f,Wf[s])
        n0=int(R.integers(80,241));n1=int(R.integers(80,241))
        c0p=R.multinomial(n0,p0p);c1p=R.multinomial(n1,p1p);c0f=R.multinomial(n0,p0f);c1f=R.multinomial(n1,p1f)
        a0p=norm_counts(c0p);a1p=norm_counts(c1p);a0f=norm_counts(c0f);a1f=norm_counts(c1f)
        rp=np.log(a1p/a0p);rf=np.log(a1f/a0f)
        edges.append({'sk':str(s),'rp':rp,'rf':rf,'sp':a0p,'tp':a1p,'sf':a0f,'tf':a1f})
    def stat(sign_by_sk=None):
        ds=[]
        for e in edges:
            train=[x for x in edges if x['sk']!=e['sk']]
            sp=[(sign_by_sk or {}).get(x['sk'],1) for x in train]
            wp=op_from_ratios([x['rp'] for x in train],sp);wf=op_from_ratios([x['rf'] for x in train],sp)
            d=.5*(delta_pair(e['sp'],e['tp'],wp)+delta_pair(e['sf'],e['tf'],wf));ds.append((e['sk'],d))
        return skeleton_median(ds)[0]
    obs=stat(); null=[];skels=sorted({e['sk'] for e in edges})
    for i in range(nnull):
        rr=rng(f'SYNNULL::{mode}::{rep}::{i}');sg={s:(1 if rr.random()>=.5 else -1) for s in skels};null.append(stat(sg))
    null=np.asarray(null);p=(1+np.sum(null>=obs))/(len(null)+1);sd=null.std(ddof=1);z=(obs-null.mean())/sd if sd>0 else float('inf')
    return {'mode':mode,'rep':rep,'median_delta':obs,'p':float(p),'z':float(z),'null_median':float(np.median(null))}

def synthetic_calibration():
    shared=[synthetic_rep('SHARED',r) for r in range(6)];idio=[synthetic_rep('IDIO',r) for r in range(6)]
    sp=sum(x['median_delta']>0 and x['p']<=.01 for x in shared);fp=sum(x['p']<=.01 for x in idio)
    ok=sp>=5 and fp<=1
    return {'shared':shared,'idiosyncratic':idio,'shared_passes':sp,'idio_false_positives':fp,'qualifies':ok}

# ---------------- Voynich data ----------------

def build_corpus():
    d=get_json(DATA_URL)
    freq={'TRAIN':collections.Counter(),'HOLD':collections.Counter()}
    ctx={sp:collections.defaultdict(lambda:{'pre':collections.Counter(),'fol':collections.Counter()}) for sp in ['TRAIN','HOLD']}
    bridges_train=collections.Counter();meta={'TRAIN':0,'HOLD':0}
    for fid,lines in sorted(d['pages'].items()):
        if fid in H1 or fid in C1:continue
        sp=split_folio(fid)
        for ln in sorted(lines,key=lambda x:int(x) if str(x).isdigit() else 999999):
            txt=lines[ln].get('t',{}).get('ZLZI','')
            if not txt:continue
            for words in parse_segments(txt):
                tr=[parse_token(w) for w in words];br=[tr[i][2]+'|'+tr[i+1][0] for i in range(len(tr)-1)]
                if sp=='TRAIN':bridges_train.update(br)
                for i,(_,n,_) in enumerate(tr):
                    if not n:continue
                    freq[sp][n]+=1;meta[sp]+=1
                    if i>0:ctx[sp][n]['pre'][br[i-1]]+=1
                    if i<len(br):ctx[sp][n]['fol'][br[i]]+=1
    vocab=[x for x,_ in bridges_train.most_common(TOPK)];index={b:i for i,b in enumerate(vocab)};OTHER=len(vocab);D=len(vocab)+1
    def vec(sp,n,side):
        a=np.zeros(D,dtype=float)
        for b,c in ctx[sp][n][side].items():a[index.get(b,OTHER)]+=c
        return norm_counts(a)
    eligible=[n for n,c in freq['TRAIN'].items() if c>=20]
    levels=collections.defaultdict(lambda:collections.defaultdict(list))
    for n in eligible:levels[esk(n)][n.count('e')].append(n)
    edges=[]
    for sk,L in levels.items():
        for m in sorted(L):
            if m+1 not in L:continue
            for a in sorted(L[m]):
                for b in sorted(L[m+1]):
                    edges.append({'sk':sk,'m':m,'a':a,'b':b,
                      'tr_ap':vec('TRAIN',a,'pre'),'tr_bp':vec('TRAIN',b,'pre'),'tr_af':vec('TRAIN',a,'fol'),'tr_bf':vec('TRAIN',b,'fol'),
                      'rp':np.log(vec('TRAIN',b,'pre')/vec('TRAIN',a,'pre')),'rf':np.log(vec('TRAIN',b,'fol')/vec('TRAIN',a,'fol')),
                      'hold_ok':freq['HOLD'][a]>=5 and freq['HOLD'][b]>=5,
                      'ho_ap':vec('HOLD',a,'pre'),'ho_bp':vec('HOLD',b,'pre'),'ho_af':vec('HOLD',a,'fol'),'ho_bf':vec('HOLD',b,'fol'),
                      'train_a':freq['TRAIN'][a],'train_b':freq['TRAIN'][b],'hold_a':freq['HOLD'][a],'hold_b':freq['HOLD'][b]})
    return edges,freq,{'bridge_vocab':vocab,'eligible_nuclei':len(eligible),'train_nucleus_events':meta['TRAIN'],'hold_nucleus_events':meta['HOLD']}

def evaluate_actual(edges):
    ev=[e for e in edges if e['hold_ok']];train_sk=sorted({e['sk'] for e in edges});ev_sk=sorted({e['sk'] for e in ev})
    def stat(sign_by_sk=None):
        ds=[]
        for e in ev:
            train=[x for x in edges if x['sk']!=e['sk']]
            signs=[(sign_by_sk or {}).get(x['sk'],1) for x in train]
            wp=op_from_ratios([x['rp'] for x in train],signs);wf=op_from_ratios([x['rf'] for x in train],signs)
            if wp is None or wf is None:continue
            d=.5*(delta_pair(e['ho_ap'],e['ho_bp'],wp)+delta_pair(e['ho_af'],e['ho_bf'],wf));ds.append((e['sk'],d))
        return skeleton_median(ds)
    obs,by=stat();null=[]
    for i in range(10000):
        rr=rng(f'ACTNULL::{i}');sg={s:(1 if rr.random()>=.5 else -1) for s in train_sk};null.append(stat(sg)[0])
    null=np.asarray(null,float);finite=null[np.isfinite(null)];p=(1+np.sum(finite>=obs))/(len(finite)+1);sd=finite.std(ddof=1);z=(obs-finite.mean())/sd if sd>0 else float('inf')
    # A/B transfer
    A=[s for s in train_sk if int(hashlib.sha256(f'{NS}::SKELSPLIT::{s}'.encode()).hexdigest()[:8],16)%2==0];B=[s for s in train_sk if s not in A]
    def transfer(src_sks,tgt_sks):
        src=[e for e in edges if e['sk'] in src_sks];tgt=[e for e in edges if e['sk'] in tgt_sks]
        wp=op_from_ratios([e['rp'] for e in src]);wf=op_from_ratios([e['rf'] for e in src]);ds=[]
        if wp is None or wf is None:return float('nan'),{}
        for e in tgt:
            d=.5*(delta_pair(e['tr_ap'],e['tr_bp'],wp)+delta_pair(e['tr_af'],e['tr_bf'],wf));ds.append((e['sk'],d))
        return skeleton_median(ds)
    ab,ab_by=transfer(A,B);ba,ba_by=transfer(B,A)
    # repeated application diagnostic
    bylevel=collections.defaultdict(lambda:collections.defaultdict(list))
    for e in edges:
        bylevel[e['sk']][e['m']].append(e['a']);b_m=e['m']+1;bylevel[e['sk']][b_m].append(e['b'])
    # de-duplicate
    reps=[]
    # use direct lookup from edges for vectors/frequencies
    lookup={e['a']:e for e in edges}; lookup.update({e['b']:e for e in edges})
    for sk,L in bylevel.items():
        for m in sorted(L):
            if m+1 not in L or m+2 not in L:continue
            for a in sorted(set(L[m])):
              for c in sorted(set(L[m+2])):
                # find vectors from any edge containing these nuclei
                ea=next((e for e in edges if e['a']==a or e['b']==a),None);ec=next((e for e in edges if e['a']==c or e['b']==c),None)
                if ea is None or ec is None:continue
                # determine hold threshold from attached counts
                ca=(ea['hold_a'] if ea['a']==a else ea['hold_b']);cc=(ec['hold_a'] if ec['a']==c else ec['hold_b'])
                if ca<5 or cc<5:continue
                train=[x for x in edges if x['sk']!=sk];wp=op_from_ratios([x['rp'] for x in train]);wf=op_from_ratios([x['rf'] for x in train])
                if wp is None or wf is None:continue
                sap=(ea['ho_ap'] if ea['a']==a else ea['ho_bp']);saf=(ea['ho_af'] if ea['a']==a else ea['ho_bf']);tcp=(ec['ho_ap'] if ec['a']==c else ec['ho_bp']);tcf=(ec['ho_af'] if ec['a']==c else ec['ho_bf'])
                idd=.5*(js(sap,tcp)+js(saf,tcf));one=.5*(js(apply_op(sap,wp),tcp)+js(apply_op(saf,wf),tcf));two=.5*(js(apply_op(apply_op(sap,wp),wp),tcp)+js(apply_op(apply_op(saf,wf),wf),tcf))
                reps.append({'sk':sk,'m':m,'identity':float(idd),'one':float(one),'two':float(two),'two_beats_both':bool(two<idd and two<one)})
    return {'eligible_edges_total':len(edges),'evaluable_edges':len(ev),'evaluable_skeletons':len(ev_sk),'primary_median_delta':obs,'skeleton_deltas':by,'null_median':float(np.median(finite)),'null_mean':float(np.mean(finite)),'null_sd':float(sd),'p':float(p),'z':float(z),'train_transfer_A_to_B':ab,'train_transfer_B_to_A':ba,'n_skel_A':len(A),'n_skel_B':len(B),'repeated_n':len(reps),'repeated_two_beats_both':sum(x['two_beats_both'] for x in reps),'repeated_examples':reps[:20]}

def main():
    cal=synthetic_calibration();print('V13_CAL='+json.dumps(cal,sort_keys=True),flush=True)
    edges,freq,meta=build_corpus();res=evaluate_actual(edges);print('V13_ACTUAL='+json.dumps({'meta':meta,'result':res},sort_keys=True),flush=True)
    gate=(cal['qualifies'] and res['evaluable_edges']>=12 and res['evaluable_skeletons']>=8 and res['primary_median_delta']>0 and res['p']<=.01 and res['z']>=2.5 and res['train_transfer_A_to_B']>0 and res['train_transfer_B_to_A']>0)
    verdict='V13_SHARED_E_OPERATOR_SUPPORTED' if gate else 'V13_NO_SHARED_E_OPERATOR_EVIDENCE'
    print('VBM_V13_FINAL_RESULT='+json.dumps({'verdict':verdict,'gate':bool(gate),'calibration_qualifies':cal['qualifies'],'evaluable_edges':res['evaluable_edges'],'evaluable_skeletons':res['evaluable_skeletons'],'median_delta':res['primary_median_delta'],'p':res['p'],'z':res['z'],'A_to_B':res['train_transfer_A_to_B'],'B_to_A':res['train_transfer_B_to_A'],'plaintext_opened':False,'gpu_used':False},sort_keys=True),flush=True)
if __name__=='__main__':main()
