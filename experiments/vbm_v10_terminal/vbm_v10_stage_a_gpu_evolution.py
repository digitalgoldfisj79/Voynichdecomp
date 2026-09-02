# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "triton>=3.0"]
# ///
"""Frozen VBM v10 Stage-A exact-likelihood GPU evolutionary optimiser.

Implements VBM_V10_GPU_EVOLUTION_ADDENDUM.md without altering the parent
synthetic data, objective, corpus sizes, oracle definitions, or recovery gates.
Binding mode is O2-positive only; O1/adversaries are opened only if O2 recovery
makes them necessary for the frozen Stage-A decision.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
with urllib.request.urlopen(BASE,timeout=120) as r: _src=r.read().decode('utf-8')
B={'__name__':'v10base'}; exec(compile(_src,BASE,'exec'),B)

VOW=B['VOW']; NV=B['NV']; KB=B['KB']; KR=B['KR']; KN=B['KN']; ALPHA=B['ALPHA']
NKEY=KB+KN
SIZES=[100,250,500,1000,2000]
CHAINS=8
POP=2_500_000
ELITE=4096
GENERATIONS=60
FRESH_N=POP//20
BATCH=32768
MAX_POLISH=40
NS='VBMV10EVOLUTION'


def hseed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff


def dense_lm(A):
    nctx=26**4
    out=np.full((nctx,26),math.log(1/26),dtype=np.float32)
    def cidx(s):
        z=0
        for ch in s: z=z*26+(ord(ch)-97)
        return z
    for ctx,counter in A['lm'].ctx.items():
        ix=cidx(ctx); tot=A['lm'].tot[ctx]
        for j in range(26):
            ch=chr(97+j)
            out[ix,j]=math.log((counter[ch]+ALPHA)/(tot+26*ALPHA))
    return out.reshape(-1)


def pack_assets(A,fit):
    run_chars=np.zeros((KR,5),dtype=np.uint8); run_lens=np.zeros(KR,dtype=np.uint8)
    for i,r in enumerate(A['runs']):
        run_lens[i]=len(r)
        for j,ch in enumerate(r): run_chars[i,j]=ord(ch)-97
    vow=np.asarray([ord(x)-97 for x in VOW],dtype=np.uint8)
    et=[]; es=[]; starts=[]; lens=[]
    for L in fit:
        starts.append(len(et)); st=len(et)
        for i,n in enumerate(L['n']):
            if n>=0: et.append(2); es.append(int(n))
            if i<len(L['b']): et.append(1); es.append(int(L['b'][i]))
        lens.append(len(et)-st)
    maxev=max(lens,default=0)
    if maxev>32: raise RuntimeError(('MAXEV',maxev))
    return dict(etype=np.asarray(et,dtype=np.uint8),esurf=np.asarray(es,dtype=np.int16),
                starts=np.asarray(starts,dtype=np.int32),lens=np.asarray(lens,dtype=np.int32),
                run_chars=run_chars.reshape(-1),run_lens=run_lens,vow=vow,maxev=maxev)


@triton.jit
def score_maps_kernel(maps, etype, esurf, lstart, llen, runchars, runlens,
                      vowchars, logp, llout, nout, BCAND,
                      BLOCK:tl.constexpr, MAXEV:tl.constexpr, STRIDE:tl.constexpr):
    pb=tl.program_id(0); line=tl.program_id(1)
    offs=pb*BLOCK+tl.arange(0,BLOCK); mask=offs<BCAND
    st=tl.load(lstart+line).to(tl.int32); ln=tl.load(llen+line).to(tl.int32)
    ctx=tl.zeros((BLOCK,),tl.int64); have=tl.zeros((BLOCK,),tl.int32)
    ll=tl.zeros((BLOCK,),tl.float32); ns=tl.zeros((BLOCK,),tl.int32)
    for jj in range(MAXEV):
        em=jj<ln
        typ=tl.load(etype+st+jj,mask=em,other=0).to(tl.int32)
        surf=tl.load(esurf+st+jj,mask=em,other=0).to(tl.int32)
        bm=mask & em & (typ==1)
        bv=tl.load(maps+offs*STRIDE+surf,mask=bm,other=0).to(tl.int32)
        bc=tl.load(vowchars+bv,mask=bm,other=0).to(tl.int64)
        sm=bm & (have>=4)
        ll += tl.load(logp+ctx*26+bc,mask=sm,other=0.0)
        ns += sm.to(tl.int32)
        nc=tl.where(have>=4,(ctx%17576)*26+bc,ctx*26+bc)
        ctx=tl.where(bm,nc,ctx); have=tl.where(bm,tl.minimum(have+1,4),have)
        nm=mask & em & (typ==2)
        nv=tl.load(maps+offs*STRIDE+KB+surf,mask=nm,other=0).to(tl.int32)
        rl=tl.load(runlens+nv,mask=nm,other=0).to(tl.int32)
        for kk in range(5):
            cm=nm & (kk<rl)
            cc=tl.load(runchars+nv*5+kk,mask=cm,other=0).to(tl.int64)
            sm2=cm & (have>=4)
            ll += tl.load(logp+ctx*26+cc,mask=sm2,other=0.0)
            ns += sm2.to(tl.int32)
            nc2=tl.where(have>=4,(ctx%17576)*26+cc,ctx*26+cc)
            ctx=tl.where(cm,nc2,ctx); have=tl.where(cm,tl.minimum(have+1,4),have)
    oi=line*BCAND+offs
    tl.store(llout+oi,ll,mask=mask); tl.store(nout+oi,ns,mask=mask)


class Engine:
    def __init__(self,A,fit,device='cuda'):
        self.A=A; self.fit=fit; self.L=len(fit); self.device=device
        P=pack_assets(A,fit); self.maxev=P['maxev']
        self.et=torch.tensor(P['etype'],dtype=torch.uint8,device=device)
        self.es=torch.tensor(P['esurf'],dtype=torch.int16,device=device)
        self.st=torch.tensor(P['starts'],dtype=torch.int32,device=device)
        self.ln=torch.tensor(P['lens'],dtype=torch.int32,device=device)
        self.rc=torch.tensor(P['run_chars'],dtype=torch.uint8,device=device)
        self.rl=torch.tensor(P['run_lens'],dtype=torch.uint8,device=device)
        self.vow=torch.tensor(P['vow'],dtype=torch.uint8,device=device)
        self.logp=torch.tensor(dense_lm(A),dtype=torch.float32,device=device)
        z=torch.zeros((1,NKEY),dtype=torch.uint8,device=device)
        _=self.score(z)

    def score(self,maps:torch.Tensor)->torch.Tensor:
        if maps.dtype!=torch.uint8: maps=maps.to(torch.uint8)
        if not maps.is_contiguous(): maps=maps.contiguous()
        q=int(maps.shape[0]); TW=128
        llo=torch.empty((self.L,q),dtype=torch.float32,device=self.device)
        no=torch.empty((self.L,q),dtype=torch.int32,device=self.device)
        grid=(triton.cdiv(q,TW),self.L)
        score_maps_kernel[grid](maps,self.et,self.es,self.st,self.ln,self.rc,self.rl,self.vow,
                                self.logp,llo,no,q,BLOCK=TW,MAXEV=32,STRIDE=NKEY,num_warps=4)
        return llo.sum(0)/no.sum(0).clamp_min(1)


def lex_order(keys:np.ndarray, scores:np.ndarray):
    ids=list(range(len(scores)))
    ids.sort(key=lambda i:(-float(scores[i]), tuple(int(x) for x in keys[i])))
    return np.asarray(ids,dtype=np.int64)


def choose_top(scores:torch.Tensor,maps:torch.Tensor,k:int):
    n=int(scores.numel())
    if n<=k:
        sc=scores.detach().cpu().numpy(); km=maps.detach().cpu().numpy()
        od=lex_order(km,sc)
        od_t=torch.tensor(od,device=maps.device,dtype=torch.long)
        return scores[od_t],maps[od_t]
    vals,idx=torch.topk(scores,k,largest=True,sorted=True)
    cut=vals[-1]
    nge=int((scores>=cut).sum().item())
    if nge==k:
        sel=idx
    else:
        gt=torch.nonzero(scores>cut,as_tuple=False).flatten(); need=k-int(gt.numel())
        eq=torch.nonzero(scores==cut,as_tuple=False).flatten()
        eqm=maps[eq].detach().cpu().numpy()
        eord=list(range(len(eqm))); eord.sort(key=lambda i:tuple(int(x) for x in eqm[i]))
        take=eq[torch.tensor(eord[:need],device=eq.device,dtype=torch.long)]
        sel=torch.cat([gt,take])
    sc=scores[sel].detach().cpu().numpy(); km=maps[sel].detach().cpu().numpy()
    od=lex_order(km,sc); od_t=torch.tensor(od,device=maps.device,dtype=torch.long)
    return scores[sel][od_t],maps[sel][od_t]


def merge_top(cur_scores,cur_maps,new_scores,new_maps,k=ELITE):
    if cur_scores is None: return choose_top(new_scores,new_maps,k)
    return choose_top(torch.cat([cur_scores,new_scores]),torch.cat([cur_maps,new_maps]),k)


def random_keys(q,g,device,fixed_cols=None,fixed_vals=None):
    out=torch.empty((q,NKEY),dtype=torch.uint8,device=device)
    out[:,:KB]=torch.randint(0,NV,(q,KB),generator=g,device=device,dtype=torch.int64).to(torch.uint8)
    out[:,KB:]=torch.randint(0,KR,(q,KN),generator=g,device=device,dtype=torch.int64).to(torch.uint8)
    if fixed_cols is not None and len(fixed_cols): out[:,fixed_cols]=fixed_vals
    return out


def mutation_k(gen):
    if gen<=10:return 12
    if gen<=25:return 6
    if gen<=45:return 3
    return 1


def mutate_offspring(elites,q,base,gen,g,cdf,active_cols,fixed_cols,fixed_vals,device,fresh_shift):
    u=torch.rand((q,),generator=g,device=device)
    parents=torch.searchsorted(cdf,u,right=False).clamp_max(ELITE-1)
    out=elites[parents].clone()
    k=mutation_k(gen); rows=torch.arange(q,device=device,dtype=torch.long)
    chosen=[]; na=int(active_cols.numel())
    if na<k: raise RuntimeError(('active dictionary too small',na,k))
    for _ in range(k):
        ix=torch.randint(0,na,(q,),generator=g,device=device,dtype=torch.long)
        if chosen:
            bad=torch.zeros((q,),dtype=torch.bool,device=device)
            for p in chosen: bad |= (ix==p)
            while bool(bad.any().item()):
                m=int(bad.sum().item()); ix[bad]=torch.randint(0,na,(m,),generator=g,device=device,dtype=torch.long)
                bad.zero_()
                for p in chosen: bad |= (ix==p)
        chosen.append(ix.clone())
        col=active_cols[ix]; old=out[rows,col].to(torch.long)
        isb=col<KB
        r5=torch.randint(0,4,(q,),generator=g,device=device,dtype=torch.long)
        r32=torch.randint(0,31,(q,),generator=g,device=device,dtype=torch.long)
        rr=torch.where(isb,r5,r32)
        nv=rr+(rr>=old).to(torch.long)
        out[rows,col]=nv.to(torch.uint8)
    cid=torch.arange(base,base+q,device=device,dtype=torch.int64)
    perm=(104729*cid+fresh_shift)%POP
    fresh=perm<FRESH_N
    if bool(fresh.any().item()):
        fi=torch.nonzero(fresh,as_tuple=False).flatten(); m=int(fi.numel())
        out[fi]=random_keys(m,g,device,fixed_cols,fixed_vals)
    if fixed_cols is not None and len(fixed_cols): out[:,fixed_cols]=fixed_vals
    return out


def score_population(engine,make_batch,previous_scores=None,previous_maps=None):
    gs=previous_scores; gm=previous_maps; base=0
    while base<POP:
        q=min(BATCH,POP-base); maps=make_batch(base,q); sc=engine.score(maps)
        ls,lm=choose_top(sc,maps,min(ELITE,q)); gs,gm=merge_top(gs,gm,ls,lm)
        base+=q
    return gs,gm


def coordinate_polish(engine,key,score,fixed_cols):
    device=key.device; fixed=set(int(x) for x in fixed_cols); accepted=0; history=[float(score)]
    while accepted<MAX_POLISH:
        neigh=[]
        for col in range(NKEY):
            if col in fixed: continue
            old=int(key[0,col].item()); dom=NV if col<KB else KR
            for v in range(dom):
                if v==old: continue
                z=key.clone(); z[0,col]=v; neigh.append(z)
        cand=torch.cat(neigh,dim=0); sc=engine.score(cand)
        best=float(sc.max().item())
        if not (best>float(score)): break
        idxs=torch.nonzero(sc==sc.max(),as_tuple=False).flatten()
        if int(idxs.numel())==1: bi=int(idxs[0].item())
        else:
            km=cand[idxs].detach().cpu().numpy(); order=list(range(len(km))); order.sort(key=lambda i:tuple(int(x) for x in km[i])); bi=int(idxs[order[0]].item())
        key=cand[bi:bi+1].clone(); score=float(sc[bi].item()); accepted+=1; history.append(score)
    return key,score,accepted,history


def active_and_fixed(fit,key,oracle):
    bc,nc=B['counts'](fit); fixed=[]
    if oracle=='O1':
        fb,fn=B['top_quarter_fixed'](fit,key)
        fixed=sorted(list(fb)+[KB+x for x in fn])
    fixed_set=set(fixed); active=[i for i in range(NKEY) if i not in fixed_set]
    true=np.concatenate([np.asarray(key['bmap'],dtype=np.uint8),np.asarray(key['nmap'],dtype=np.uint8)])
    return torch.tensor(active,dtype=torch.long,device='cuda'),fixed,true,bc,nc


def run_chain(engine,A,fit,key,lang,rep,size,oracle,chain,active,fixed,truevals):
    device='cuda'; fixed_cols=torch.tensor(fixed,dtype=torch.long,device=device) if fixed else None
    fixed_vals=torch.tensor(truevals[fixed],dtype=torch.uint8,device=device) if fixed else None
    g=torch.Generator(device=device); g.manual_seed(hseed(NS,lang,rep,size,oracle,chain,0))
    def mk0(base,q): return random_keys(q,g,device,fixed_cols,fixed_vals)
    t0=time.time(); scores,elites=score_population(engine,mk0)
    weights=(1.0/torch.sqrt(torch.arange(1,ELITE+1,device=device,dtype=torch.float64))).to(torch.float64)
    cdf=torch.cumsum(weights/weights.sum(),0).to(torch.float32); cdf[-1]=1.0
    for gen in range(1,GENERATIONS+1):
        gg=torch.Generator(device=device); gg.manual_seed(hseed(NS,lang,rep,size,oracle,chain,gen))
        shift=hseed(NS,'FRESH',lang,rep,size,oracle,chain,gen)%POP
        def mkg(base,q,gen=gen,gg=gg,shift=shift): return mutate_offspring(elites,q,base,gen,gg,cdf,active,fixed_cols,fixed_vals,device,shift)
        scores,elites=score_population(engine,mkg,scores,elites)
        if gen in (1,10,25,45,60): print('V10EV_PROGRESS='+json.dumps({'lang':lang,'rep':rep,'size':size,'oracle':oracle,'chain':chain,'gen':gen,'best_fit':float(scores[0].item())},sort_keys=True),flush=True)
    best=elites[0:1].clone(); bestscore=float(scores[0].item())
    best,bestscore,nacc,hist=coordinate_polish(engine,best,bestscore,fixed)
    arr=best[0].detach().cpu().numpy()
    out={'chain':chain,'fit_score':bestscore,'bmap':arr[:KB].astype(int).tolist(),'nmap':arr[KB:].astype(int).tolist(),
         'polish_accept':nacc,'polish_history':hist,'evaluations':int((GENERATIONS+1)*POP),'runtime_s':time.time()-t0}
    p=Path(f'/tmp/v10ev_{lang}_r{rep}_n{size}_{oracle}_c{chain}.pkl'); tmp=p.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(out,protocol=5)); os.replace(tmp,p)
    print('V10EV_CHAIN='+json.dumps(out,sort_keys=True,separators=(',',':')),flush=True)
    return out


def map_from_result(r): return {'bmap':np.asarray(r['bmap'],dtype=np.int16),'nmap':np.asarray(r['nmap'],dtype=np.int16),'fit_score':r['fit_score']}


def one_binding(A,lines,key,lang,rep,size,oracle='O2'):
    z=lines[:size]; cut=int(.8*size); fit=z[:cut]; hold=z[cut:]
    active,fixed,truevals,bc,nc=active_and_fixed(fit,key,oracle); engine=Engine(A,fit); results=[]
    for ch in range(CHAINS): results.append(run_chain(engine,A,fit,key,lang,rep,size,oracle,ch,active,fixed,truevals))
    results.sort(key=lambda r:(-float(r['fit_score']),tuple(r['bmap']+r['nmap'])))
    best=results[0]; m=map_from_result(best); fc=(bc,nc)
    rb,rn,rc=B['recovery'](m,key,hold,fc,A,0); rb5,rn5,rc5=B['recovery'](m,key,hold,fc,A,5)
    cb,cn=B['coverage'](hold,*fc,1); holdlm=B['score_lines'](hold,A,m); rand=B['random_baseline'](hold,A,f'EV:{lang}:R{rep}:N{size}:{oracle}'); adv=holdlm-rand
    truehold=B['score_lines'](hold,A,{'bmap':key['bmap'],'nmap':key['nmap']})
    row={'language':lang,'rep':rep,'size':size,'oracle':oracle,'fit_lines':len(fit),'hold_lines':len(hold),
         'REC_B':rb,'REC_N':rn,'REC_CHAR':rc,'REC_B5':rb5,'REC_N5':rn5,'REC_CHAR5':rc5,
         'COV_B':cb,'COV_N':cn,'HOLD_LM':holdlm,'RAND_HOLD_LM':rand,'HOLD_ADV':adv,
         'FIT_SCORE':best['fit_score'],'O0_HOLD_LM_TRUE_KEY':truehold,'best_chain':best['chain'],
         'chain_scores':[float(x['fit_score']) for x in results]}
    print('V10EV_ROW='+json.dumps(row,sort_keys=True,separators=(',',':')),flush=True); return row


def smoke():
    A=B['assets']('DE'); pl=B['plaintext_lines'](A,'SMOKE:EVOLUTION',120); key=B['codebook'](A,'SMOKE:EVOLUTION'); lines=[B['encode'](x,A,key,f'SMOKE:EVOLUTION:L{i}') for i,x in enumerate(pl)]; fit=lines[:96]
    eng=Engine(A,fit); rng=np.random.default_rng(hseed(NS,'SMOKEKEY')); maps=[]
    for _ in range(4):
        bm=rng.integers(0,NV,KB,dtype=np.uint8); nm=rng.integers(0,KR,KN,dtype=np.uint8); maps.append(np.concatenate([bm,nm]))
    mt=torch.tensor(np.stack(maps),dtype=torch.uint8,device='cuda'); gs=eng.score(mt).detach().cpu().numpy(); dif=[]
    for i,k in enumerate(maps):
        cm={'bmap':k[:KB].astype(np.int16),'nmap':k[KB:].astype(np.int16)}; cs=B['score_lines'](fit,A,cm); dif.append(abs(float(gs[i])-float(cs)))
    truth=np.concatenate([np.asarray(key['bmap'],dtype=np.uint8),np.asarray(key['nmap'],dtype=np.uint8)])
    corrupt=truth.copy(); idx=rng.choice(NKEY,10,replace=False)
    for col in idx:
        dom=NV if col<KB else KR; old=int(corrupt[col]); r=int(rng.integers(0,dom-1)); corrupt[col]=r+(r>=old)
    kt=torch.tensor(corrupt[None,:],dtype=torch.uint8,device='cuda'); s0=float(eng.score(kt)[0].item()); kp,sp,nacc,hist=coordinate_polish(eng,kt,s0,[])
    mono=all(hist[i+1]>=hist[i] for i in range(len(hist)-1)); ok=max(dif)<=1e-5 and sp>s0 and mono
    out={'max_cpu_gpu_abs_diff':max(dif),'tolerance':1e-5,'corrupt_start':s0,'polished':sp,'accepted':nacc,'monotone':mono,'pass':ok}
    print('V10EV_SMOKE='+json.dumps(out,sort_keys=True),flush=True)
    if not ok: raise SystemExit(2)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','bind'],required=True); ap.add_argument('--lang',choices=['DE','IT']); ap.add_argument('--rep',type=int,choices=[0,1,2]); ap.add_argument('--sizes',default='100,250,500,1000,2000'); ap.add_argument('--oracle',choices=['O2','O1'],default='O2'); a=ap.parse_args()
    if a.mode=='smoke': smoke(); return
    if a.lang is None or a.rep is None: ap.error('--lang and --rep required in bind mode')
    sizes=[int(x) for x in a.sizes.split(',') if x]
    if any(x not in SIZES for x in sizes): raise SystemExit('non-frozen Stage-A size')
    A=B['assets'](a.lang); lines,key=B['make_positive'](a.lang,a.rep,A); rows=[]
    for size in sizes: rows.append(one_binding(A,lines,key,a.lang,a.rep,size,a.oracle))
    print('VBM_V10_EVOLUTION_RESULT='+json.dumps({'language':a.lang,'rep':a.rep,'oracle':a.oracle,'rows':rows},sort_keys=True,separators=(',',':')),flush=True)

if __name__=='__main__': main()
