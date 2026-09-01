# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json, math, urllib.request
import numpy as np
import torch
import triton
import triton.language as tl

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
with urllib.request.urlopen(BASE,timeout=120) as r: src=r.read().decode('utf-8')
ns={'__name__':'v10base'}; exec(compile(src,BASE,'exec'),ns)
VOW=ns['VOW']; KR=ns['KR']; ALPHA=ns['ALPHA']; KB=30; KN=96; KALL=126

@triton.jit
def explicit_score_kernel(maps, etype, esurf, lstart, llen, runchars, runlens, vowchars, logp, llout, nout, B: tl.constexpr, STRIDE: tl.constexpr, MAXEV: tl.constexpr, BLOCK: tl.constexpr):
    pb=tl.program_id(0); line=tl.program_id(1)
    offs=pb*BLOCK+tl.arange(0,BLOCK); mask=offs<B
    st=tl.load(lstart+line).to(tl.int32); ln=tl.load(llen+line).to(tl.int32)
    ctx=tl.zeros((BLOCK,),tl.uint32); have=tl.zeros((BLOCK,),tl.int32); ll=tl.zeros((BLOCK,),tl.float32); nsc=tl.zeros((BLOCK,),tl.int32)
    for jj in range(MAXEV):
        em=jj<ln
        t=tl.load(etype+st+jj,mask=em,other=0).to(tl.int32)
        s=tl.load(esurf+st+jj,mask=em,other=0).to(tl.int32)
        bm=mask & em & (t==1)
        bv=tl.load(maps+offs*STRIDE+s,mask=bm,other=0).to(tl.int32)
        bc=tl.load(vowchars+bv,mask=bm,other=0).to(tl.uint32)
        scored=bm & (have>=4)
        lp=tl.load(logp+ctx.to(tl.int64)*26+bc.to(tl.int64),mask=scored,other=0.0)
        ll += lp; nsc += scored.to(tl.int32)
        nctx=tl.where(have>=4,(ctx % 17576)*26+bc,ctx*26+bc)
        ctx=tl.where(bm,nctx,ctx); have=tl.where(bm,tl.minimum(have+1,4),have)
        nm=mask & em & (t==2)
        rv=tl.load(maps+offs*STRIDE+30+s,mask=nm,other=0).to(tl.int32)
        rlen=tl.load(runlens+rv,mask=nm,other=0).to(tl.int32)
        for kk in range(5):
            cm=nm & (kk<rlen)
            cc=tl.load(runchars+rv*5+kk,mask=cm,other=0).to(tl.uint32)
            scored2=cm & (have>=4)
            lp2=tl.load(logp+ctx.to(tl.int64)*26+cc.to(tl.int64),mask=scored2,other=0.0)
            ll += lp2; nsc += scored2.to(tl.int32)
            nctx2=tl.where(have>=4,(ctx % 17576)*26+cc,ctx*26+cc)
            ctx=tl.where(cm,nctx2,ctx); have=tl.where(cm,tl.minimum(have+1,4),have)
    outidx=line*B+offs
    tl.store(llout+outidx,ll,mask=mask); tl.store(nout+outidx,nsc,mask=mask)

class ExactScorer:
    def __init__(self,lines,A,device='cuda'):
        self.lines=lines; self.A=A; self.device=device; self.L=len(lines)
        logp=np.full((26**4,26),math.log(1/26),dtype=np.float32)
        def cidx(s):
            z=0
            for ch in s:z=z*26+(ord(ch)-97)
            return z
        for ctx,counter in A['lm'].ctx.items():
            ix=cidx(ctx); tot=A['lm'].tot[ctx]
            for j in range(26):
                ch=chr(97+j); logp[ix,j]=math.log((counter[ch]+ALPHA)/(tot+26*ALPHA))
        rc=np.zeros((KR,5),dtype=np.uint8); rl=np.zeros(KR,dtype=np.uint8)
        for i,r in enumerate(A['runs']):
            rl[i]=len(r)
            for j,ch in enumerate(r):rc[i,j]=ord(ch)-97
        et=[];es=[];starts=[];lens=[]
        for L in lines:
            starts.append(len(et)); p=len(et)
            for i,n in enumerate(L['n']):
                if n>=0:et.append(2);es.append(int(n))
                if i<len(L['b']):et.append(1);es.append(int(L['b'][i]))
            lens.append(len(et)-p)
        self.maxev=max(lens); assert self.maxev<=32
        self.et=torch.tensor(et,dtype=torch.uint8,device=device);self.es=torch.tensor(es,dtype=torch.int16,device=device)
        self.starts=torch.tensor(starts,dtype=torch.int32,device=device);self.lens=torch.tensor(lens,dtype=torch.int32,device=device)
        self.rc=torch.tensor(rc.reshape(-1),dtype=torch.uint8,device=device);self.rl=torch.tensor(rl,dtype=torch.uint8,device=device)
        self.vow=torch.tensor([ord(c)-97 for c in VOW],dtype=torch.uint8,device=device)
        self.logp=torch.tensor(logp.reshape(-1),dtype=torch.float32,device=device)
    def score(self,maps):
        if not torch.is_tensor(maps):maps=torch.tensor(maps,dtype=torch.uint8,device=self.device)
        maps=maps.to(device=self.device,dtype=torch.uint8,non_blocking=True).contiguous();B=maps.shape[0];BLOCK=128
        ll=torch.empty((self.L,B),dtype=torch.float32,device=self.device);nn=torch.empty((self.L,B),dtype=torch.int32,device=self.device)
        grid=(triton.cdiv(B,BLOCK),self.L)
        explicit_score_kernel[grid](maps,self.et,self.es,self.starts,self.lens,self.rc,self.rl,self.vow,self.logp,ll,nn,B=B,STRIDE=KALL,MAXEV=32,BLOCK=BLOCK,num_warps=4)
        return ll.sum(0)/nn.sum(0).clamp_min(1)

def cpu_score(lines,A,m):
    mm={'bmap':m[:30].astype(np.int16),'nmap':m[30:].astype(np.int16)}
    return float(ns['score_lines'](lines,A,mm))

def all_neighbors(m,fixed=set()):
    rows=[]
    base=m.cpu().numpy().astype(np.uint8)
    for p in range(KALL):
        if p in fixed:continue
        dom=5 if p<30 else 32
        for v in range(dom):
            if v==int(base[p]):continue
            z=base.copy();z[p]=v;rows.append(z)
    return torch.tensor(np.stack(rows),dtype=torch.uint8,device=m.device)

def coordinate_polish(m,scorer,max_accept=40):
    cur=m.clone();cur_score=float(scorer.score(cur[None,:])[0].item());trace=[cur_score];accepted=0
    while accepted<max_accept:
        neigh=all_neighbors(cur);scores=scorer.score(neigh);val,ix=torch.max(scores,0);best=float(val.item())
        if best<=cur_score+1e-8:break
        cur=neigh[int(ix.item())].clone();cur_score=best;trace.append(best);accepted+=1
    return cur,cur_score,trace

def main():
    assert torch.cuda.is_available();torch.cuda.set_device(0)
    A=ns['assets']('DE');lines,key=ns['make_positive']('DE',991,A);fit=lines[:80]
    scorer=ExactScorer(fit,A)
    rng=np.random.default_rng(ns['seed']('V10GPU','SMOKE','MAPS'))
    maps=np.empty((32,KALL),dtype=np.uint8);maps[:,:30]=rng.integers(0,5,size=(32,30),dtype=np.uint8);maps[:,30:]=rng.integers(0,32,size=(32,96),dtype=np.uint8)
    gs=scorer.score(maps).detach().cpu().numpy();cs=np.asarray([cpu_score(fit,A,m) for m in maps]);diff=float(np.max(np.abs(gs-cs)))
    scorer_ok=diff<=1e-5
    truth=np.concatenate([key['bmap'].astype(np.uint8),key['nmap'].astype(np.uint8)]);bad=truth.copy();pos=rng.choice(KALL,size=10,replace=False)
    for p in pos:
        dom=5 if p<30 else 32;old=int(bad[p]);r=int(rng.integers(0,dom-1));bad[p]=r+(r>=old)
    bad_t=torch.tensor(bad,dtype=torch.uint8,device='cuda');start=float(scorer.score(bad_t[None,:])[0].item());pol,finish,trace=coordinate_polish(bad_t,scorer,40)
    nondecreasing=all(trace[i+1]>=trace[i]-1e-8 for i in range(len(trace)-1));improved=finish>start+1e-8
    rec_start=float(np.mean(bad==truth));rec_finish=float(np.mean(pol.cpu().numpy()==truth))
    out={'device':torch.cuda.get_device_name(0),'gpu_cpu_max_abs_diff':diff,'scorer_gate':scorer_ok,'corrupted_positions':sorted(int(x) for x in pos),'start_fit':start,'finish_fit':finish,'coordinate_accepts':len(trace)-1,'likelihood_nondecreasing':nondecreasing,'improved':improved,'map_recovery_start':rec_start,'map_recovery_finish':rec_finish,'pass':bool(scorer_ok and nondecreasing and improved)}
    print('VBM_V10_GPU_SMOKE='+json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
