# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, json, math, time, urllib.request
import numpy as np
import torch
import triton
import triton.language as tl

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
with urllib.request.urlopen(BASE,timeout=120) as r: src=r.read().decode('utf-8')
ns={'__name__':'v10base'}; exec(compile(src,BASE,'exec'),ns)

ap=argparse.ArgumentParser(); ap.add_argument('--size',type=int,default=2000); ap.add_argument('--candidates',type=int,default=2500000); ap.add_argument('--batch',type=int,default=32768); a=ap.parse_args()
A=ns['assets']('DE'); lines,key=ns['make_positive']('DE',0,A); fit=lines[:int(a.size*0.8)]
ALPHA=ns['ALPHA']; VOW=ns['VOW']; KR=ns['KR']

# Dense exact 4-char-context / next-char log-probability table for the frozen 5-gram LM.
NCTX=26**4; logp=np.full((NCTX,26),math.log(1/26),dtype=np.float32)
def cidx(s):
    z=0
    for ch in s: z=z*26+(ord(ch)-97)
    return z
for ctx,counter in A['lm'].ctx.items():
    ix=cidx(ctx); tot=A['lm'].tot[ctx]
    for j in range(26):
        ch=chr(97+j); logp[ix,j]=math.log((counter[ch]+ALPHA)/(tot+26*ALPHA))

run_chars=np.zeros((KR,5),dtype=np.uint8); run_lens=np.zeros(KR,dtype=np.uint8)
for i,r in enumerate(A['runs']):
    run_lens[i]=len(r)
    for j,ch in enumerate(r): run_chars[i,j]=ord(ch)-97
vowel_chars=np.array([ord(ch)-97 for ch in VOW],dtype=np.uint8)

# Per-line event stream: 1 bridge, 2 nonempty nucleus. Empty nuclei emit nothing.
et=[]; es=[]; starts=[]; lens=[]
for L in fit:
    starts.append(len(et)); n0=len(et)
    for i,n in enumerate(L['n']):
        if n>=0: et.append(2); es.append(int(n))
        if i<len(L['b']): et.append(1); es.append(int(L['b'][i]))
    lens.append(len(et)-n0)
maxev=max(lens)
if maxev>32: raise RuntimeError(('MAXEV',maxev))

dev='cuda'; et_t=torch.tensor(et,dtype=torch.uint8,device=dev); es_t=torch.tensor(es,dtype=torch.int16,device=dev)
starts_t=torch.tensor(starts,dtype=torch.int32,device=dev); lens_t=torch.tensor(lens,dtype=torch.int32,device=dev)
run_chars_t=torch.tensor(run_chars.reshape(-1),dtype=torch.uint8,device=dev); run_lens_t=torch.tensor(run_lens,dtype=torch.uint8,device=dev)
vow_t=torch.tensor(vowel_chars,dtype=torch.uint8,device=dev); logp_t=torch.tensor(logp.reshape(-1),dtype=torch.float32,device=dev)

@triton.jit
def mix32(x):
    x = x ^ (x >> 16)
    x = x * 0x7feb352d
    x = x ^ (x >> 15)
    x = x * 0x846ca68b
    x = x ^ (x >> 16)
    return x

@triton.jit
def score_kernel(cid_base, etype, esurf, lstart, llen, runchars, runlens, vowchars, logp, llout, nout, B, BLOCK: tl.constexpr, MAXEV: tl.constexpr):
    pb=tl.program_id(0); line=tl.program_id(1)
    offs=pb*BLOCK+tl.arange(0,BLOCK); mask=offs<B
    cid=(cid_base+offs).to(tl.uint32)
    st=tl.load(lstart+line).to(tl.int32); ln=tl.load(llen+line).to(tl.int32)
    ctx=tl.zeros((BLOCK,),tl.uint32); have=tl.zeros((BLOCK,),tl.int32); ll=tl.zeros((BLOCK,),tl.float32); ns=tl.zeros((BLOCK,),tl.int32)
    for jj in range(MAXEV):
        evmask=jj<ln
        t=tl.load(etype+st+jj,mask=evmask,other=0).to(tl.int32)
        s=tl.load(esurf+st+jj,mask=evmask,other=0).to(tl.uint32)
        # bridge push
        bm=mask & evmask & (t==1)
        hv=mix32(cid ^ ((s+1)*0x9e3779b9))
        bv=(hv % 5).to(tl.int32)
        bc=tl.load(vowchars+bv,mask=bm,other=0).to(tl.uint32)
        sm=bm & (have>=4)
        lp=tl.load(logp+ctx.to(tl.int64)*26+bc.to(tl.int64),mask=sm,other=0.0)
        ll += lp; ns += sm.to(tl.int32)
        nctx=tl.where(have>=4,(ctx % 17576)*26+bc,ctx*26+bc)
        ctx=tl.where(bm,nctx,ctx); have=tl.where(bm,tl.minimum(have+1,4),have)
        # nucleus: up to 5 consonants
        nm=mix32((cid*747796405) ^ ((s+1)*2891336453)) & 31
        rlen=tl.load(runlens+nm.to(tl.int32),mask=mask & evmask & (t==2),other=0).to(tl.int32)
        for kk in range(5):
            cm=mask & evmask & (t==2) & (kk<rlen)
            cc=tl.load(runchars+nm.to(tl.int32)*5+kk,mask=cm,other=0).to(tl.uint32)
            sm2=cm & (have>=4)
            lp2=tl.load(logp+ctx.to(tl.int64)*26+cc.to(tl.int64),mask=sm2,other=0.0)
            ll += lp2; ns += sm2.to(tl.int32)
            nctx2=tl.where(have>=4,(ctx % 17576)*26+cc,ctx*26+cc)
            ctx=tl.where(cm,nctx2,ctx); have=tl.where(cm,tl.minimum(have+1,4),have)
    outidx=line*B+offs
    tl.store(llout+outidx,ll,mask=mask); tl.store(nout+outidx,ns,mask=mask)

L=len(fit); BLOCK=128; MAXEV=32

def score_batch(base,B):
    llm=torch.empty((L,B),dtype=torch.float32,device=dev); nm=torch.empty((L,B),dtype=torch.int32,device=dev)
    grid=(triton.cdiv(B,BLOCK),L)
    score_kernel[grid](base,et_t,es_t,starts_t,lens_t,run_chars_t,run_lens_t,vow_t,logp_t,llm,nm,B,BLOCK=BLOCK,MAXEV=MAXEV,num_warps=4)
    ll=llm.sum(dim=0); nn=nm.sum(dim=0).clamp_min(1); return ll/nn

# warm-up / compile
z=score_batch(0,min(1024,a.batch)); _=float(z.max().item()); torch.cuda.synchronize()
remaining=a.candidates; total=0; base=0; best=-1e30; t0=time.time()
while remaining>0:
    B=min(a.batch,remaining); z=score_batch(base,B); best=max(best,float(z.max().item())); total+=B; remaining-=B; base+=B
torch.cuda.synchronize(); elapsed=time.time()-t0; rate=total/elapsed
p=torch.cuda.get_device_properties(0)
print('EXACT_TRITON_BENCH='+json.dumps({'device':p.name,'fit_lines':L,'events':len(et),'max_events_per_line':maxev,'candidates':total,'elapsed_s':elapsed,'cand_per_s':rate,'cand_10min':rate*600,'best_score':best,'batch':a.batch},sort_keys=True),flush=True)
