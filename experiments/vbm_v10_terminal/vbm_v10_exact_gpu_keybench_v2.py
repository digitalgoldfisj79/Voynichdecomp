# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "cupy-cuda12x>=13,<14"]
# ///
from __future__ import annotations
import argparse, json, math, time, urllib.request
import numpy as np
import cupy as cp

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
with urllib.request.urlopen(BASE,timeout=120) as r: src=r.read().decode('utf-8')
ns={'__name__':'v10base'}; exec(compile(src,BASE,'exec'),ns)

ap=argparse.ArgumentParser(); ap.add_argument('--size',type=int,default=2000); ap.add_argument('--candidates',type=int,default=2500000); ap.add_argument('--batch',type=int,default=65536); a=ap.parse_args()
LANG='DE'; rep=0
A=ns['assets'](LANG); lines,key=ns['make_positive'](LANG,rep,A); fit=lines[:int(a.size*0.8)]
ALPHA=ns['ALPHA']; VOW=ns['VOW']; KB=ns['KB']; KN=ns['KN']; KR=ns['KR']

NCTX=26**4; logp=np.full((NCTX,26),math.log(1/26),dtype=np.float32)
def cidx(s):
    z=0
    for ch in s: z=z*26+(ord(ch)-97)
    return z
for ctx,counter in A['lm'].ctx.items():
    ix=cidx(ctx); tot=A['lm'].tot[ctx]
    for j in range(26):
        ch=chr(97+j); logp[ix,j]=math.log((counter[ch]+ALPHA)/(tot+26*ALPHA))
logp_gpu=cp.asarray(logp.ravel())
run_chars=np.zeros((KR,5),dtype=np.uint8); run_lens=np.zeros(KR,dtype=np.uint8)
for i,r in enumerate(A['runs']):
    run_lens[i]=len(r)
    for j,ch in enumerate(r): run_chars[i,j]=ord(ch)-97
run_chars_gpu=cp.asarray(run_chars.ravel()); run_lens_gpu=cp.asarray(run_lens); vowel_chars_gpu=cp.asarray(np.array([ord(ch)-97 for ch in VOW],dtype=np.uint8))

types=[]; surfs=[]
for L in fit:
    types.append(0); surfs.append(0)
    for i,n in enumerate(L['n']):
        if n>=0: types.append(2); surfs.append(int(n))
        if i<len(L['b']): types.append(1); surfs.append(int(L['b'][i]))
types_gpu=cp.asarray(np.asarray(types,dtype=np.uint8)); surfs_gpu=cp.asarray(np.asarray(surfs,dtype=np.int16)); E=len(types)

kernel=cp.RawKernel(r'''
__device__ __forceinline__ unsigned int mix32(unsigned int x){
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
extern "C" __global__
void score_keys(unsigned long long base_id, int B,
                const unsigned char* etype, const short* esurf, int E,
                const unsigned char* runchars, const unsigned char* runlens,
                const unsigned char* vowchars, const float* logp,
                float* out) {
    int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=B) return;
    unsigned long long cid=base_id+(unsigned long long)i;
    unsigned int ctx=0; int have=0; float ll=0.0f; int nscore=0;
    #define BMVAL(ss) ((unsigned char)(mix32((unsigned int)(cid ^ (0x9e3779b9ULL*((unsigned int)(ss)+1U)))) % 5U))
    #define NMVAL(ss) ((unsigned char)(mix32((unsigned int)((cid>>17) ^ (0x85ebca6bULL*((unsigned int)(ss)+1U)))) % 32U))
    #define PUSHCHAR(cc) { unsigned int c=(unsigned int)(cc); if(have>=4){ ll += logp[((long long)ctx)*26 + c]; nscore++; ctx=((ctx % 17576u)*26u)+c; } else { ctx=ctx*26u+c; have++; } }
    for(int e=0;e<E;e++){
        unsigned char t=etype[e]; short s=esurf[e];
        if(t==0){ ctx=0; have=0; continue; }
        if(t==1){ unsigned char v=BMVAL((int)s); PUSHCHAR(vowchars[(int)v]); }
        else { unsigned char rv=NMVAL((int)s); unsigned char ln=runlens[(int)rv]; const unsigned char* p=runchars+((int)rv)*5; for(int j=0;j<(int)ln;j++){ PUSHCHAR(p[j]); } }
    }
    out[i]=ll/((float)(nscore>0?nscore:1));
}
''','score_keys')

wb=min(a.batch,8192); out=cp.empty(wb,dtype=cp.float32); kernel(((wb+255)//256,),(256,),(np.uint64(0),wb,types_gpu,surfs_gpu,E,run_chars_gpu,run_lens_gpu,vowel_chars_gpu,logp_gpu,out)); cp.cuda.Stream.null.synchronize()
remaining=a.candidates; total=0; base=0; t0=time.time(); best=-1e30
while remaining>0:
    B=min(a.batch,remaining); out=cp.empty(B,dtype=cp.float32)
    kernel(((B+255)//256,),(256,),(np.uint64(base),B,types_gpu,surfs_gpu,E,run_chars_gpu,run_lens_gpu,vowel_chars_gpu,logp_gpu,out)); best=max(best,float(out.max().get())); total+=B; remaining-=B; base+=B
cp.cuda.Stream.null.synchronize(); elapsed=time.time()-t0; rate=total/elapsed
props=cp.cuda.runtime.getDeviceProperties(0); name=props['name'].decode() if isinstance(props['name'],bytes) else str(props['name'])
print('EXACT_KEY_BENCH='+json.dumps({'device':name,'fit_lines':len(fit),'events':E,'candidates':total,'elapsed_s':elapsed,'cand_per_s':rate,'cand_10min':rate*600,'best_score':best,'batch':a.batch},sort_keys=True),flush=True)
