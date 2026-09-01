# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, time, urllib.request
import numpy as np
import torch
import triton
import triton.language as tl

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a.py'
with urllib.request.urlopen(BASE,timeout=120) as r: _src=r.read().decode('utf-8')
B={'__name__':'v10base'}; exec(compile(_src,BASE,'exec'),B)

VOW=B['VOW']; NV=B['NV']; KB=B['KB']; KR=B['KR']; KN=B['KN']; ALPHA=B['ALPHA']
SIZES=[100,250,500,1000,2000]
BRIDGE_BLOCK=8; NUC_BLOCK=4; SWEEPS=3; CHAINS=8; BATCH=32768
NS='VBMV10GPUV2'


def hseed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff


def dense_lm(A):
    nctx=26**4
    out=np.full((nctx,26),math.log(1/26),dtype=np.float32)
    def cidx(s):
        z=0
        for ch in s:z=z*26+(ord(ch)-97)
        return z
    for ctx,counter in A['lm'].ctx.items():
        ix=cidx(ctx); tot=A['lm'].tot[ctx]
        for j in range(26):
            ch=chr(97+j);out[ix,j]=math.log((counter[ch]+ALPHA)/(tot+26*ALPHA))
    return out.reshape(-1)


def pack_assets(A,fit):
    run_chars=np.zeros((KR,5),dtype=np.uint8);run_lens=np.zeros(KR,dtype=np.uint8)
    for i,r in enumerate(A['runs']):
        run_lens[i]=len(r)
        for j,ch in enumerate(r):run_chars[i,j]=ord(ch)-97
    vow=np.asarray([ord(x)-97 for x in VOW],dtype=np.uint8)
    et=[];es=[];starts=[];lens=[]
    for L in fit:
        starts.append(len(et));st=len(et)
        for i,n in enumerate(L['n']):
            if n>=0:et.append(2);es.append(int(n))
            if i<len(L['b']):et.append(1);es.append(int(L['b'][i]))
        lens.append(len(et)-st)
    if max(lens,default=0)>32:raise RuntimeError(('MAXEV',max(lens)))
    return dict(etype=np.asarray(et,dtype=np.uint8),esurf=np.asarray(es,dtype=np.int16),starts=np.asarray(starts,dtype=np.int32),lens=np.asarray(lens,dtype=np.int32),run_chars=run_chars.reshape(-1),run_lens=run_lens,vow=vow)


@triton.jit
def block_score_kernel(cid_base, radix, blen, etype, esurf, lstart, llen,
                       runchars, runlens, vowchars, logp, bmap, nmap, bpos, npos, pows,
                       llout, nout, BCAND, BLOCK:tl.constexpr, MAXEV:tl.constexpr):
    pb=tl.program_id(0);line=tl.program_id(1)
    offs=pb*BLOCK+tl.arange(0,BLOCK);mask=offs<BCAND
    cid=(cid_base+offs).to(tl.int64)
    st=tl.load(lstart+line).to(tl.int32);ln=tl.load(llen+line).to(tl.int32)
    ctx=tl.zeros((BLOCK,),tl.int64);have=tl.zeros((BLOCK,),tl.int32);ll=tl.zeros((BLOCK,),tl.float32);ns=tl.zeros((BLOCK,),tl.int32)
    for jj in range(MAXEV):
        em=jj<ln
        typ=tl.load(etype+st+jj,mask=em,other=0).to(tl.int32)
        surf=tl.load(esurf+st+jj,mask=em,other=0).to(tl.int32)
        # bridge
        bm=mask & em & (typ==1)
        bp=tl.load(bpos+surf,mask=bm,other=-1).to(tl.int32)
        bv0=tl.load(bmap+surf,mask=bm,other=0).to(tl.int64)
        pw=tl.load(pows+bp,mask=bm & (bp>=0),other=1).to(tl.int64)
        dig=(cid//pw)%radix
        bv=tl.where(bp>=0,dig,bv0).to(tl.int32)
        bc=tl.load(vowchars+bv,mask=bm,other=0).to(tl.int64)
        sm=bm & (have>=4)
        lp=tl.load(logp+ctx*26+bc,mask=sm,other=0.0);ll+=lp;ns+=sm.to(tl.int32)
        nc=tl.where(have>=4,(ctx%17576)*26+bc,ctx*26+bc);ctx=tl.where(bm,nc,ctx);have=tl.where(bm,tl.minimum(have+1,4),have)
        # nucleus
        nm=mask & em & (typ==2)
        np=tl.load(npos+surf,mask=nm,other=-1).to(tl.int32)
        nv0=tl.load(nmap+surf,mask=nm,other=0).to(tl.int64)
        pw2=tl.load(pows+np,mask=nm & (np>=0),other=1).to(tl.int64)
        dig2=(cid//pw2)%radix
        nv=tl.where(np>=0,dig2,nv0).to(tl.int32)
        rl=tl.load(runlens+nv,mask=nm,other=0).to(tl.int32)
        for kk in range(5):
            cm=nm & (kk<rl)
            cc=tl.load(runchars+nv*5+kk,mask=cm,other=0).to(tl.int64)
            sm2=cm & (have>=4)
            lp2=tl.load(logp+ctx*26+cc,mask=sm2,other=0.0);ll+=lp2;ns+=sm2.to(tl.int32)
            nc2=tl.where(have>=4,(ctx%17576)*26+cc,ctx*26+cc);ctx=tl.where(cm,nc2,ctx);have=tl.where(cm,tl.minimum(have+1,4),have)
    oi=line*BCAND+offs
    tl.store(llout+oi,ll,mask=mask);tl.store(nout+oi,ns,mask=mask)


class Engine:
    def __init__(self,device,A,fit,logp_np,packed):
        self.device=device;self.A=A;self.fit=fit;self.L=len(fit)
        with torch.cuda.device(device):
            d=f'cuda:{device}'
            self.et=torch.tensor(packed['etype'],dtype=torch.uint8,device=d);self.es=torch.tensor(packed['esurf'],dtype=torch.int16,device=d)
            self.st=torch.tensor(packed['starts'],dtype=torch.int32,device=d);self.ln=torch.tensor(packed['lens'],dtype=torch.int32,device=d)
            self.rc=torch.tensor(packed['run_chars'],dtype=torch.uint8,device=d);self.rl=torch.tensor(packed['run_lens'],dtype=torch.uint8,device=d);self.vow=torch.tensor(packed['vow'],dtype=torch.uint8,device=d)
            self.logp=torch.tensor(logp_np,dtype=torch.float32,device=d)
            # execution-only warmup
            z=np.zeros(KB,dtype=np.int16);n=np.zeros(KN,dtype=np.int16)
            self.score_block(z,n,'b',[0])

    def score_block(self,bmap,nmap,fam,block):
        radix=5 if fam=='b' else 32; blen=len(block); total=radix**blen
        bpos=np.full(KB,-1,dtype=np.int16);npos=np.full(KN,-1,dtype=np.int16)
        if fam=='b':
            for p,s in enumerate(block):bpos[s]=p
        else:
            for p,s in enumerate(block):npos[s]=p
        # Candidate integer order is tuple lexicographic: first block variable is most significant.
        pows=np.ones(max(1,blen),dtype=np.int64)
        for p in range(blen):pows[p]=radix**(blen-1-p)
        with torch.cuda.device(self.device):
            d=f'cuda:{self.device}';bt=torch.tensor(bmap,dtype=torch.int16,device=d);nt=torch.tensor(nmap,dtype=torch.int16,device=d)
            bp=torch.tensor(bpos,dtype=torch.int16,device=d);np0=torch.tensor(npos,dtype=torch.int16,device=d);pw=torch.tensor(pows,dtype=torch.int64,device=d)
            bestscore=-1e300;bestcid=0;base=0;rem=total;TW=128
            while rem:
                q=min(BATCH,rem);llo=torch.empty((self.L,q),dtype=torch.float32,device=d);no=torch.empty((self.L,q),dtype=torch.int32,device=d)
                grid=(triton.cdiv(q,TW),self.L)
                block_score_kernel[grid](base,radix,blen,self.et,self.es,self.st,self.ln,self.rc,self.rl,self.vow,self.logp,bt,nt,bp,np0,pw,llo,no,q,BLOCK=TW,MAXEV=32,num_warps=4)
                sc=llo.sum(0)/no.sum(0).clamp_min(1)
                val,ix=torch.max(sc,dim=0);v=float(val.item());i=int(ix.item())
                # Strict greater retains earliest cid under exact ties.
                if v>bestscore:
                    bestscore=v;bestcid=base+i
                base+=q;rem-=q
            vals=[];x=bestcid
            for p in range(blen):
                div=radix**(blen-1-p);vals.append((x//div)%radix)
            return bestscore,vals,total


def counts(fit):
    bc=np.zeros(KB,dtype=np.int64);nc=np.zeros(KN,dtype=np.int64)
    for L in fit:
        for x in L['b']:bc[x]+=1
        for x in L['n']:
            if x>=0:nc[x]+=1
    return bc,nc


def ordered_ids(cnt,fixed):
    return sorted([i for i,c in enumerate(cnt) if c>0 and i not in fixed],key=lambda i:(-int(cnt[i]),i))


def chunks(order,bs):return [order[i:i+bs] for i in range(0,len(order),bs)]


def partition(order,bs,sweep,stage,lang,rep,size,chain,fam):
    if sweep==0:return chunks(order,bs)
    if sweep==1:
        k=math.ceil(len(order)/bs);idx=[]
        for off in range(k):
            for j in range(off,len(order),k):idx.append(j)
        return chunks([order[j] for j in idx],bs)
    q=list(order);rng=np.random.default_rng(hseed('VBMV10_GPU_BLOCK_SWEEP3',stage,lang,rep,size,chain,fam));rng.shuffle(q);return chunks(q,bs)


def apply_block(arr,block,vals):
    for s,v in zip(block,vals):arr[s]=v


def run_chain(device,chain,A,fit,key,lang,rep,size,oracle,logp_np,packed,bc,nc,fixed_b,fixed_n):
    eng=Engine(device,A,fit,logp_np,packed)
    bm,nm=B['init_map'](A,f'V10GPU:{lang}:R{rep}:N{size}:{oracle}:C{chain}',key,fixed_b,fixed_n)
    bm=np.asarray(bm,dtype=np.int16);nm=np.asarray(nm,dtype=np.int16)
    ob=ordered_ids(bc,fixed_b);on=ordered_ids(nc,fixed_n);evaluated=0;t0=time.time()
    # pre single-site exact polish, descending frequency
    for s in ob:
        _,v,k=eng.score_block(bm,nm,'b',[s]);apply_block(bm,[s],v);evaluated+=k
    for s in on:
        _,v,k=eng.score_block(bm,nm,'n',[s]);apply_block(nm,[s],v);evaluated+=k
    # three exact block sweeps
    for sw in range(SWEEPS):
        for bl in partition(ob,BRIDGE_BLOCK,sw,'A',lang,rep,size,chain,'bridge'):
            _,v,k=eng.score_block(bm,nm,'b',bl);apply_block(bm,bl,v);evaluated+=k
        for bl in partition(on,NUC_BLOCK,sw,'A',lang,rep,size,chain,'nucleus'):
            _,v,k=eng.score_block(bm,nm,'n',bl);apply_block(nm,bl,v);evaluated+=k
    # final single-site polish
    for s in ob:
        _,v,k=eng.score_block(bm,nm,'b',[s]);apply_block(bm,[s],v);evaluated+=k
    for s in on:
        _,v,k=eng.score_block(bm,nm,'n',[s]);apply_block(nm,[s],v);evaluated+=k
    m={'bmap':bm,'nmap':nm}
    fitscore=B['score_lines'](fit,A,m)
    return {'chain':chain,'device':device,'fit_score':float(fitscore),'bmap':bm.tolist(),'nmap':nm.tolist(),'candidates_evaluated':int(evaluated),'elapsed_s':time.time()-t0}


def fit_gpu(A,fit,key,lang,rep,size,oracle):
    bc,nc=counts(fit);ob=[i for i,c in enumerate(bc) if c>0];on=[i for i,c in enumerate(nc) if c>0]
    fixed_b=set();fixed_n=set()
    if oracle=='O1':
        rb=math.ceil(.25*len(ob));rn=math.ceil(.25*len(on))
        fixed_b=set(sorted(ob,key=lambda i:(-int(bc[i]),i))[:rb]);fixed_n=set(sorted(on,key=lambda i:(-int(nc[i]),i))[:rn])
    logp_np=dense_lm(A);packed=pack_assets(A,fit);ng=min(4,torch.cuda.device_count())
    if ng<1:raise RuntimeError('GPU required')
    def worker(dev):
        return [run_chain(dev,ch,A,fit,key,lang,rep,size,oracle,logp_np,packed,bc,nc,fixed_b,fixed_n) for ch in range(dev,CHAINS,ng)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=ng) as ex:
        nested=list(ex.map(worker,range(ng)))
    chains=[x for sub in nested for x in sub];chains.sort(key=lambda x:x['chain'])
    best=max(chains,key=lambda x:(x['fit_score'],-x['chain']))
    return {'bmap':np.asarray(best['bmap'],dtype=np.int16),'nmap':np.asarray(best['nmap'],dtype=np.int16),'fit_score':best['fit_score']},chains,bc,nc,sorted(fixed_b),sorted(fixed_n)


def recovery(m,key,hold,bc,nc,A,minfit=0):
    bcor=btot=ncor=ntot=ccor=ctot=0
    for L in hold:
        for i,n in enumerate(L['n']):
            if n>=0 and nc[n]>=minfit:
                tr=A['runs'][int(key['nmap'][n])];pr=A['runs'][int(m['nmap'][n])];ntot+=1;ncor+=int(tr==pr)
                M=max(len(tr),len(pr));ctot+=M;ccor+=sum(j<len(tr) and j<len(pr) and tr[j]==pr[j] for j in range(M))
            if i<len(L['b']):
                x=L['b'][i]
                if bc[x]>=minfit:
                    btot+=1;bcor+=int(m['bmap'][x]==key['bmap'][x]);ctot+=1;ccor+=int(m['bmap'][x]==key['bmap'][x])
    return {'REC_B':bcor/max(1,btot),'REC_N':ncor/max(1,ntot),'REC_CHAR':ccor/max(1,ctot),'B_events':btot,'N_events':ntot,'CHAR_slots':ctot}


def coverage(hold,bc,nc):
    be=[x for L in hold for x in L['b']];ne=[x for L in hold for x in L['n'] if x>=0]
    return {'COV_B':sum(bc[x]>0 for x in be)/max(1,len(be)),'COV_N':sum(nc[x]>0 for x in ne)/max(1,len(ne))}


def random_hold(A,hold,tag):
    vals=[]
    for r in range(20):
        bm,nm=B['init_map'](A,f'{tag}:RAND:{r}');vals.append(B['score_lines'](hold,A,{'bmap':bm,'nmap':nm}))
    return float(np.median(vals))


def one(lang,rep,size,A,all_lines,key):
    cur=all_lines[:size];cut=int(size*.8);fit=cur[:cut];hold=cur[cut:]
    bc,nc=counts(fit)
    true={'bmap':key['bmap'],'nmap':key['nmap']}
    o0=recovery(true,key,hold,bc,nc,A,0);o05=recovery(true,key,hold,bc,nc,A,5);o0.update({k+'5':v for k,v in o05.items() if k.startswith('REC_')});o0['HOLD_LM']=B['score_lines'](hold,A,true)
    base=random_hold(A,hold,f'A:{lang}:R{rep}:N{size}')
    out={'lang':lang,'rep':rep,'size':size,'fit_lines':len(fit),'hold_lines':len(hold),'O0_TRUE_KEY':o0,'RAND_HOLD_LM':base}
    for oracle in ['O1','O2']:
        t=time.time();m,chains,bc,nc,fb,fn=fit_gpu(A,fit,key,lang,rep,size,oracle)
        r=recovery(m,key,hold,bc,nc,A,0);r5=recovery(m,key,hold,bc,nc,A,5);r.update({'REC_B5':r5['REC_B'],'REC_N5':r5['REC_N'],'REC_CHAR5':r5['REC_CHAR']});r.update(coverage(hold,bc,nc))
        hl=B['score_lines'](hold,A,m);r['HOLD_LM']=float(hl);r['HOLD_ADV']=float(hl-base);r['FIT_LM']=float(m['fit_score']);r['revealed_B']=len(fb);r['revealed_N']=len(fn);r['elapsed_s']=time.time()-t
        r['best_chain']=max(chains,key=lambda x:(x['fit_score'],-x['chain']))['chain'];r['chain_fit_scores']=[x['fit_score'] for x in chains];r['candidate_evals_total']=sum(x['candidates_evaluated'] for x in chains)
        out[oracle]=r
    return out


def gate(rows,size):
    z=[r for r in rows if r['size']==size];o=[r['O2'] for r in z]
    c1=sum(x['REC_CHAR']>=.80 for x in o);c2=sum(x['REC_B']>=.70 and x['REC_N']>=.70 for x in o);c4=sum(x['REC_CHAR5']>=.90 and x['REC_B5']>=.80 and x['REC_N5']>=.80 for x in o)
    lang={}
    for la in ['DE','IT']:
        q=[r['O2'] for r in z if r['lang']==la];lang[la]={'char_pass':sum(x['REC_CHAR']>=.80 for x in q),'key_pass':sum(x['REC_B']>=.70 and x['REC_N']>=.70 for x in q)}
    first4=(c1>=5 and c2>=5 and c4>=5 and all(v['char_pass']>=2 and v['key_pass']>=2 for v in lang.values()))
    return {'size':size,'n':len(z),'char_pass':c1,'key_pass':c2,'frequent_pass':c4,'by_language':lang,'FIRST_FOUR_RECOVERY_GATES_PASS':first4}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lang',choices=['DE','IT','ALL'],default='ALL');ap.add_argument('--rep',default='ALL');ap.add_argument('--sizes',default='100,250,500,1000,2000');args=ap.parse_args()
    langs=['DE','IT'] if args.lang=='ALL' else [args.lang];reps=range(3) if args.rep=='ALL' else [int(args.rep)];sizes=[int(x) for x in args.sizes.split(',')]
    meta={'protocol':'VBM_V10_TERMINAL_IDENTIFIABILITY_PROTOCOL.md','impl':'VBM_V10_IMPLEMENTATION_SPEC.md','gpu_addendum':'VBM_V10_GPU_SEARCH_ADDENDUM_V2.md','bridge_block':BRIDGE_BLOCK,'nucleus_block':NUC_BLOCK,'sweeps':SWEEPS,'chains':CHAINS,'gpus_visible':torch.cuda.device_count()}
    print('V10_GPU_META='+json.dumps(meta,sort_keys=True),flush=True)
    rows=[]
    for la in langs:
        A=B['assets'](la)
        for rep in reps:
            all_lines,key=B['make_positive'](la,rep,A)
            for size in sizes:
                print('V10_START='+json.dumps({'lang':la,'rep':rep,'size':size}),flush=True)
                r=one(la,rep,size,A,all_lines,key);rows.append(r)
                print('V10_POS_RESULT='+json.dumps(r,sort_keys=True),flush=True)
    gates=[gate(rows,s) for s in sizes if len([r for r in rows if r['size']==s])==6]
    if gates:
        print('V10_STAGE_A_FIRST4='+json.dumps(gates,sort_keys=True),flush=True)
        if not any(g['FIRST_FOUR_RECOVERY_GATES_PASS'] for g in gates) and max(sizes)>=2000:
            print('V10_TERMINAL_PRELIM='+json.dumps({'verdict':'VBM_GLOBAL_KEY_NOT_RECOVERABLE_EVEN_COMPACT','reason':'no Stage-A size through 2000 satisfies first four O2 recovery gates; adversaries/stability cannot rescue conjunctive gate','stage_B_opened':False},sort_keys=True),flush=True)

if __name__=='__main__':main()
