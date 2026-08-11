# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,json,statistics,tempfile,os,subprocess
import numpy as np
import babuini_core_v1 as c
import sta_representation_census as rc
import amadi_residuals_v1 as ar

NS='AMADIBABUINISTAV1DEV'
c.NS=NS
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}

def seed(*x):return int.from_bytes(hashlib.sha256('::'.join(map(str,x)).encode()).digest()[:8],'big')&0x7fffffff

def setup(strong=True,smoke=False):
    if smoke:
        c.PROPS=8000;c.RESTARTS=4;c.BATCH=2
    elif strong:
        c.PROPS=100000;c.RESTARTS=24;c.BATCH=6
    else:
        c.PROPS=40000;c.RESTARTS=12;c.BATCH=4

def direct_control(lm,tag,fitn,holdn,smoke=False):
    fw,hw=c.span(lm.control,tag,fitn,holdn);fc,inv=c.encrypt(fw,tag);p2c=np.empty(c.SIG,np.int32)
    for surf,plain in enumerate(inv):p2c[plain]=surf
    hc=[[int(p2c[x]) for x in w] for w in hw];sol=c.solve(c.stats(fc),lm,tag,smoke);hs=c.fixed(c.stats(hc),lm,sol['dec'])
    return {'score':hs,'recovery':c.recovery(hw,hc,sol['dec']),'agreement':sol['agreement'],'converged':sol['converged'],'score_diff':sol['score_diff'],'restarts_each':sol['restarts_each']}

def neg_words(kind,rep,total=16000):
    rg=np.random.default_rng(seed(NS,'neg',kind,rep));lens=np.array([1,2,2,3,3,3,4,4,5,6,7],int);p=np.arange(c.SIG,0,-1,dtype=float);p/=p.sum();out=[];n=0;prev=int(rg.choice(c.SIG,p=p))
    while n<total:
        L=int(rg.choice(lens));w=[]
        if kind=='iid':w=[int(x) for x in rg.choice(c.SIG,L,p=p)]
        elif kind=='markov':
            for j in range(L):
                if rg.random()<.72:x=(prev+1+int(rg.integers(0,7))+rep)%c.SIG
                else:x=int(rg.choice(c.SIG,p=p))
                w.append(x);prev=x
        elif kind=='motif':
            mot=[int(x) for x in rg.choice(c.SIG,max(1,min(3,L)),p=p)];w=(mot*((L+len(mot)-1)//len(mot)))[:L]
            for j in range(L):
                if rg.random()<.1:w[j]=int(rg.choice(c.SIG,p=p))
        elif kind=='copy':
            if out and rg.random()<.75:
                w=(list(out[int(rg.integers(0,len(out)))])+[int(rg.choice(c.SIG,p=p))]*L)[:L]
                for j in range(len(w)):
                    if rg.random()<.15:w[j]=int(rg.choice(c.SIG,p=p))
            else:w=[int(x) for x in rg.choice(c.SIG,L,p=p)]
        elif kind=='slot':
            bands=[list(range(0,22)),list(range(22,45)),list(range(45,67)),list(range(67,89))];w=[int(rg.choice(bands[j%4])) for j in range(L)]
        out.append(w);n+=L
    return out

def neg_eval(lm,kind,rep,smoke=False):
    w=neg_words(kind,rep,3000 if smoke else 16000);cut=len(w)//2;f=w[:cut];h=w[cut:];sol=c.solve(c.stats(f),lm,f'N:{kind}:{rep}',smoke);return {'kind':kind,'rep':rep,'score':c.fixed(c.stats(h),lm,sol['dec']),'converged':sol['converged'],'agreement':sol['agreement']}

def load_sta_aaa():
    sb=rc.get(rc.STA_URL,rc.STA_SHA);bb=rc.get(rc.BIT_URL,rc.BIT_SHA);mb=rc.get(rc.MAP_URL,rc.MAP_SHA);td=tempfile.mkdtemp(prefix='babsta_');sp=os.path.join(td,'RF.txt');bp=os.path.join(td,'bitrans.c');mp=os.path.join(td,'STA-aaa.bit');ap=os.path.join(td,'RF.aaa.txt');open(sp,'wb').write(sb);open(bp,'wb').write(bb);open(mp,'wb').write(mb);exe=os.path.join(td,'bitrans');subprocess.run(['gcc','-O2','-o',exe,bp],check=True);q=subprocess.run([exe,'-1','-m2','-f',mp,sp,ap],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert q.returncode==0,q.stderr
    return rc.parse_sta(sb.decode('utf-8','replace')),rc.parse_aaa(open(ap,encoding='utf-8').read())

def select_vocab(pages,fitfolios,K=89):
    C=collections.Counter()
    for f in fitfolios:
        for w in pages.get(f,[]):C.update(w)
    v=sorted(C,key=lambda x:(-C[x],x))[:K];return v

def project(pages,folios,vocab):
    V={x:i for i,x in enumerate(vocab)};out=[];tot=keep=words=kw=0
    for f in folios:
        for w in pages.get(f,[]):
            words+=1;tot+=len(w)
            if all(x in V for x in w):out.append([V[x] for x in w]);keep+=len(w);kw+=1
    return out,{'whole_word_char_coverage':keep/max(1,tot),'word_coverage':kw/max(1,words),'retained_events':keep,'recognized_events':tot,'retained_words':kw,'words':words}

def splits():
    rf,_=ar.parse_rf();T,H,C1,H2,C2=ar.target_split(rf);fit=T+H;z=sorted(C2,key=lambda f:hashlib.sha256(f'{NS}split::{f}'.encode()).digest());n=len(z)//2;return fit,z[:n],z[n:]

def target_rep(lm,pages,rep,fit,h1,c1):
    v=select_vocab(pages,fit,89);fw,fcov=project(pages,fit,v);hw,hcov=project(pages,h1,v);_,ccov=project(pages,c1,v)
    if fcov['whole_word_char_coverage']<.995 or hcov['whole_word_char_coverage']<.995:return {'rep':rep,'status':'SURFACE_INCOMPATIBLE','vocab':v,'fit_coverage':fcov,'H1_coverage':hcov,'C1_coverage_sealed_census_only':ccov}
    sol=c.solve(c.stats(fw),lm,f'TARGET:{rep}',False);hs=c.fixed(c.stats(hw),lm,sol['dec']);return {'rep':rep,'vocab':v,'fit_score':sol['fit_score'],'H1_score':hs,'agreement':sol['agreement'],'converged':sol['converged'],'fit_coverage':fcov,'H1_coverage':hcov,'C1_coverage_sealed_census_only':ccov}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['dev','qualify','manifest','target'],default='dev');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--qual-url');a=ap.parse_args();tr,ct=c.italian_words();lm=c.build_lm(tr,ct)
    if a.mode=='dev':
        setup(strong=False);rows=[direct_control(lm,f'DEV:{r}',12000,12000,False) for r in range(2)];print('RESULT_JSON',json.dumps({'rows':rows,'lm_meta':lm.meta},sort_keys=True));return
    if a.mode=='manifest':
        sta,aaa=load_sta_aaa();fit,h1,c1=splits();out={'fit':fit,'BAB_H1':h1,'BAB_C1':c1,'sta_vocab':select_vocab(sta,fit),'aaa_vocab':select_vocab(aaa,fit)};print('RESULT_JSON',json.dumps(out,sort_keys=True));return
    if a.mode=='qualify':
        c.NS=NS.replace('DEV','Q1');setup(strong=True);jobs=list(range(12));rows=[]
        def one(r):return direct_control(lm,f'Q1:{r}',18000,18000,False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for z in ex.map(one,jobs):rows.append(z);print('Q1',json.dumps(z,sort_keys=True),flush=True)
        floor=float(np.quantile([x['score'] for x in rows],.05));kinds=['iid','markov','motif','copy','slot'];nj=[(k,r) for k in kinds for r in range(12)];neg=[]
        def none(q):return neg_eval(lm,q[0],q[1],False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for z in ex.map(none,nj):neg.append(z);print('NEG',json.dumps(z,sort_keys=True),flush=True)
        fp=sum(z['score']>=floor for z in neg);summ={'ABS_FLOOR':floor,'median_recovery':statistics.median(x['recovery'] for x in rows),'min_recovery':min(x['recovery'] for x in rows),'all_converged':all(x['converged'] for x in rows),'min_agreement':min(x['agreement'] for x in rows),'false_positives':fp,'neg_trials':len(neg)};summ['pass']=bool(summ['median_recovery']>=.95 and summ['min_recovery']>=.85 and summ['all_converged'] and summ['min_agreement']>=.90 and fp<=2);print('RESULT_JSON',json.dumps({'summary':summ,'controls':rows,'negatives':neg,'lm_meta':lm.meta,'namespace':c.NS},sort_keys=True));return
    if not a.qual_url:raise RuntimeError('target needs --qual-url')
    q=json.loads(ar.getb(a.qual_url).decode());assert q['summary']['pass'];c.NS=NS.replace('DEV','TARGET');setup(strong=True);sta,aaa=load_sta_aaa();fit,h1,c1=splits();rows=[]
    for rep,p in [('sta',sta),('aaa',aaa)]:
        z=target_rep(lm,p,rep,fit,h1,c1);z['ABS_FLOOR']=q['summary']['ABS_FLOOR'];z['abs_pass']=z.get('H1_score',-1e300)>=z['ABS_FLOOR'];z['verdict']='BAB_H1_CANDIDATE' if z.get('converged') and z['abs_pass'] else ('UNRESOLVED_SEARCH' if z.get('converged') is False else 'CLOSED_NEGATIVE' if z.get('converged') else z.get('status','NO_RESULT'));rows.append(z);print('TARGET',json.dumps({k:v for k,v in z.items() if k!='vocab'},sort_keys=True),flush=True)
    unlock=all(z.get('verdict')=='BAB_H1_CANDIDATE' for z in rows);print('RESULT_JSON',json.dumps({'representations':rows,'BAB_C1_opened':False,'candidate_for_narrowing':unlock,'manifest':{'FIT_A':fit,'BAB_H1':h1,'BAB_C1':c1}},sort_keys=True))
if __name__=='__main__':main()
