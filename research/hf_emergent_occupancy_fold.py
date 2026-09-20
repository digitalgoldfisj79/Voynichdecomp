#!/usr/bin/env python3
import argparse, collections, hashlib, json, math, os, re, sys, urllib.request
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.decomposition import NMF
from sklearn.covariance import LedoitWolf

CORPUS_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
CORPUS_SHA='26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f'
ROWS_CANON_SHA='74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f'
FOLDS_CANON_SHA='e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888'
METRICS=['page_family_types_mean','page_family_types_sd','page_family_simpson_mean','page_family_newtype_slope_mean','bif_family_jaccard','same_section_nonmate_jaccard','k4_cont_jaccard','d4_jaccard','k4_gain_vs_global','bif_same_state']
VARIANTS=['A3_SG1_FULL','A4_SG1_FULL_NO_CODEBOOK']
NREP=200;BNULL=500;BETA=5.0
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF_BY_NUM={n:f'B{a:03d}_{b:03d}' for a,b in PAIRS for n in (a,b)}

def parse_num(f):
    m=re.match(r'f(\d+)',str(f));return int(m.group(1)) if m else None

def section(f):
    n=parse_num(f)
    if n is None:return 'UNK'
    if n<=66:return 'HERBAL'
    if n<=73:return 'ASTRO'
    if 75<=n<=84:return 'BIO'
    if 85<=n<=102:return 'PHARMA'
    if 103<=n<=116:return 'RECIPES'
    return 'UNK'

def davis_hand(folio,line_no):
    s=str(folio);n=parse_num(s);side='r' if 'r' in s else ('v' if 'v' in s else '');ln=int(line_no or 0)
    if n is None:return 'UNKNOWN'
    if n==115 and side=='r':return 'S2' if ln<=12 else 'S3'
    if 1<=n<=24:return 'S1'
    maps=[{25:'S1',26:'S2',27:'S1',28:'S1',29:'S1',30:'S1',31:'S2',32:'S1'},
          {33:'S2',34:'S2',35:'S1',36:'S1',37:'S1',38:'S1',39:'S2',40:'S2'},
          {41:'S5',42:'S1',43:'S2',44:'S1',45:'S1',46:'S2',47:'S1',48:'S5'},
          {49:'S1',50:'S2',51:'S1',52:'S1',53:'S1',54:'S1',55:'S2',56:'S1'}]
    for mp in maps:
        if n in mp:return mp[n]
    if n==57:return 'S1' if side=='v' else 'S5'
    if n in (58,65):return 'S3'
    if n==66:return 'S5'
    if 67<=n<=73:return 'S4'
    if 75<=n<=84:return 'S2'
    if 85<=n<=86:return 'MIXED_ROSE'
    if 87<=n<=90:return 'S1'
    if n==93:return 'S1'
    if n in (94,95):return 'S3'
    if n==96:return 'S1'
    if 99<=n<=102:return 'S1'
    if 103<=n<=116:return 'S3'
    return 'UNKNOWN'

def lenbin(t):
    n=len(t);return '12' if n<=2 else ('34' if n<=4 else ('56' if n<=6 else '7p'))
def family(t):return f'{t[0]}|{t[-1]}|{lenbin(t)}'
def posclass(r):
    if r['pos']==0:return 'FIRST'
    if r['pos']==r['line_len']-1:return 'LAST'
    return 'MID'

def build_events(obj):
    rows=[];eid=0
    for fol,ld in obj['pages'].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        for ls,rec in ld.items():
            if 'P' not in str(rec.get('u','')):continue
            txt=rec.get('t',{}).get('ZLZI','')
            toks=[t.lower() for t in txt.split() if re.fullmatch(r'[a-z]+',t.lower())]
            for pos,t in enumerate(toks):
                rows.append(dict(eid=eid,folio=fol,line=int(ls),pos=pos,line_len=len(toks),page=fol,bifolium=BIF_BY_NUM[n],section=section(fol),hand=davis_hand(fol,int(ls)),token=t,family=family(t)));eid+=1
    return rows

def fold_assignment(rows):
    by=collections.defaultdict(list)
    for r in rows:by[r['bifolium']].append(r)
    secs=sorted({r['section'] for r in rows});hands=sorted({r['hand'] for r in rows});feats={}
    for bif,rs in by.items():
        cs=collections.Counter(r['section'] for r in rs);ch=collections.Counter(r['hand'] for r in rs)
        feats[bif]=np.array([len(rs)]+[cs[x] for x in secs]+[ch[x] for x in hands],float)
    total=sum(feats.values(),np.zeros(1+len(secs)+len(hands)));target=total/5;scale=np.maximum(target,1)
    loads=[np.zeros_like(target) for _ in range(5)];counts=[0]*5;assign={}
    for bif in sorted(feats,key=lambda z:(-feats[z][0],z)):
        cand=[]
        for f in range(5):
            trial=[x.copy() for x in loads];trial[f]+=feats[bif];cc=counts.copy();cc[f]+=1
            sc=float(sum(np.sum(((x-target)/scale)**2) for x in trial))+0.002*sum(x*x for x in cc);cand.append((sc,loads[f][0],counts[f],f))
        f=min(cand)[-1];assign[bif]=f;loads[f]+=feats[bif];counts[f]+=1
    return assign

def canon_sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_rows(local_events=None,local_folds=None):
    if local_events:
        import pickle
        rows=pickle.load(open(local_events,'rb'));folds=json.load(open(local_folds))
    else:
        p=Path('/tmp/voynich_transcriptions_slim.json');urllib.request.urlretrieve(CORPUS_URL,p)
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=CORPUS_SHA:raise RuntimeError(f'corpus sha mismatch {got}')
        rows=build_events(json.load(open(p)));folds=fold_assignment(rows)
    if len(rows)!=34087:raise RuntimeError(f'event count {len(rows)}')
    if canon_sha(rows)!=ROWS_CANON_SHA:raise RuntimeError(f'row canonical sha {canon_sha(rows)}')
    if canon_sha(folds)!=FOLDS_CANON_SHA:raise RuntimeError(f'fold canonical sha {canon_sha(folds)}')
    return rows,folds

def fast_sample_counter(c,rng):
    items=list(c.keys());w=np.fromiter((c[x] for x in items),float);p=w/w.sum();u=rng.random();i=int(np.searchsorted(np.cumsum(p),u,side='right'));return items[min(i,len(items)-1)]

def fit_alpha(cells):
    def neg(loga):
        a=math.exp(loga);ll=0.0
        for c,q in cells:
            n=sum(c.values())
            if n<2:continue
            ll+=gammaln(a)-gammaln(a+n)
            for x,v in c.items():
                qq=max(q.get(x,0),1e-15);ll+=gammaln(a*qq+v)-gammaln(a*qq)
        return -ll
    res=minimize_scalar(neg,bounds=(-2,8),method='bounded',options={'xatol':1e-3});return float(math.exp(res.x))

class FamilyModel:
    def __init__(self,train):
        self.train=train;self.fam_global=collections.Counter();self.fc=[collections.defaultdict(collections.Counter) for _ in range(5)];prev={}
        for r in train:
            f=r['family'];self.fam_global[f]+=1;p=posclass(r);key=(r['page'],r['line']);pf=None if r['pos']==0 else prev.get(key)
            keys=[(r['section'],r['hand'],p,pf),(r['section'],p,pf),(r['section'],pf),(r['section'],p),(r['section'],)]
            for d,k in zip(self.fc,keys):d[k][f]+=1
            prev[key]=f
        N=sum(self.fam_global.values());self.fam_gp={f:n/N for f,n in self.fam_global.items()};self.alpha_f=self._fit_alpha_family()
    def _fit_alpha_family(self):
        secq=collections.defaultdict(collections.Counter);pages=collections.OrderedDict()
        for r in self.train:secq[r['section']][r['family']]+=1;pages.setdefault(r['page'],[]).append(r)
        probs={s:{k:v/sum(c.values()) for k,v in c.items()} for s,c in secq.items()};cells=[]
        for p,rs in pages.items():cells.append((collections.Counter(r['family'] for r in rs),probs[rs[0]['section']]))
        return fit_alpha(cells)
    def _fctx(self,sec,hand,pos,prev):
        keys=[(sec,hand,pos,prev),(sec,pos,prev),(sec,prev),(sec,pos),(sec,)]
        for d,k in zip(self.fc,keys):
            c=d.get(k)
            if c and sum(c.values())>=20:return c
        return self.fam_global
    def sample_fctx(self,sec,hand,pos,prev,rng):
        c=self._fctx(sec,hand,pos,prev);n=sum(c.values())
        return fast_sample_counter(c,rng) if rng.random()<n/(n+BETA) else fast_sample_counter(self.fam_global,rng)

class State:
    def __init__(self):self.page=None;self.line=None;self.famc=collections.Counter();self.prevfam=None
    def boundary(self,r):
        if r['page']!=self.page:self.page=r['page'];self.line=None;self.famc.clear();self.prevfam=None
        if (r['page'],r['line'])!=self.line:self.line=(r['page'],r['line']);self.prevfam=None
    def update_family(self,f):self.famc[f]+=1;self.prevfam=f

def sample_family(model,st,r,rng):
    n=sum(st.famc.values())
    if n and rng.random()<n/(model.alpha_f+n):return fast_sample_counter(st.famc,rng)
    return model.sample_fctx(r['section'],r['hand'],posclass(r),st.prevfam,rng)

def bin_count(n,edges=(60,120,220,400)):
    for i,e in enumerate(edges):
        if n<=e:return i
    return len(edges)
def linekey(r):return (r['page'],r['line'])
def page_metadata(rows):
    byp=collections.OrderedDict();byl=collections.OrderedDict()
    for r in rows:byp.setdefault(r['page'],[]).append(r);byl.setdefault(linekey(r),[]).append(r)
    return byp,byl

def weighted_subset(counter,k,rng):
    items=list(counter);w=np.array([counter[x] for x in items],float);k=min(k,len(items))
    if k<=0:return set()
    idx=rng.choice(len(items),size=k,replace=False,p=w/w.sum());return {items[int(i)] for i in idx}

class H1Bank:
    def __init__(self,train,model):
        self.m=model;self.fam_count=collections.Counter(r['family'] for r in train);self.sec_count=collections.defaultdict(collections.Counter)
        for r in train:self.sec_count[r['section']][r['family']]+=1
        byp,_=page_metadata(train);temp=collections.defaultdict(list)
        for p,rs in byp.items():temp[(rs[0]['section'],bin_count(len(rs)))].append(len(set(r['family'] for r in rs)))
        self.page_k={k:max(3,int(np.median(v))) for k,v in temp.items()}
    def basefam(self,st,r,rng):return self.m.sample_fctx(r['section'],r['hand'],posclass(r),st.prevfam,rng)
    def section_counter(self,sec):return self.sec_count.get(sec) or self.fam_count

class ControlState:
    def __init__(self):self.page=None;self.line=None;self.page_pool=None;self.page_index=0;self.line_index=0

def restrict_baseline(bank,st,r,rng,pool,tries=24):
    if not pool:return bank.basefam(st,r,rng)
    for _ in range(tries):
        f=bank.basefam(st,r,rng)
        if f in pool:return f
    c=collections.Counter({f:bank.fam_count.get(f,1) for f in pool});return fast_sample_counter(c,rng)

def h1_family(bank,cs,st,r,rng,page_rows):
    newpage=r['page']!=cs.page;newline=newpage or linekey(r)!=cs.line
    if newpage:
        cs.page=r['page'];cs.line=None;cs.page_index+=1;cs.line_index=0;cs.page_pool=None
        kk=bank.page_k.get((r['section'],bin_count(len(page_rows))),max(5,int(np.median(list(bank.page_k.values())))))
        cs.page_pool=weighted_subset(bank.section_counter(r['section']),kk,rng)
    if newline:cs.line=linekey(r);cs.line_index+=1
    return bank.basefam(st,r,rng) if rng.random()<.15 else restrict_baseline(bank,st,r,rng,cs.page_pool)

def generate_family(model,bank,skeleton,seed,variant):
    rng=np.random.default_rng(seed);st=State();cs=ControlState();out=[];byp,_=page_metadata(skeleton)
    for r0 in skeleton:
        r=dict(r0);st.boundary(r)
        if variant=='A3_SG1_FULL':f=h1_family(bank,cs,st,r,rng,byp[r['page']])
        else:f=sample_family(model,st,r,rng)
        r['family']=f;r['token']='';out.append(r);st.update_family(f)
    return out

def pages_from_rows(rows):
    by=collections.OrderedDict()
    for r in rows:by.setdefault(r['page'],[]).append(r)
    return by
def jacc(a,b):
    u=a|b;return len(a&b)/len(u) if u else 0.0

class K4Ref:
    def __init__(self,train):
        by=pages_from_rows(train);self.fams=sorted({r['family'] for r in train});self.fi={f:i for i,f in enumerate(self.fams)};X=np.zeros((len(by),len(self.fams)),float)
        for i,rs in enumerate(by.values()):
            for f in set(r['family'] for r in rs):X[i,self.fi[f]]=1.0
        self.nmf=NMF(n_components=4,init='nndsvda',random_state=1,max_iter=400,tol=3e-3).fit(X);W=self.nmf.transform(X);H=self.nmf.components_;shares=W/np.maximum(W.sum(axis=1,keepdims=True),1e-12);states=np.argmax(shares,axis=1);self.state_score=[]
        for s in range(4):
            ix=np.where(states==s)[0];cent=shares[ix].mean(axis=0) if len(ix) else np.eye(4)[s];self.state_score.append(cent@H)
        self.state_score=np.asarray(self.state_score);pres=X.sum(axis=0);self.global_order=np.argsort(pres)[::-1]
    def reconstruction_metrics(self,by):
        if not by:return 0.,0.,0.,0.,{}
        X=np.zeros((len(by),len(self.fams)),float);allsets=[]
        for i,rs in enumerate(by.values()):
            allf=set(r['family'] for r in rs);allsets.append(allf)
            for f in allf:
                if f in self.fi:X[i,self.fi[f]]=1.0
        W=self.nmf.transform(X);shares=W/np.maximum(W.sum(axis=1,keepdims=True),1e-12);hard=np.argmax(shares,axis=1);cont=W@self.nmf.components_;cj=[];dj=[];gj=[];state_by={}
        for i,(p,rs) in enumerate(by.items()):
            obs=allsets[i];k=min(len(obs),len(self.fams))
            if k<=0:cj.append(0.);dj.append(0.);gj.append(0.);state_by[p]=int(hard[i]);continue
            co=set(self.fams[j] for j in np.argsort(cont[i])[::-1][:k]);ds=self.state_score[int(hard[i])];do=set(self.fams[j] for j in np.argsort(ds)[::-1][:k]);go=set(self.fams[j] for j in self.global_order[:k]);cj.append(jacc(obs,co));dj.append(jacc(obs,do));gj.append(jacc(obs,go));state_by[p]=int(hard[i])
        c=float(np.mean(cj));d=float(np.mean(dj));g=float(np.mean(gj));bif=collections.defaultdict(list)
        for p,rs in by.items():bif[rs[0]['bifolium']].append(p)
        ss=[]
        for ps in bif.values():
            for i in range(len(ps)):
                for j in range(i+1,len(ps)):ss.append(float(state_by[ps[i]]==state_by[ps[j]]))
        return c,d,c-g,float(np.mean(ss)) if ss else 0.0,state_by

def repertoire_metrics(rows,k4):
    by=pages_from_rows(rows);type_counts=[];simps=[];slopes=[];fsets={}
    for p,rs in by.items():
        seq=[r['family'] for r in rs];c=collections.Counter(seq);n=len(seq);type_counts.append(len(c));fsets[p]=set(c);simps.append(sum(v*(v-1) for v in c.values())/max(n*(n-1),1));num=np.zeros(5);den=np.zeros(5);seen=set()
        for i,f in enumerate(seq):z=min(4,int(5*i/max(n,1)));den[z]+=1;num[z]+=float(f not in seen);seen.add(f)
        rates=np.divide(num,den,out=np.zeros(5),where=den>0);slopes.append(float(np.polyfit(np.linspace(0,1,5),rates,1)[0]))
    bif=collections.defaultdict(list)
    for p,rs in by.items():bif[rs[0]['bifolium']].append(p)
    bj=[]
    for ps in bif.values():
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):bj.append(jacc(fsets[ps[i]],fsets[ps[j]]))
    sec=collections.defaultdict(list)
    for p,rs in by.items():sec[rs[0]['section']].append(p)
    perpage=[]
    for p,rs in by.items():
        cand=[q for q in sec[rs[0]['section']] if q!=p and by[q][0]['bifolium']!=rs[0]['bifolium']]
        if cand:perpage.append(float(np.mean([jacc(fsets[p],fsets[q]) for q in cand])))
    kc,kd,kg,bs,_=k4.reconstruction_metrics(by)
    vals={'page_family_types_mean':float(np.mean(type_counts)) if type_counts else 0.,'page_family_types_sd':float(np.std(type_counts,ddof=1)) if len(type_counts)>1 else 0.,'page_family_simpson_mean':float(np.mean(simps)) if simps else 0.,'page_family_newtype_slope_mean':float(np.mean(slopes)) if slopes else 0.,'bif_family_jaccard':float(np.mean(bj)) if bj else 0.,'same_section_nonmate_jaccard':float(np.mean(perpage)) if perpage else 0.,'k4_cont_jaccard':kc,'d4_jaccard':kd,'k4_gain_vs_global':kg,'bif_same_state':bs}
    return np.array([vals[m] for m in METRICS],float)

def run_fold(f,rows,folds,nrep=200,bnull=500):
    train=[r for r in rows if folds[r['bifolium']]!=f];test=[r for r in rows if folds[r['bifolium']]==f];model=FamilyModel(train);bank=H1Bank(train,model);k4=K4Ref(train);target=repertoire_metrics(test,k4);tr_bifs=sorted(set(r['bifolium'] for r in train));mtest=len(set(r['bifolium'] for r in test));panel=[]
    for d in range(bnull):
        rng=np.random.default_rng(202609240000+f*10000+d);take=set(rng.choice(tr_bifs,size=min(mtest,len(tr_bifs)),replace=False));rr=[r for r in train if r['bifolium'] in take];panel.append(repertoire_metrics(rr,k4))
    P=np.asarray(panel,float);sd=P.std(axis=0,ddof=1)
    if not np.all(np.isfinite(sd)&(sd>0)):raise RuntimeError('degenerate calibration metric')
    lw=LedoitWolf().fit(P);prec=lw.precision_;pm=P.mean(axis=0);D=P-pm;dist=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',D,prec,D),0));d95=float(np.quantile(dist,.95));out={'fold':f,'ntrain':len(train),'ntest':len(test),'target':dict(zip(METRICS,target.tolist())),'null_sd':dict(zip(METRICS,sd.tolist())),'d95':d95,'variants':{}}
    for vi,variant in enumerate(VARIANTS):
        arr=[]
        for j in range(nrep):
            seed=202609230000+f*100000+vi*1000+j;g=generate_family(model,bank,test,seed,variant);arr.append(repertoire_metrics(g,k4))
        A=np.asarray(arr,float);gm=A.mean(axis=0);delta=gm-target;z=delta/sd;joint=float(math.sqrt(max(float(delta@prec@delta),0)));within2=int(np.sum(np.abs(z)<=2));gt3=int(np.sum(np.abs(z)>3));adequate=bool(joint<=d95 and gt3==0 and within2>=8)
        out['variants'][variant]={'ensemble_mean':dict(zip(METRICS,gm.tolist())),'delta':dict(zip(METRICS,delta.tolist())),'z':dict(zip(METRICS,z.tolist())),'joint':joint,'within2':within2,'gt3':gt3,'adequate':adequate,'replicate_matrix_sha256':hashlib.sha256(A.tobytes()).hexdigest()}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--nrep',type=int,default=200);ap.add_argument('--bnull',type=int,default=500);ap.add_argument('--local-events');ap.add_argument('--local-folds');args=ap.parse_args();rows,folds=load_rows(args.local_events,args.local_folds);out=run_fold(args.fold,rows,folds,args.nrep,args.bnull);print('HF_FOLD_RESULT='+json.dumps(out,separators=(',',':'),sort_keys=True),flush=True)
if __name__=='__main__':main()
