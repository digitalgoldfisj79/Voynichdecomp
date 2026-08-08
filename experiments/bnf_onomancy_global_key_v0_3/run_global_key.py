#!/usr/bin/env python3
import json, math, re, hashlib, urllib.request, html
from collections import Counter
import numpy as np
from unidecode import unidecode

SEED0=20260808
ALPH='abcdefghiklmnopqrstuxyz'
A2I={c:i for i,c in enumerate(ALPH)}
N=len(ALPH); SPACE=N; PK=N+1
LANGS=['latin','italian','german','french','greek','hebrew','arabic','spanish']
TARGETS=['latin','italian','german','hebrew']
LENGTHS=[11264,45056]
REPS=[0,1]
CAP2={c:4 for c in ALPH}
CAP3={'a':7,'b':7,'c':7,'d':10,'e':10,'f':8,'g':10,'h':10,'i':10,'k':9,'l':9,'m':9,'n':9,'o':7,'p':10,'q':9,'r':9,'s':6,'t':9,'u':6,'x':8,'y':10,'z':10}
LM_URLS={
'latin':'https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu',
'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
'french':'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu',
'greek':'https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu',
'hebrew':'https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu',
'arabic':'https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu',
'spanish':'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu'}
SLIM='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
STEPS=6500; RESTARTS=4; POLISH=1200; PERM_NULLS=100

def stable_seed(*parts):
    h=hashlib.sha256('::'.join(map(str,parts)).encode()).digest()
    return (SEED0+int.from_bytes(h[:8],'big'))&0xffffffff

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 BnF-global-key/0.3'})
    with urllib.request.urlopen(req,timeout=90) as r:return r.read().decode('utf-8','replace')

def conllu_sents(txt):
    out=[]; cur=[]
    for line in txt.splitlines():
        if not line:
            if cur:out.append(' '.join(cur));cur=[]
            continue
        if line.startswith('#'):continue
        cc=line.split('\t')
        if len(cc)>=2 and cc[0].isdigit():cur.append(cc[1])
    if cur:out.append(' '.join(cur))
    return out

def norm(s):
    s=unidecode(html.unescape(s)).lower().replace('j','i').replace('v','u').replace('w','u')
    words=[]
    for w in re.findall(r'[a-z]+',s):
        z=''.join(c for c in w if c in A2I)
        if z:words.append(z)
    return ' '.join(words)

def split_train_hold(sents):
    return [s for i,s in enumerate(sents) if i%5!=0],[s for i,s in enumerate(sents) if i%5==0]

def concat_norm(sents):
    return ' '.join(z for z in (norm(s) for s in sents) if z)

def build_lm(sents,max_letters=2500000):
    V=PK**4; counts=np.zeros(V,dtype=np.float64); unig=np.ones(N)*.1; letters=0
    for raw in sents:
        s=norm(raw)
        if not s:continue
        a=np.fromiter((SPACE if c==' ' else A2I[c] for c in s),dtype=np.int16,count=len(s))
        la=a[a<N]; unig+=np.bincount(la,minlength=N); letters+=len(la)
        if len(a)>=4:
            idx=((a[:-3].astype(np.int64)*PK+a[1:-2])*PK+a[2:-1])*PK+a[3:]
            counts+=np.bincount(idx,minlength=V)
        if letters>=max_letters:break
    alpha=.05; logp=np.log((counts+alpha)/(counts.sum()+alpha*V)); unig/=unig.sum()
    return logp,unig,letters

def choose_letter_span(text,L,tag):
    pos=np.flatnonzero(np.fromiter((c!=' ' for c in text),dtype=bool,count=len(text)))
    if len(pos)<L:raise RuntimeError(('short control',tag,L,len(pos)))
    maxstart=len(pos)-L; st=stable_seed('span',tag,L)% (maxstart+1)
    i0=int(pos[st]); i1=int(pos[st+L-1])+1
    return text[i0:i1].strip(),st

def text_to_plain_array(text):
    return np.fromiter((SPACE if c==' ' else A2I[c] for c in text),dtype=np.int16,count=len(text))

def make_control_cipher(text,cap,tag):
    pa=text_to_plain_array(text); letters=pa[pa<N]
    freq=np.bincount(letters,minlength=N); present=np.flatnonzero(freq>0)
    assign=[]
    for a in present:assign.append(int(a))
    while len(assign)<25:
        cnt=np.bincount(assign,minlength=N); candidates=[i for i in present if cnt[i]<cap[ALPH[i]]]
        if not candidates:raise RuntimeError('cannot allocate 25 control symbols')
        # add homophones to letters with largest expected mass per currently allocated symbol
        best=max(candidates,key=lambda i:freq[i]/(cnt[i]+1))
        assign.append(int(best))
    assign=assign[:25]
    rng=np.random.default_rng(stable_seed('opaque',tag)); perm=rng.permutation(25)
    true=np.empty(25,dtype=np.int16)
    byletter={i:[] for i in range(N)}
    for rawid,a in enumerate(assign):
        cid=int(perm[rawid]); true[cid]=a; byletter[a].append(cid)
    # deterministic shuffled cycles guarantee every allocated homophone is exercised when possible
    curs={}; cycles={}
    for a,codes in byletter.items():
        if codes:
            rr=np.random.default_rng(stable_seed('cycle',tag,a)); codes=list(codes); rr.shuffle(codes); cycles[a]=codes;curs[a]=0
    out=np.full(len(pa),-1,dtype=np.int16)
    for i,a in enumerate(pa):
        if a==SPACE:continue
        codes=cycles[int(a)]; out[i]=codes[curs[int(a)]%len(codes)]; curs[int(a)]+=1
    if len(set(int(x) for x in out if x>=0))!=25:raise RuntimeError(('not all 25 symbols observed',tag))
    return out,true,pa

def split_control(seq,pa,frac=.8):
    letterpos=np.flatnonzero(seq>=0); cutletter=int(len(letterpos)*frac); cut=int(letterpos[cutletter])
    return seq[:cut],pa[:cut],seq[cut:],pa[cut:]

class QuadAgg:
    def __init__(self,seq,nsym):
        self.nsym=nsym; self.B=nsym+1; sp=nsym
        x=np.where(seq<0,sp,seq).astype(np.int64)
        if len(x)<4:raise RuntimeError('too short')
        ids=((x[:-3]*self.B+x[1:-2])*self.B+x[2:-1])*self.B+x[3:]
        uid,cnt=np.unique(ids,return_counts=True); self.counts=cnt.astype(np.float64); self.total=float(cnt.sum())
        q=np.empty((len(uid),4),dtype=np.int16); y=uid.copy()
        for k in (3,2,1,0):q[:,k]=(y%self.B).astype(np.int16);y//=self.B
        self.q=q
        self.aff=[]
        for s in range(nsym):self.aff.append(np.flatnonzero(np.any(q==s,axis=1)))
    def contrib(self,mapping,logp,rows=None):
        q=self.q if rows is None else self.q[rows]; cnt=self.counts if rows is None else self.counts[rows]
        ext=np.empty(self.nsym+1,dtype=np.int16);ext[:self.nsym]=mapping;ext[self.nsym]=SPACE
        p=ext[q]
        idx=((p[:,0].astype(np.int64)*PK+p[:,1])*PK+p[:,2])*PK+p[:,3]
        return logp[idx]*cnt
    def score(self,mapping,logp):return float(self.contrib(mapping,logp).sum()/self.total)

def init_map(nsym,cap,unig,seq,rng):
    caps=np.array([cap[c] for c in ALPH],dtype=int); m=np.empty(nsym,dtype=np.int16); used=np.zeros(N,dtype=int)
    f=np.bincount(seq[seq>=0],minlength=nsym); order=np.argsort(-f)
    for s in order:
        avail=np.flatnonzero(used<caps); w=unig[avail]**.65/(1+used[avail]*.7);w/=w.sum();a=int(rng.choice(avail,p=w));m[s]=a;used[a]+=1
    return m

def optimize(seq,cap,unig,logp,tag):
    nsym=int(seq[seq>=0].max())+1; agg=QuadAgg(seq,nsym); caps=np.array([cap[c] for c in ALPH],dtype=int)
    best_score=-1e99;best_map=None
    for rr in range(RESTARTS):
        rng=np.random.default_rng(stable_seed('opt',tag,rr)); m=init_map(nsym,cap,unig,seq,rng); cnt=np.bincount(m,minlength=N)
        cv=agg.contrib(m,logp); cur=float(cv.sum()/agg.total)
        if cur>best_score:best_score,best_map=cur,m.copy()
        for step in range(STEPS+POLISH):
            anneal=step<STEPS; frac=step/max(1,STEPS-1); T=.035*(1-frac)+.00015 if anneal else 0.0
            s=int(rng.integers(nsym)); old=int(m[s]); new=int(rng.integers(N))
            if new==old:continue
            changed=[s]; swap=None
            if cnt[new]>=caps[new]:
                cand=np.flatnonzero(m==new)
                if not len(cand):continue
                swap=int(rng.choice(cand));changed.append(swap)
            rows=agg.aff[s] if swap is None else np.union1d(agg.aff[s],agg.aff[swap])
            oldsum=float(cv[rows].sum()); old2=None
            m[s]=new
            if swap is not None:old2=int(m[swap]);m[swap]=old
            nv=agg.contrib(m,logp,rows); delta=(float(nv.sum())-oldsum)/agg.total
            accept=delta>=0 or (anneal and rng.random()<math.exp(max(-50,min(0,delta/max(T,1e-12)))))
            if accept:
                cv[rows]=nv;cur+=delta;cnt=np.bincount(m,minlength=N)
                if cur>best_score:best_score,best_map=cur,m.copy()
            else:
                m[s]=old
                if swap is not None:m[swap]=old2
    return best_score,best_map

def score_seq(seq,mapping,logp):
    return QuadAgg(seq,len(mapping)).score(mapping,logp)

def char_accuracy(seq,pa,mapping):
    mask=seq>=0; return float(np.mean(mapping[seq[mask]]==pa[mask]))

def perm_z(seq,mapping,logp,tag,n=PERM_NULLS):
    agg=QuadAgg(seq,len(mapping)); obs=agg.score(mapping,logp); vals=[]
    for j in range(n):
        rr=np.random.default_rng(stable_seed('perm',tag,j)); mm=mapping.copy();rr.shuffle(mm);vals.append(agg.score(mm,logp))
    mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));z=(obs-mu)/sd if sd>1e-12 else 0.0
    return obs,mu,sd,z

def extract_page(data,folio,tid):
    ls=data['pages'][folio]
    def key(k):
        try:return int(k)
        except:return 999999
    text=' '.join(line.get('t',{}).get(tid,'') for _,line in sorted(ls.items(),key=lambda kv:key(kv[0]))).strip().lower()
    # transliteration labels are opaque cipher symbols; preserve only non-whitespace characters and spaces
    return ' '.join(text.split())

def vms_pages(data,tid):
    out=[]
    for f in sorted(data['pages']):
        t=extract_page(data,f,tid)
        if t:out.append((f,t))
    return out

def folio_holdout(f):return int.from_bytes(hashlib.sha256(f.encode()).digest()[:8],'big')%5==0

def pages_to_cipher(pages,symbols=None):
    if symbols is None:symbols=sorted(set(c for _,t in pages for c in t if not c.isspace()))
    s2i={s:i for i,s in enumerate(symbols)}; seq=[]
    for _,t in pages:
        for c in t:
            if c.isspace():seq.append(-1)
            elif c in s2i:seq.append(s2i[c])
        seq.append(-1)
    return np.asarray(seq,dtype=np.int16),symbols

def representative_pages(pages,capletters=45056):
    total=sum(sum(not c.isspace() for c in t) for _,t in pages)
    if total<=capletters:return pages
    # evenly cover the folio list, then stop once cap reached
    n=len(pages); want=max(1,int(round(n*capletters/total))); idx=np.linspace(0,n-1,want,dtype=int)
    sel=[pages[int(i)] for i in sorted(set(idx))]
    # if slightly short, add hashed remaining pages
    got=sum(sum(not c.isspace() for c in t) for _,t in sel)
    if got<capletters:
        chosen={f for f,_ in sel}; rem=sorted((p for p in pages if p[0] not in chosen),key=lambda p:stable_seed('samplepage',p[0]))
        for p in rem:
            sel.append(p);got+=sum(not c.isspace() for c in p[1])
            if got>=capletters:break
    return sorted(sel)

def transfer_score(pages,labelmap,logp,tag):
    # Unknown glyph labels become breaks; score only 4-grams containing known labels/spaces.
    vals=[];covered=0;total=0
    for _,t in pages:
        arr=[]
        for c in t:
            if c.isspace():arr.append(SPACE)
            else:
                total+=1
                if c in labelmap:arr.append(labelmap[c]);covered+=1
                else:arr.append(-2)
        a=np.asarray(arr,dtype=np.int16)
        if len(a)<4:continue
        ok=(a[:-3]>=0)&(a[1:-2]>=0)&(a[2:-1]>=0)&(a[3:]>=0)
        if np.any(ok):
            p0=a[:-3][ok];p1=a[1:-2][ok];p2=a[2:-1][ok];p3=a[3:][ok]
            idx=((p0.astype(np.int64)*PK+p1)*PK+p2)*PK+p3;vals.append(logp[idx])
    obs=float(np.concatenate(vals).mean()) if vals else -1e99
    # mapping-permutation null on literal labels; compute by redecoding 100 times
    keys=list(labelmap); ass=np.array([labelmap[k] for k in keys],dtype=np.int16); null=[]
    for j in range(PERM_NULLS):
        rr=np.random.default_rng(stable_seed('transferperm',tag,j));bb=ass.copy();rr.shuffle(bb);lm={k:int(v) for k,v in zip(keys,bb)}
        vv=[]
        for _,t in pages:
            a=np.fromiter((SPACE if c.isspace() else lm.get(c,-2) for c in t),dtype=np.int16,count=len(t));
            if len(a)<4:continue
            ok=(a[:-3]>=0)&(a[1:-2]>=0)&(a[2:-1]>=0)&(a[3:]>=0)
            if np.any(ok):
                p0=a[:-3][ok];p1=a[1:-2][ok];p2=a[2:-1][ok];p3=a[3:][ok];idx=((p0.astype(np.int64)*PK+p1)*PK+p2)*PK+p3;vv.append(logp[idx])
        null.append(float(np.concatenate(vv).mean()) if vv else -1e99)
    mu=float(np.mean(null));sd=float(np.std(null,ddof=1));z=(obs-mu)/sd if sd>1e-12 else 0
    return {'score':obs,'null_mean':mu,'null_sd':sd,'z':z,'coverage':covered/max(1,total)}

def main():
    # corpora + LMs
    lms={};unigs={};holds={};meta={}
    for lang in LANGS:
        ss=conllu_sents(fetch(LM_URLS[lang]));tr,ho=split_train_hold(ss) if lang in TARGETS else (ss,[])
        lm,ug,nlet=build_lm(tr);lms[lang]=lm;unigs[lang]=ug
        if lang in TARGETS:holds[lang]=concat_norm(ho)
        meta[lang]={'sentences':len(ss),'train_sentences':len(tr),'lm_letters':nlet,'hold_letters':sum(c!=' ' for c in holds.get(lang,''))}
        print('LM',lang,meta[lang],flush=True)

    # Binding P0, T2. T3 not needed to decide permission to enter VMS.
    controls=[]
    for L in LENGTHS:
        for lang in TARGETS:
            for rep in REPS:
                plain,st=choose_letter_span(holds[lang],L,(lang,rep,L));seq,true,pa=make_control_cipher(plain,CAP2,('P0',lang,L,rep));trseq,trpa,hseq,hpa=split_control(seq,pa)
                rr=[]
                for cand in LANGS:
                    sc,mp=optimize(trseq,CAP2,unigs[cand],lms[cand],('P0',lang,L,rep,cand))
                    obs,mu,sd,z=perm_z(hseq,mp,lms[cand],('P0',lang,L,rep,cand));acc=char_accuracy(hseq,hpa,mp)
                    rr.append({'cand':cand,'train_score':sc,'hold_score':obs,'z':z,'acc':acc})
                rr.sort(key=lambda x:x['z'],reverse=True); target=next(x for x in rr if x['cand']==lang)
                row={'L':L,'lang':lang,'rep':rep,'top':rr[0]['cand'],'top_z':rr[0]['z'],'target_rank':1+next(i for i,x in enumerate(rr) if x['cand']==lang),'target_z':target['z'],'target_acc':target['acc'],'span_start':st,'ranking':[(x['cand'],x['z']) for x in rr]}
                controls.append(row);print('CONTROL',json.dumps(row,separators=(',',':')),flush=True)
    bind=[r for r in controls if r['L']==45056];correct=sum(r['top']==r['lang'] for r in bind);medacc=float(np.median([r['target_acc'] for r in bind]));minacc=float(min(r['target_acc'] for r in bind));medz=float(np.median([r['target_z'] for r in bind]))
    p0={'correct':correct,'n':len(bind),'median_acc':medacc,'min_acc':minacc,'median_z':medz,'P01':correct==8,'P02':medacc>=.90,'P03':minacc>=.75,'P04':medz>=10}
    p0['pass']=all(p0[k] for k in ['P01','P02','P03','P04']);print('P0',json.dumps(p0,separators=(',',':')),flush=True)
    out={'protocol':'v0.3','lm_meta':meta,'controls':controls,'P0':p0}
    if not p0['pass']:
        out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return

    # Voynich only after P0 PASS
    data=json.loads(fetch(SLIM));zp=vms_pages(data,'ZLZI');train=[p for p in zp if not folio_holdout(p[0])];hold=[p for p in zp if folio_holdout(p[0])];sample=representative_pages(train,45056)
    trseq,syms=pages_to_cipher(sample);hseq,_=pages_to_cipher(hold,syms)
    print('VMS_CENSUS',json.dumps({'train_pages':len(train),'hold_pages':len(hold),'sample_pages':len(sample),'symbols':syms,'train_sample_letters':int(np.sum(trseq>=0)),'hold_letters':int(np.sum(hseq>=0))}),flush=True)
    vres=[]
    for model,cap in [('T2',CAP2),('T3',CAP3)]:
        for lang in LANGS:
            sc,mp=optimize(trseq,cap,unigs[lang],lms[lang],('VMS','ZLZI',model,lang));obs,mu,sd,z=perm_z(hseq,mp,lms[lang],('VMS','ZLZI',model,lang))
            counts=np.bincount(mp,minlength=N); labelmap={s:ALPH[int(mp[i])] for i,s in enumerate(syms)}
            row={'model':model,'lang':lang,'train_score':sc,'hold_score':obs,'z':z,'max_homophones':int(counts.max()),'mapping':labelmap}
            vres.append(row);print('VMS',json.dumps(row,separators=(',',':')),flush=True)
    t2=sorted([r for r in vres if r['model']=='T2'],key=lambda r:r['z'],reverse=True); top=t2[0];second=t2[1]
    transfers={}
    if top['z']>=10 and top['z']-second['z']>=5:
        lm=lms[top['lang']];mapping_plain={k:A2I[v] for k,v in top['mapping'].items()}
        for tid in ['TTLI','VDRB']:
            pp=[p for p in vms_pages(data,tid) if folio_holdout(p[0])]
            transfers[tid]=transfer_score(pp,mapping_plain,lm,('VMS-transfer',tid,top['lang']))
            print('TRANSFER',tid,json.dumps(transfers[tid],separators=(',',':')),flush=True)
    gate={'top_language':top['lang'],'top_z':top['z'],'second_language':second['lang'],'second_z':second['z'],'margin':top['z']-second['z'],'T2_capacity_ok':top['max_homophones']<=4,'transfers':transfers}
    gate['signal']=bool(top['z']>=10 and gate['margin']>=5 and gate['T2_capacity_ok'] and all(transfers.get(t,{}).get('z',-1e99)>=5 for t in ['TTLI','VDRB']))
    out.update({'vms':vres,'gate':gate,'verdict':'GLOBAL_KEY_SIGNAL' if gate['signal'] else 'GLOBAL_FIXED_KEY REJECTED'})
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
