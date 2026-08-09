#!/usr/bin/env python3
import os,re,json,math,hashlib,urllib.request,subprocess,tempfile
from collections import Counter,defaultdict
import numpy as np
from unidecode import unidecode

# Reuse only the frozen BnF tables, normalization and LM construction from v0.7.
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ccea68e5eef0b551cff7cb2703c20c9868e294c/experiments/bnf_free_switch_m19_v0_7/run_m19.py'
ns={'__name__':'m19base'}
exec(compile(urllib.request.urlopen(BASE,timeout=90).read().decode(),'run_m19.py','exec'),ns)

SEEDNS='M19STAv17'
LANGS=list(ns['LANGS'])
QUAL=['latin','italian','german','french','arabic','spanish']
VALUES=ns['VALUES']; NV=ns['NV']; EMIT=ns['EMIT']; LETTER_VALS=ns['LETTER_VALS']; A2I=ns['A2I']; V2I=ns['V2I']
TRAIN=45000; HOLD=39000
LM_TRAIN_RES={3,4,8,9}

SRC={
 'RF':('https://voynich.nu/data/sta/RF1b.txt','81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'),
 'IT':('https://voynich.nu/data/sta/IT2a.txt','215f2d05690828c00bd4ae00d6201df31050adcd81601343b142ae91b9dfeee4'),
 'ZL':('https://voynich.nu/data/sta/ZL3b.txt','8438ba1c45f47fe1d06b5262cbcdf60ce69158a0edbd4dd802612896f3217e2a'),
 'GC':('https://voynich.nu/data/sta/GC2a_1.txt','0c0d1eea4b5ab87f8a65fb7f4346864cd90758ad993812b4f2122b3899d4ac88'),
 'bitrans.c':('https://www.voynich.nu/software/bitrans/bitrans.c','3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a'),
 'STA-aaa.bit':('https://www.voynich.nu/software/bitrans/STA-aaa.bit','622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64'),
}
HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36','Referer':'https://voynich.nu/transcr.html','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7','Accept-Language':'en-GB,en;q=0.9','Cache-Control':'no-cache','Pragma':'no-cache'}
CONTROL_URLS={
 'latin':['https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-test.conllu'],
 'italian':['https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-test.conllu'],
 'german':['https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-test.conllu'],
 'french':['https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-test.conllu'],
 'arabic':['https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-test.conllu'],
 'spanish':['https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-test.conllu'],
}
STA_RE=re.compile(r'[A-Z][0-9a-z]')

def seed(*parts):
    h=hashlib.sha256(('::'.join([SEEDNS]+list(map(str,parts)))).encode()).digest()
    return int.from_bytes(h[:8],'big') & 0xffffffff

def get_bytes(url,tag):
    sep='&' if '?' in url else '?'; req=urllib.request.Request(url+sep+'m19v17='+tag,headers=HEADERS)
    with urllib.request.urlopen(req,timeout=120) as r:return r.read()

def fetch_text(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 M19STA-v17'})
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode('utf-8','replace')

def acquire_sources(td):
    out={}; meta={}
    for key,(url,want) in SRC.items():
        b=get_bytes(url,key.replace('.','_')); got=hashlib.sha256(b).hexdigest()
        if got!=want:raise RuntimeError(('source hash mismatch',key,got,want))
        fn=os.path.join(td,key if '.' in key else key+'.txt');open(fn,'wb').write(b);out[key]=fn;meta[key]={'bytes':len(b),'sha256':got,'url':url}
    exe=os.path.join(td,'bitrans');subprocess.run(['gcc','-O2','-o',exe,out['bitrans.c']],check=True)
    for key in ['RF','IT','ZL','GC']:
        dst=os.path.join(td,key+'.aaa.txt')
        p=subprocess.run([exe,'-1','-m2','-f',out['STA-aaa.bit'],out[key],dst],cwd=td,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if p.returncode!=0:raise RuntimeError(('bitrans failed',key,p.returncode,p.stderr[-2000:]))
        out[key+'_aaa']=dst;meta[key+'_aaa']={'bytes':os.path.getsize(dst),'sha256':hashlib.sha256(open(dst,'rb').read()).hexdigest()}
    print('SOURCES',json.dumps(meta,separators=(',',':')),flush=True)
    return out,meta

def locus_bodies(txt,short=False):
    for ln in txt.splitlines():
        if ln.startswith('#') or not ln.startswith('<'):continue
        m=re.match(r'<([^>]+)>\s+(.*)$',ln)
        if not m:continue
        loc,body=m.groups()
        if body.startswith('<!'):continue
        page=loc.split('.')[0]
        body=re.sub(r'\[[^\]]*\]','.',body)
        body=re.sub(r'<[^>]*>','.',body)
        body=body.replace(',', '.' if short else '')
        yield page,body

def parse_sta(txt,mode,short=False):
    pages=defaultdict(list)
    for page,body in locus_bodies(txt,short):
        for chunk in body.split('.'):
            codes=STA_RE.findall(chunk)
            if not codes:continue
            if mode=='family': toks=[c[0] for c in codes]
            else:toks=codes
            pages[page].append(toks)
    return dict(pages)

def aaa_units(s):
    out=[];i=0
    while i<len(s):
        if i+1<len(s) and s[i].islower() and s[i+1].isdigit():
            u=s[i:i+2];i+=2
            while i<len(s) and s[i]==':' and i+2<len(s) and s[i+1].islower() and s[i+2].isdigit():
                u+=':'+s[i+1:i+3];i+=3
            out.append(u)
        else:i+=1
    return out

def parse_aaa(txt,short=False):
    pages=defaultdict(list)
    for page,body in locus_bodies(txt,short):
        for chunk in body.split('.'):
            u=aaa_units(chunk)
            if u:pages[page].append(u)
    return dict(pages)

def is_unknown(tok,rep):return (tok=='Z' if rep=='family' else tok=='Z1' if rep=='sta' else tok.startswith('z'))

def split_pages(pages):
    fol=sorted(pages,key=lambda f:hashlib.sha256((SEEDNS+'split::'+f).encode()).digest())
    n=len(fol);nt=round(.60*n);nh=round(.20*n)
    return fol[:nt],fol[nt:nt+nh],fol[nt+nh:],fol

def count_tokens(pages,folios,rep):
    C=Counter()
    for f in folios:
        for w in pages.get(f,[]):
            for t in w:
                if not is_unknown(t,rep):C[t]+=1
    return C

def choose_vocab(C,rep):
    if rep=='family':
        v=sorted(C);k=len(v)
        if not (19<=k<=38):raise RuntimeError(('family K inadmissible',k))
        return v,1.0
    total=sum(C.values());ordered=sorted(C,key=lambda x:(-C[x],x));cum=0
    for k,t in enumerate(ordered,1):
        cum+=C[t]
        if k>=19 and cum/max(1,total)>=.995:
            if k>38:break
            return ordered[:k],cum/max(1,total)
    raise RuntimeError(('no <=38 vocabulary reaches .995',rep,len(C),total))

def project(pages,folios,rep,vocab):
    V=set(vocab);out=[];recognized=retained=0;total_words=kept_words=0;unknown=0
    byfolio={}
    for f in folios:
        fw=[]
        for w in pages.get(f,[]):
            total_words+=1
            known=[not is_unknown(t,rep) for t in w];recognized+=sum(known);unknown+=len(w)-sum(known)
            if all(known) and all(t in V for t in w):
                fw.append(w);out.append(w);retained+=len(w);kept_words+=1
        byfolio[f]=fw
    return out,byfolio,{'recognized_chars':recognized,'retained_chars':retained,'coverage':retained/max(1,recognized),'unknown_units':unknown,'words':total_words,'retained_words':kept_words,'word_coverage':kept_words/max(1,total_words)}

def stats(words,symbols):
    s2i={s:i for i,s in enumerate(symbols)};K=len(symbols);B=np.zeros((K,K),np.int64);st=np.zeros(K,np.int64);en=np.zeros(K,np.int64);freq=np.zeros(K,np.int64)
    for w in words:
        q=[s2i[x] for x in w];
        if not q:continue
        st[q[0]]+=1;en[q[-1]]+=1
        for x in q:freq[x]+=1
        for x,y in zip(q,q[1:]):B[x,y]+=1
    denom=max(1,int(B.sum()+st.sum()+en.sum()+freq.sum()))
    return {'B':B,'st':st,'en':en,'freq':freq,'denom':denom,'symbols':symbols}

def valid_map(m,K):
    c=np.bincount(m,minlength=NV)
    return len(m)==K and np.all(c>=1) and np.all(c<=2) and int(np.sum(c==2))==K-NV

def init_map(K,rng):
    d=K-NV;dup=[] if d==0 else list(map(int,rng.choice(NV,d,replace=False)));a=np.array(list(range(NV))+dup,np.int16);rng.shuffle(a);return a

def score_num(S,m,comp):
    lt,ls,le=comp;B=S['B'];c=np.bincount(m,minlength=NV)
    z=float(np.sum(B*lt[np.ix_(m,m)])+np.dot(S['st'],ls[m])+np.dot(S['en'],le[m])-np.dot(S['freq'],np.log(c[m])))
    return z/S['denom']

def delta_score(S,m,x,changed,comp):
    lt,ls,le=comp;B=S['B'];K=len(m);C=np.array(sorted(set(changed)),dtype=int);mask=np.ones(K,bool);mask[C]=False;O=np.flatnonzero(mask)
    old=float(np.sum(B[np.ix_(C,np.arange(K))]*lt[m[C][:,None],m[None,:]]));new=float(np.sum(B[np.ix_(C,np.arange(K))]*lt[x[C][:,None],x[None,:]]))
    if len(O):
        old+=float(np.sum(B[np.ix_(O,C)]*lt[m[O][:,None],m[C][None,:]]));new+=float(np.sum(B[np.ix_(O,C)]*lt[x[O][:,None],x[C][None,:]]))
    old+=float(np.dot(S['st'][C],ls[m[C]])+np.dot(S['en'][C],le[m[C]]));new+=float(np.dot(S['st'][C],ls[x[C]])+np.dot(S['en'][C],le[x[C]]))
    co=np.bincount(m,minlength=NV);cn=np.bincount(x,minlength=NV);old-=float(np.dot(S['freq'],np.log(co[m])));new-=float(np.dot(S['freq'],np.log(cn[x])))
    return (new-old)/S['denom']

def proposal(m,rng):
    K=len(m);cnt=np.bincount(m,minlength=NV)
    if rng.random()<.75 or not (np.any(cnt==2) and np.any(cnt==1)):
        for _ in range(20):
            a,b=rng.choice(K,2,replace=False)
            if m[a]!=m[b]:
                x=m.copy();x[a],x[b]=x[b],x[a];return x,[int(a),int(b)]
    sv=int(rng.choice(np.flatnonzero(cnt==2)));dv=int(rng.choice(np.flatnonzero(cnt==1)));i=int(rng.choice(np.flatnonzero(m==sv)));x=m.copy();x[i]=dv;return x,[i]

def optimize(S,comp,tag,K):
    steps=26000 if K<=26 else 40000;restarts=6 if K<=26 else 8;best=(-1e100,None)
    for rr in range(restarts):
        rng=np.random.default_rng(seed('opt',tag,rr));m=init_map(K,rng);s=score_num(S,m,comp)
        ds=[]
        for _ in range(50):
            x,ch=proposal(m,rng);ds.append(abs(delta_score(S,m,x,ch,comp)))
        t0=max(1e-6,float(np.median(ds))*4)
        local_best=(s,m.copy())
        for k in range(steps):
            frac=k/max(1,steps-1);temp=max(1e-7,t0*(0.01**frac));x,ch=proposal(m,rng);d=delta_score(S,m,x,ch,comp)
            if d>=0 or rng.random()<math.exp(max(-50,d/temp)):
                m=x;s+=d
                if s>local_best[0]:local_best=(s,m.copy())
        m=local_best[1].copy();s=score_num(S,m,comp)
        for _ in range(8):
            bd=1e-12;bx=None;bch=None;cnt=np.bincount(m,minlength=NV)
            for a in range(K):
                for bb in range(a+1,K):
                    if m[a]==m[bb]:continue
                    x=m.copy();x[a],x[bb]=x[bb],x[a];d=delta_score(S,m,x,[a,bb],comp)
                    if d>bd:bd=d;bx=x;bch=[a,bb]
            if np.any(cnt==2) and np.any(cnt==1):
                for sv in np.flatnonzero(cnt==2):
                    for dv in np.flatnonzero(cnt==1):
                        for i in np.flatnonzero(m==sv):
                            x=m.copy();x[i]=dv;d=delta_score(S,m,x,[int(i)],comp)
                            if d>bd:bd=d;bx=x;bch=[int(i)]
            if bx is None:break
            m=bx;s+=bd
        s=score_num(S,m,comp)
        if s>best[0]:best=(s,m.copy())
    assert valid_map(best[1],K)
    return best

def agreement(freq,a,b):return float(np.dot(freq,a==b)/max(1,freq.sum()))
def map_acc(freq,a,true):return float(np.dot(freq,a==true)/max(1,freq.sum()))

def forward_word(obs,lm):
    if not obs:return 0.0,0
    T=lm['T'];a=lm['st']*EMIT[:,obs[0]];z=float(a.sum())
    if z<=0:return -1e100,0
    ll=math.log(z);a/=z
    for v in obs[1:]:
        a=(a@T)*EMIT[:,v];z=float(a.sum())
        if z<=0:return -1e100,0
        ll+=math.log(z);a/=z
    z=float(np.dot(a,lm['en']))
    if z>0:ll+=math.log(z)
    return ll,len(obs)

def forward(words,m,symbols,lm):
    s2i={s:i for i,s in enumerate(symbols)};ll=0.0;n=0
    for w in words:
        obs=[int(m[s2i[x]]) for x in w];q,k=forward_word(obs,lm);ll+=q;n+=k
    return ll/max(1,n),n

def build_lms():
    lms={};meta={}
    for la,u in ns['LM_URLS'].items():
        ss=ns['conllu'](fetch_text(u));tr=[s for i,s in enumerate(ss) if i%10 in LM_TRAIN_RES];lms[la]=ns['build_lm'](tr);meta[la]={'sentences':len(tr),'letters':lms[la]['letters']};print('LM',la,meta[la],flush=True)
    return lms,meta

def control_pools():
    out={};meta={}
    for la,urls in CONTROL_URLS.items():
        ss=[]
        for u in urls:ss.extend(ns['conllu'](fetch_text(u)))
        p=ns['pool_text'](ss);out[la]=p;meta[la]={'sentences':len(ss),'letters':sum(c!=' ' for c in p)}
        if meta[la]['letters']<TRAIN+HOLD:raise RuntimeError(('control pool short',la,meta[la]))
        print('CONTROL_POOL',la,meta[la],flush=True)
    return out,meta

def split_token_words(words,n):
    a=[];b=[];k=0
    for w in words:
        if k>=n:b.append(w);continue
        if k+len(w)<=n:a.append(w);k+=len(w);continue
        cut=n-k
        if cut:a.append(w[:cut])
        if cut<len(w):b.append(w[cut:])
        k=n
    return a,b

def gen_control(pool,la,K):
    span=ns['choose_span'](pool,TRAIN+HOLD,('v17qual',la,K));plain=span.split()
    for attempt in range(1000):
        rv=np.random.default_rng(seed('control-values',la,K,attempt));valwords=[];traincnt=Counter();n=0
        for w in plain:
            vv=[]
            for c in w:
                v=V2I[int(rv.choice(LETTER_VALS[A2I[c]]))];vv.append(v)
                if n<TRAIN:traincnt[v]+=1
                n+=1
            valwords.append(vv)
        if n<TRAIN+HOLD:continue
        d=K-NV;dup=[v for v,_ in sorted(traincnt.items(),key=lambda kv:(-kv[1],kv[0]))[:d]]
        if len(dup)!=d:continue
        raw={v:[v] for v in range(NV)}
        for j,v in enumerate(dup):raw[v].append(NV+j)
        ro=np.random.default_rng(seed('control-opaque',la,K,attempt));perm=np.arange(K);ro.shuffle(perm);true=np.full(K,-1,np.int16)
        for v,ff in raw.items():
            for q in ff:true[int(perm[q])]=v
        rs=np.random.default_rng(seed('control-surface',la,K,attempt));surf=[]
        for vw in valwords:surf.append([int(perm[int(rs.choice(raw[v]))]) for v in vw])
        tr,ho=split_token_words(surf,TRAIN);used={q for w in tr for q in w}
        if len(used)==K:
            syms=[f'S{i:02d}' for i in range(K)];tr=[[syms[x] for x in w] for w in tr];ho=[[syms[x] for x in w] for w in ho];assert valid_map(true,K);return tr,ho,syms,true,attempt
    raise RuntimeError(('control generation exhausted',la,K))

def qualify(K,lms,comps,pools):
    rows=[]
    for la in QUAL:
        tr,ho,syms,true,attempt=gen_control(pools[la],la,K);S=stats(tr,syms);H=stats(ho,syms)
        s1,m1=optimize(S,comps[la],('qual',K,la,1),K);s2,m2=optimize(S,comps[la],('qual',K,la,2),K);m=m1 if s1>=s2 else m2
        agr=agreement(S['freq'],m1,m2);acc=map_acc(H['freq'],m,true);rank=[]
        for cand in LANGS:
            fw,n=forward(ho,m,syms,lms[cand]);rank.append((cand,fw))
        rank.sort(key=lambda x:x[1],reverse=True);row={'lang':la,'K':K,'attempt':attempt,'top':rank[0][0],'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'margin':rank[0][1]-rank[1][1],'mapping_acc':acc,'fit_agreement':agr,'ranking':rank};rows.append(row);print('QUAL',json.dumps(row,separators=(',',':')),flush=True)
    gate={'K':K,'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'median_acc':float(np.median([r['mapping_acc'] for r in rows])),'min_acc':min(r['mapping_acc'] for r in rows),'min_agreement':min(r['fit_agreement'] for r in rows)}
    gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90
    print('QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);return rows,gate

def rank_fixed(words,fits,symbols,lms):
    rows=[]
    for la,m in fits.items():
        sc,n=forward(words,m,symbols,lms[la]);rows.append((la,sc,n))
    rows.sort(key=lambda x:x[1],reverse=True);return rows

def fit_rep(name,Tw,Hw,symbols,lms,comps,Hcov):
    K=len(symbols);S=stats(Tw,symbols);fits={};agreements={};train_scores={}
    for la in LANGS:
        s1,m1=optimize(S,comps[la],('RF',name,la,1),K);s2,m2=optimize(S,comps[la],('RF',name,la,2),K);fits[la]=m1 if s1>=s2 else m2;agreements[la]=agreement(S['freq'],m1,m2);train_scores[la]=max(s1,s2);print('FIT',name,la,'agreement',agreements[la],'score',train_scores[la],flush=True)
    rank=rank_fixed(Hw,fits,symbols,lms);top=rank[0][0];margin=rank[0][1]-rank[1][1];gate=Hcov>=.97 and margin>=.05 and agreements[top]>=.90
    row={'rep':name,'K':K,'top':top,'margin':margin,'top_agreement':agreements[top],'coverage':Hcov,'gate':gate,'ranking':rank,'agreements':agreements,'candidate_map':{symbols[i]:VALUES[int(fits[top][i])] for i in range(K)}};print('H17',json.dumps(row,separators=(',',':')),flush=True);return row,fits,agreements

def bucket_id(f):return hashlib.sha256((SEEDNS+'bucket::'+f).encode()).digest()[0]%4

def evaluate_source(label,rep,pages,folios,vocab,fits,lms,short=False):
    words,byf,cov=project(pages,folios,rep,vocab);rank=rank_fixed(words,fits,vocab,lms);out={'source':label,'rep':rep,'short':short,'coverage':cov,'ranking':rank,'top':rank[0][0],'margin':rank[0][1]-rank[1][1]};return out,byf

def main():
    td=tempfile.mkdtemp(prefix='m19sta17_');paths,source_meta=acquire_sources(td)
    raw={};raw_short={}
    for k in ['RF','IT','ZL','GC']:
        st=open(paths[k],encoding='utf-8').read();aa=open(paths[k+'_aaa'],encoding='utf-8').read()
        raw[(k,'family')]=parse_sta(st,'family',False);raw_short[(k,'family')]=parse_sta(st,'family',True)
        raw[(k,'sta')]=parse_sta(st,'sta',False);raw_short[(k,'sta')]=parse_sta(st,'sta',True)
        raw[(k,'aaa')]=parse_aaa(aa,False);raw_short[(k,'aaa')]=parse_aaa(aa,True)
    T,H,C,allf=split_pages(raw[('RF','sta')]);print('SPLIT',json.dumps({'n':len(allf),'T':len(T),'H':len(H),'C':len(C),'Tfolios':T,'Hfolios':H,'Cfolios':C},separators=(',',':')),flush=True)
    reps={};
    for rep in ['family','sta','aaa']:
        Cnt=count_tokens(raw[('RF',rep)],T,rep);v,cov=choose_vocab(Cnt,rep);Tw,Tby,Tcov=project(raw[('RF',rep)],T,rep,v);Hw,Hby,Hcov=project(raw[('RF',rep)],H,rep,v);Cw,Cby,Ccov=project(raw[('RF',rep)],C,rep,v);reps[rep]={'vocab':v,'K':len(v),'Twords':Tw,'Hwords':Hw,'Cwords':Cw,'Tcov':Tcov,'Hcov':Hcov,'Ccov':Ccov};print('VOCAB',rep,json.dumps({'K':len(v),'training_selection_coverage':cov,'vocab':v,'T':Tcov,'H':Hcov,'C':Ccov},separators=(',',':')),flush=True)
    lms,lmmeta=build_lms();comps={la:ns['induced'](lms[la]) for la in LANGS};pools,poolmeta=control_pools();qres={};qg={}
    for K in sorted({r['K'] for r in reps.values()}):qres[K],qg[K]=qualify(K,lms,comps,pools)
    out={'protocol':'v1.7','sources':source_meta,'split':{'T':T,'H':H,'C':C},'representation_meta':{r:{k:v for k,v in d.items() if k not in ['Twords','Hwords','Cwords']} for r,d in reps.items()},'lm_meta':lmmeta,'control_pool_meta':poolmeta,'qualification':qres,'qualification_gates':qg}
    if not all(qg[r['K']]['pass'] for r in reps.values()):
        out['verdict']='STA/AAA INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    hrows={};fits={}
    for rep,d in reps.items():hrows[rep],fits[rep],_=fit_rep(rep,d['Twords'],d['Hwords'],d['vocab'],lms,comps,d['Hcov']['coverage'])
    tops=[hrows[r]['top'] for r in ['family','sta','aaa']];hier=all(hrows[r]['gate'] for r in hrows) and len(set(tops))==1;out['H17']=hrows;out['hierarchy_pass']=hier
    if not hier:
        out['verdict']='REPRESENTATION-SENSITIVE / NO HIERARCHY SIGNAL' if len(set(tops))>1 else 'NO STA/AAA M19 SIGNAL';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    cand=tops[0];cres={};cpass=True
    for rep,d in reps.items():
        rank=rank_fixed(d['Cwords'],fits[rep],d['vocab'],lms);row={'top':rank[0][0],'margin':rank[0][1]-rank[1][1],'coverage':d['Ccov']['coverage'],'ranking':rank,'buckets':[]}
        for bkt in range(4):
            fs=[f for f in C if bucket_id(f)==bkt];w,_,cv=project(raw[('RF',rep)],fs,rep,d['vocab']);br=rank_fixed(w,fits[rep],d['vocab'],lms);row['buckets'].append({'bucket':bkt,'folios':len(fs),'top':br[0][0],'margin':br[0][1]-br[1][1],'coverage':cv['coverage']})
        row['pass']=row['top']==cand and row['margin']>=.05 and row['coverage']>=.97 and all(x['top']==cand and x['margin']>0 for x in row['buckets']);cres[rep]=row;cpass&=row['pass'];print('C17_RF',rep,json.dumps(row,separators=(',',':')),flush=True)
    out['C17_RF']=cres
    if not cpass:
        out['verdict']='H17 STA/AAA CANDIDATE / C17 FAILED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    transfers={};itpass=True;robpass=True
    for src in ['IT','ZL','GC']:
        transfers[src]={}
        for rep,d in reps.items():
            rr,_=evaluate_source(src,rep,raw[(src,rep)],C,d['vocab'],fits[rep],lms,False);transfers[src][rep]=rr;ok=rr['top']==cand and rr['coverage']['coverage']>=.95 and (rr['margin']>=.03 if src=='IT' else rr['margin']>0);itpass&=(ok if src=='IT' else True);robpass&=(ok if src!='IT' else True);print('TRANSFER',src,rep,json.dumps(rr,separators=(',',':')),flush=True)
    shortres={};shortpass=True
    for src in ['RF','IT']:
        shortres[src]={}
        for rep,d in reps.items():
            rr,_=evaluate_source(src,rep,raw_short[(src,rep)],C,d['vocab'],fits[rep],lms,True);shortres[src][rep]=rr;shortpass&=rr['top']==cand;print('SHORT',src,rep,json.dumps(rr,separators=(',',':')),flush=True)
    out['candidate']=cand;out['transfers']=transfers;out['short_word_robustness']=shortres;out['IT_pass']=itpass;out['ZL_GC_robustness_pass']=robpass;out['short_pass']=shortpass
    if not itpass:out['verdict']='RF CONFIRMED / IT FAILED'
    elif not (robpass and shortpass):out['verdict']='REPRESENTATION-SENSITIVE / NO HIERARCHY SIGNAL'
    else:out['verdict']='CONFIRMED STA/AAA M19 SIGNAL '+cand
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
