#!/usr/bin/env python3
import json, math, random, re, hashlib, urllib.request, unicodedata
from collections import Counter
import numpy as np
from unidecode import unidecode

SEED0 = 20260808
FOLIO = 'f10r'
TRANSCRIBERS = ['ZLZI','TTLI','PCCA','FFSG','GCGA','VDRB','RGVN']
ALPH = 'abcdefghiklmnopqrstuxyz'
A2I = {c:i for i,c in enumerate(ALPH)}
N = len(ALPH)
CAP_AGGR = {'a':7,'b':7,'c':7,'d':10,'e':10,'f':8,'g':10,'h':10,'i':10,'k':9,'l':9,'m':9,'n':9,'o':7,'p':10,'q':9,'r':9,'s':6,'t':9,'u':6,'x':8,'y':10,'z':10}
MODELS = {
    'G1': ('glyph', {c:1 for c in ALPH}),
    'G2': ('glyph', {c:4 for c in ALPH}),
    'G3': ('glyph', CAP_AGGR),
    'T2': ('token', {c:4 for c in ALPH}),
    'T3': ('token', CAP_AGGR),
}
CORPORA = {
    'latin': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu'],
    'italian': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu'],
    'german': ['https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu'],
    'french': [
        'https://raw.githubusercontent.com/UniversalDependencies/UD_Old_French-SRCMF/master/fro_srcmf-ud-train.conllu',
        'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu'],
    'greek': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu'],
    'hebrew': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu'],
    'arabic': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu'],
    'spanish': ['https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu'],
}
STEPS = 2200
RESTARTS = 2
NULLS_ALL = 4
NULLS_ZLZI = 12


def stable_seed(*parts):
    h = hashlib.sha256(('::'.join(map(str,parts))).encode()).digest()
    return (SEED0 + int.from_bytes(h[:8],'big')) & 0xffffffff


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read().decode('utf-8')


def conllu_sentences(txt, max_chars=2500000):
    out=[]; cur=[]; total=0
    for line in txt.splitlines():
        if not line:
            if cur:
                s=' '.join(cur); out.append(s); total += len(s); cur=[]
                if total >= max_chars: break
            continue
        if line.startswith('#'): continue
        cols=line.split('\t')
        if len(cols)>=2 and cols[0].isdigit(): cur.append(cols[1])
    if cur and total < max_chars: out.append(' '.join(cur))
    return out


def normalize_plain(s, keep_space=True):
    s = unidecode(s).lower()
    s = s.replace('j','i').replace('v','u').replace('w','u')
    if keep_space:
        words=[]
        for w in re.findall(r'[a-z]+',s):
            z=''.join(c for c in w if c in A2I)
            if z: words.append(z)
        return ' '.join(words)
    return ''.join(c for c in s if c in A2I)


def load_corpora():
    corp={}; meta={}
    for lang,urls in CORPORA.items():
        used=None; err=[]
        for u in urls:
            try:
                txt=fetch(u); used=u; break
            except Exception as e: err.append(repr(e))
        if used is None: raise RuntimeError((lang,err))
        sents=conllu_sentences(txt)
        corp[lang]=sents
        meta[lang]={'url':used,'sentences':len(sents),'raw_chars':sum(map(len,sents))}
        print('CORPUS',lang,len(sents),sum(map(len,sents)),used,flush=True)
    return corp,meta


def build_lm(sents, keep_space):
    K=N+1 if keep_space else N
    V=K**4
    counts=np.zeros(V,dtype=np.float64)
    unig=np.ones(N,dtype=np.float64)*0.1
    chars=0
    for raw in sents:
        s=normalize_plain(raw,keep_space=keep_space)
        if len(s)<4: continue
        arr=[]
        for c in s:
            if c==' ' and keep_space: arr.append(N)
            elif c in A2I:
                arr.append(A2I[c]); unig[A2I[c]] += 1
        a=np.asarray(arr,dtype=np.int64)
        if len(a)<4: continue
        idx=((a[:-3]*K+a[1:-2])*K+a[2:-1])*K+a[3:]
        counts += np.bincount(idx,minlength=V)
        chars += len(a)
    alpha=0.05
    logp=np.log((counts+alpha)/(counts.sum()+alpha*V))
    unig /= unig.sum()
    return logp,K,unig,chars


def folio_text(data,tid):
    lines=data['pages'][FOLIO]
    return ' '.join(line.get('t',{}).get(tid,'') for _,line in sorted(lines.items(),key=lambda kv:int(kv[0]) if kv[0].isdigit() else 99999)).strip()


def make_cipher(text, rendering):
    if rendering=='glyph':
        syms=sorted(set(c.lower() for c in text if not c.isspace()))
        s2i={s:i for i,s in enumerate(syms)}
        seq=[]
        for c in text.lower():
            if c.isspace(): seq.append(-1)
            else: seq.append(s2i[c])
        return np.asarray(seq,dtype=np.int32), syms
    toks=text.lower().split()
    syms=sorted(set(toks)); s2i={s:i for i,s in enumerate(syms)}
    return np.asarray([s2i[t] for t in toks],dtype=np.int32), syms


def score_mapping(cseq,mapping,logp,K,rendering):
    if rendering=='glyph':
        p=np.empty(len(cseq),dtype=np.int16)
        mask=cseq<0; p[mask]=N; p[~mask]=mapping[cseq[~mask]]
    else:
        p=mapping[cseq]
    if len(p)<4: return -1e9
    idx=((p[:-3]*K+p[1:-2])*K+p[2:-1])*K+p[3:]
    return float(logp[idx].mean())


def init_mapping(nsym,caps,unig,rng):
    cap=np.array([caps[c] for c in ALPH],dtype=int)
    if nsym>cap.sum(): return None
    # frequency-aware but randomized capacity pool
    pool=[]
    for i,nc in enumerate(cap): pool += [i]*int(nc)
    weights=np.array([unig[i] for i in pool],dtype=float); weights/=weights.sum()
    chosen=[]; avail=list(range(len(pool)))
    # weighted without replacement over capacity slots
    for _ in range(nsym):
        ww=np.array([weights[j] for j in avail]); ww/=ww.sum()
        k=int(rng.choice(len(avail),p=ww)); chosen.append(pool[avail[k]]); avail.pop(k)
    rng.shuffle(chosen)
    return np.asarray(chosen,dtype=np.int16)


def optimize(cseq,nsym,caps,unig,logp,K,rendering,seed):
    rng=np.random.default_rng(seed)
    cap=np.array([caps[c] for c in ALPH],dtype=int)
    if nsym>cap.sum(): return None
    best_s=-1e99; best_m=None
    for rr in range(RESTARTS):
        m=init_mapping(nsym,caps,unig,rng)
        cnt=np.bincount(m,minlength=N).astype(int)
        s=score_mapping(cseq,m,logp,K,rendering)
        if s>best_s: best_s,best_m=s,m.copy()
        for step in range(STEPS):
            frac=step/max(1,STEPS-1); T=0.22*(1-frac)+0.005
            si=int(rng.integers(nsym)); old=int(m[si]); new=int(rng.integers(N))
            if new==old: continue
            m2=m.copy()
            if cnt[new] < cap[new]:
                m2[si]=new
            else:
                cand=np.flatnonzero(m==new)
                if len(cand)==0: continue
                sj=int(rng.choice(cand)); m2[si]=new; m2[sj]=old
            s2=score_mapping(cseq,m2,logp,K,rendering)
            if s2>=s or rng.random()<math.exp(max(-50,min(50,(s2-s)/T))):
                if m2[si]!=old:
                    cnt=np.bincount(m2,minlength=N).astype(int)
                m,s=m2,s2
                if s>best_s: best_s,best_m=s,m.copy()
    return best_s,best_m


def decode_text(cseq,mapping,rendering):
    if rendering=='glyph':
        out=[]
        for x in cseq:
            out.append(' ' if x<0 else ALPH[int(mapping[int(x)])])
        return ''.join(out)
    return ''.join(ALPH[int(mapping[int(x)])] for x in cseq)


def shuffle_cipher(cseq,rendering,rng):
    x=cseq.copy()
    if rendering=='glyph':
        mask=x>=0; vals=x[mask].copy(); rng.shuffle(vals); x[mask]=vals
    else: rng.shuffle(x)
    return x


def main():
    slim_url='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
    data=json.loads(fetch(slim_url))
    corp,corp_meta=load_corpora()
    lms={}
    for lang,sents in corp.items():
        lms[(lang,'glyph')]=build_lm(sents,True)
        lms[(lang,'token')]=build_lm(sents,False)
    results=[]
    for tid in TRANSCRIBERS:
        text=folio_text(data,tid)
        for model,(rendering,caps) in MODELS.items():
            cseq,syms=make_cipher(text,rendering)
            if len(syms)>sum(caps.values()):
                results.append({'tid':tid,'model':model,'status':'INFEASIBLE','nsym':len(syms)}); continue
            for lang in CORPORA:
                logp,K,unig,_=lms[(lang,rendering)]
                opt=optimize(cseq,len(syms),caps,unig,logp,K,rendering,stable_seed(tid,model,lang,'true'))
                if opt is None:
                    results.append({'tid':tid,'model':model,'lang':lang,'status':'INFEASIBLE'}); continue
                sc,mp=opt
                nnull=NULLS_ZLZI if tid=='ZLZI' else NULLS_ALL
                null=[]
                for j in range(nnull):
                    rg=np.random.default_rng(stable_seed(tid,model,lang,'shuffle',j))
                    sh=shuffle_cipher(cseq,rendering,rg)
                    so=optimize(sh,len(syms),caps,unig,logp,K,rendering,stable_seed(tid,model,lang,'nullopt',j))
                    null.append(so[0])
                mu=float(np.mean(null)); sd=float(np.std(null,ddof=1)) if len(null)>1 else 0.0
                z=(sc-mu)/sd if sd>1e-12 else 0.0
                emp=(1+sum(v>=sc for v in null))/(1+len(null))
                results.append({'tid':tid,'model':model,'lang':lang,'status':'OK','nsym':len(syms),'score':sc,'null_mean':mu,'null_sd':sd,'z':z,'emp_p':emp,'decode':decode_text(cseq,mp,rendering)[:800]})
            print('DONE',tid,model,flush=True)
    # rank languages by z within each tid/model
    tops=[]
    for tid in TRANSCRIBERS:
        for model in MODELS:
            rr=[r for r in results if r.get('tid')==tid and r.get('model')==model and r.get('status')=='OK']
            if not rr: continue
            rr=sorted(rr,key=lambda r:r['z'],reverse=True)
            tops.append({'tid':tid,'model':model,'top':rr[0]['lang'],'z':rr[0]['z'],'score':rr[0]['score'],'decode':rr[0]['decode'],'ranking':[(r['lang'],r['z']) for r in rr]})
    # predeclared signal rule
    signals=[]
    for model in MODELS:
        tm=[x for x in tops if x['model']==model]
        cc=Counter(x['top'] for x in tm)
        if not cc: continue
        lang,n=cc.most_common(1)[0]
        zr=next((r['z'] for r in results if r.get('tid')=='ZLZI' and r.get('model')==model and r.get('lang')==lang and r.get('status')=='OK'),None)
        if n>=5 and zr is not None and zr>=3.0:
            verdict='FLEXIBILITY_ONLY' if model=='T3' else 'PILOT_SIGNAL'
        else: verdict='NONRESOLVING'
        signals.append({'model':model,'consensus_top':lang,'families':n,'zlzi_z':zr,'verdict':verdict})
    out={'folio':FOLIO,'transcribers':TRANSCRIBERS,'parameters':{'steps':STEPS,'restarts':RESTARTS,'nulls_all':NULLS_ALL,'nulls_zlzi':NULLS_ZLZI},'corpora':corp_meta,'signals':signals,'tops':tops,'results':results}
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__': main()
