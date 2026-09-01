# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from unidecode import unidecode
from wordfreq import top_n_list, zipf_frequency

NS='VBMJOACHIMEXACTV9Q1'
DATA_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
ATOMS=('ckh','cth','cph','cfh','ch','sh','qo')
VOWELS=set('aeiou')
CONSONANTS='bcdfghjklmnpqrstvwxyz'
LOG5=math.log2(5); LOG21=math.log2(21)
FIXTURE='dcheedy kchedy lcheey ror al chokedy dol qokeeeos qolkeedy qokar ar'
FIXTURE_PLAIN='tizsichtrageundegetnichtsnelle'
UA={'User-Agent':'VBMJoachimExactV9Q1/2026-09-01'}
MAX_PER_B=80
MAX_CAND=5000
CORPUS_CHARS=400_000
SEED=90117


def hseed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)

def left_half(w):
    for a in ATOMS:
        if len(w)>=len(a)+1 and w.startswith(a): return a
    return w[0]

def parse_token(w):
    if not re.fullmatch(r'[a-z]+',w): return None
    if len(w)==1:return (w,'',w)
    L=left_half(w);R=w[-1]
    if len(w)<len(L)+1:return None
    return (L,w[len(L):-1],R)

def parse_line(txt):
    ws=txt.split()
    tr=[parse_token(w) for w in ws]
    if not ws or any(x is None for x in tr):return None
    ns=[x[1] for x in tr];bs=[a[2]+'|'+b[0] for a,b in zip(tr,tr[1:])]
    return {'tokens':ws,'nuclei':ns,'bridges':bs,'B':len(bs)}

def norm_letters(s):
    return ''.join(ch for ch in unidecode(s).lower() if 'a'<=ch<='z')

def make_bank(lang,tag):
    words=[];weights=[]
    for w in top_n_list(lang,30000):
        q=norm_letters(w)
        if not q or len(q)>24:continue
        z=zipf_frequency(w,lang)
        if not np.isfinite(z):continue
        words.append(q);weights.append(10.0**(0.45*z))
    if len(words)<1000:raise RuntimeError(('wordfreq shortage',lang,len(words)))
    p=np.asarray(weights,float);p/=p.sum();rng=np.random.default_rng(hseed(NS,'BANK',tag,SEED));parts=[];n=0
    while n<CORPUS_CHARS:
        ix=rng.choice(len(words),size=4096,p=p)
        block=''.join(words[int(i)] for i in ix);parts.append(block);n+=len(block)
    return ''.join(parts)[:CORPUS_CHARS]

def shuffle_bank(s):
    a=np.frombuffer(s.encode('ascii'),dtype=np.uint8).copy();rng=np.random.default_rng(hseed(NS,'SHUFFLE_DE',SEED));rng.shuffle(a);return bytes(a).decode('ascii')

def decompose(s):
    runs=[];vs=[];cur=[]
    for ch in s:
        if ch in VOWELS:
            runs.append(''.join(cur));cur=[];vs.append(ch)
        else:cur.append(ch)
    runs.append(''.join(cur))
    return np.asarray(runs,dtype=object),np.asarray(vs,dtype='<U1')

def groups(vals,skip_empty=False):
    d=collections.defaultdict(list)
    for i,x in enumerate(vals):
        if skip_empty and x=='':continue
        d[x].append(i)
    return list(d.values())

def sampled_starts(nv,B,linekey,bank):
    n=nv-B+1
    if n<=0:return np.asarray([],dtype=np.int64)
    if n<=MAX_CAND:return np.arange(n,dtype=np.int64)
    rng=np.random.default_rng(hseed(NS,'CAND',linekey,bank));return np.sort(rng.choice(n,size=MAX_CAND,replace=False).astype(np.int64))

def fit_windows(rec,runs,vs,starts):
    B=rec['B'];ns=rec['nuclei'];bs=rec['bridges'];m=len(starts)
    if m==0:return np.zeros(0,bool)
    lens=np.fromiter((len(x) for x in runs),dtype=np.int16,count=len(runs));mask=np.ones(m,bool)
    # Exact empty/non-empty and <=5 run-length requirements at every nucleus position.
    for p,n in enumerate(ns):
        ll=lens[starts+p]
        if n=='':mask &= (ll==0)
        else:mask &= ((ll>=1)&(ll<=5))
        if not mask.any():return mask
    # Same nucleus surface type -> same consonant run.
    for g in groups(ns,True):
        if len(g)>1:
            base=runs[starts+g[0]]
            for p in g[1:]:mask &= (runs[starts+p]==base)
            if not mask.any():return mask
    # Same bridge surface type -> same vowel.
    for g in groups(bs,False):
        if len(g)>1:
            base=vs[starts+g[0]]
            for p in g[1:]:mask &= (vs[starts+p]==base)
            if not mask.any():return mask
    return mask

def key_bits_for_fit(rec,runs,start):
    kb=len(set(rec['bridges']))*LOG5;kn=0.0
    seen={}
    for p,n in enumerate(rec['nuclei']):
        if not n or n in seen:continue
        val=str(runs[start+p]);seen[n]=val;kn+=LOG5+len(val)*LOG21
    return kb+kn

def one_score(rec,linekey,bankname,bdat):
    runs,vs=bdat;starts=sampled_starts(len(vs),rec['B'],linekey,bankname);mask=fit_windows(rec,runs,vs,starts);nf=int(mask.sum());nc=len(starts)
    frac=nf/max(1,nc);p=(nf+1)/(nc+1);surp=-math.log2(p)
    mink=None;net=None
    if nf:
        fi=np.flatnonzero(mask);mink=min(key_bits_for_fit(rec,runs,int(starts[int(j)])) for j in fi);net=surp-mink
    return {'candidates':nc,'fits':nf,'fit_fraction':frac,'smoothed_p':p,'fit_surprisal_bits':surp,'min_fresh_key_bits':mink,'net_single_line_bits':net}

def exact_plain_fit(rec,plain):
    runs,vs=decompose(norm_letters(plain));
    if len(vs)!=rec['B'] or len(runs)!=rec['B']+1:return False,None
    st=np.asarray([0],dtype=np.int64);mask=fit_windows(rec,runs,vs,st)
    if not len(mask) or not bool(mask[0]):return False,None
    return True,key_bits_for_fit(rec,runs,0)

def line_sample(data):
    strata=collections.defaultdict(list)
    for fid,lines in sorted(data['pages'].items()):
        if fid in H1 or fid in C1:continue
        for ln,obj in lines.items():
            txt=obj.get('t',{}).get('ZLZI','')
            if not txt:continue
            r=parse_line(txt)
            if r is None or not (5<=r['B']<=15) or len(r['tokens'])<6:continue
            key=f'{fid}:{ln}';r.update({'folio':fid,'line':str(ln),'key':key,'text':txt});strata[r['B']].append(r)
    out=[]
    for B in sorted(strata):
        z=sorted(strata[B],key=lambda r:hashlib.sha256(f'{NS}::{r["key"]}'.encode()).hexdigest())[:MAX_PER_B];out.extend(z)
    return out,{str(B):len(v) for B,v in sorted(strata.items())},{str(B):min(MAX_PER_B,len(v)) for B,v in sorted(strata.items())}

def aggregate(rows,bank):
    vals=[r['scores'][bank] for r in rows];fr=np.asarray([x['fit_fraction'] for x in vals],float);su=np.asarray([x['fit_surprisal_bits'] for x in vals],float);fits=np.asarray([x['fits'] for x in vals],int)
    by={}
    for B in sorted(set(r['B'] for r in rows)):
        vv=[r['scores'][bank] for r in rows if r['B']==B]
        by[str(B)]={'lines':len(vv),'median_fit_fraction':float(np.median([x['fit_fraction'] for x in vv])),'median_fit_surprisal_bits':float(np.median([x['fit_surprisal_bits'] for x in vv])),'fraction_ge1_fit':float(np.mean([x['fits']>=1 for x in vv]))}
    return {'lines':len(vals),'median_fit_fraction':float(np.median(fr)),'median_fit_surprisal_bits':float(np.median(su)),'fraction_ge1_fit':float(np.mean(fits>=1)),'fraction_ge10_fits':float(np.mean(fits>=10)),'mean_fit_fraction':float(np.mean(fr)),'by_bridge_count':by}

def shuffled_rec(rec,rng):
    ns=list(rec['nuclei']);bs=list(rec['bridges']);rng.shuffle(ns);rng.shuffle(bs);return {**rec,'nuclei':ns,'bridges':bs}

def shuffle_null(rows,de_bank):
    subset=sorted(rows,key=lambda r:hashlib.sha256(f'{NS}::NULL::{r["key"]}'.encode()).hexdigest())[:120]
    real=float(np.median([r['scores']['DE']['fit_surprisal_bits'] for r in subset])) if subset else float('nan');med=[]
    for rr in range(20):
        vals=[]
        for r in subset:
            rng=np.random.default_rng(hseed(NS,'NULL',rr,r['key']));q=shuffled_rec(r,rng);sc=one_score(q,r['key']+f':NULL{rr}','DE_NULL',de_bank);vals.append(sc['fit_surprisal_bits'])
        med.append(float(np.median(vals)))
    mu=float(np.mean(med));sd=float(np.std(med,ddof=1));z=(real-mu)/sd if sd>0 else 0.0
    return {'lines':len(subset),'null_replicates':20,'real_median_surprisal_bits':real,'shuffle_medians':med,'shuffle_mean':mu,'shuffle_sd':sd,'real_minus_shuffle_z':z}

def classify(de,sh):
    f=de['median_fit_fraction'];ge1=de['fraction_ge1_fit'];ge10=de['fraction_ge10_fits'];z=sh['real_minus_shuffle_z']
    if f>=.01 or ge10>=.50:return 'HIGHLY_NONSELECTIVE'
    if f>=.001 or ge1>=.50:return 'NONSELECTIVE'
    if f<1e-4 and ge1<=.25 and z>=2:return 'SELECTIVE'
    if 1e-4<=f<1e-3 and z>=2:return 'INTERMEDIATE'
    return 'NONSELECTIVE'  # conflicts/failed topology condition use less favourable band

def main():
    data=get_json(DATA_URL);sample,eligible,selected=line_sample(data)
    banks={}
    for name,la in [('DE','de'),('IT','it'),('EN','en')]:
        s=make_bank(la,name);banks[name]=decompose(s);print(json.dumps({'event':'bank','bank':name,'chars':len(s),'vowels':len(banks[name][1])}),flush=True)
    de_string=''.join(list(banks['DE'][0])[:0])  # sentinel; shuffled bank is regenerated from same deterministic source
    sde=make_bank('de','DE');banks['SHUFFLED_DE']=decompose(shuffle_bank(sde))
    rows=[]
    for i,r in enumerate(sample):
        scores={b:one_score(r,r['key'],b,banks[b]) for b in banks};rows.append({'folio':r['folio'],'line':r['line'],'key':r['key'],'B':r['B'],'tokens':len(r['tokens']),'unique_nuclei':len(set(x for x in r['nuclei'] if x)),'unique_bridges':len(set(r['bridges'])),'scores':scores})
        if (i+1)%100==0:print(json.dumps({'event':'lines_done','done':i+1,'total':len(sample)}),flush=True)
    agg={b:aggregate(rows,b) for b in banks};sh=shuffle_null(rows,banks['DE']);band=classify(agg['DE'],sh)
    fr=parse_line(FIXTURE);ok,kbits=exact_plain_fit(fr,FIXTURE_PLAIN);fp_runs,fp_vs=decompose(FIXTURE_PLAIN);fixture={'parse_exact_fit':ok,'tokens':len(fr['tokens']),'bridges':fr['B'],'unique_bridge_types':len(set(fr['bridges'])),'unique_nonempty_nucleus_types':len(set(x for x in fr['nuclei'] if x)),'fresh_key_bits':kbits,'plaintext_chars':len(FIXTURE_PLAIN),'key_bits_per_plaintext_char':kbits/len(FIXTURE_PLAIN) if kbits else None}
    out={'protocol':'VBM_JOACHIM_EXACT_V9_Q1_FRESHFIT_PROTOCOL.md','namespace':NS,'fixture':fixture,'eligible_by_B':eligible,'selected_by_B':selected,'sample_lines':len(rows),'banks':agg,'structural_shuffle_null':sh,'interpretation_band':band,'decision':'FRESH_FIT_NO_EVIDENTIAL_WEIGHT_REQUIRE_GLOBAL_CODEBOOK' if band in {'HIGHLY_NONSELECTIVE','NONSELECTIVE'} else 'MAY_PREREGISTER_GLOBAL_CODEBOOK_IDENTIFIABILITY','target_firewall':{'H1':sorted(H1),'C1':sorted(C1),'opened':False},'rows':rows}
    print('VBM_V9_Q1_RESULT='+json.dumps(out,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
