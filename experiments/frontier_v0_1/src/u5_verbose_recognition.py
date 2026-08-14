from __future__ import annotations
import argparse, csv, hashlib, json, math, random, statistics, unicodedata, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHABET="ABCDEFGHILMNOPQRSTUVXYZ"
ART="abcdefghijklmnopqrst"
STATES=("alpha","beta1","beta2","beta3","gamma1","gamma2")
UNIGRAM_DECK=(0,1,2,3,4,5,0,1,2,3)
TRAIN_URLS={"la":"https://www.gutenberg.org/cache/epub/218/pg218.txt","it":"https://www.gutenberg.org/cache/epub/52484/pg52484.txt"}
TEST_FILES={"la":"input/examples/nathist_book16.txt","it":"input/examples/divina_commedia.txt"}
SAMPLE_CHARS=4096
FEATURE_NAMES=[
 "type_token","hapax_type_share","repeat_occurrence_share",
 "prefix_support_1","prefix_support_2","prefix_support_3","suffix_support_1","suffix_support_2","suffix_support_3",
 "best_split_log_support","recomposable_ge3","recomposable_ge5","dictionary_cost_ratio","rectangle_edge_fraction",
 "suffix_partner_entropy","prefix_partner_entropy","within_equal_rate","boundary_equal_rate","within_bigram_mi","boundary_bigram_mi",
 "best_lcp_fraction","best_lcs_fraction",
]
NULL_FAMILIES=("type_recode","global_char_reshuffle","token_internal_shuffle","dependent_slot3")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"big") & 0x7fffffffffffffff


def normalize(text):
    text=unicodedata.normalize("NFKD",text)
    text="".join(c for c in text if not unicodedata.combining(c)).upper().replace("W","UU").replace("J","I").replace("K","C")
    return "".join(c for c in text if c in ALPHABET)


def strip_gutenberg(text):
    lines=text.splitlines(); start=0; end=len(lines)
    for i,x in enumerate(lines):
        if "*** START OF" in x.upper(): start=i+1; break
    for i in range(start,len(lines)):
        if "*** END OF" in lines[i].upper(): end=i; break
    return "\n".join(lines[start:end])


def fetch(url, path):
    if not path.exists():
        req=urllib.request.Request(url,headers={"User-Agent":"VoynichFrontierU5B/0.1"})
        with urllib.request.urlopen(req,timeout=60) as r: path.write_bytes(r.read())
    return path.read_text(encoding="utf-8",errors="ignore")


def parse_role_lengths(path):
    out={role:{s:[] for s in STATES} for role in ("unigram","prefix","suffix")}
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            code=r["code"]
            role=next((z for z in out if code.startswith(z+"_")),None)
            if role is None: continue
            rem=code[len(role)+1:]
            state=next(s for s in STATES if rem.startswith(s+"_"))
            out[role][state].append(len(r["glyphs"]))
    for role in out:
        for s in STATES:
            if len(out[role][s])!=23: raise RuntimeError(f"bad length table {role}/{s}: {len(out[role][s])}")
    return out


def fresh_word(rng, L, used):
    cap=len(ART)**L
    if sum(len(x)==L for x in used)>=cap: return None
    for _ in range(10000):
        s="".join(rng.choice(ART) for _ in range(L))
        if s not in used: used.add(s); return s
    return None


def build_codebook(lengths, rng):
    cb={role:{s:{} for s in STATES} for role in lengths}
    for role in lengths:
        for s in STATES:
            used=set()
            for letter in ALPHABET:
                for _ in range(1000):
                    L=rng.choice(lengths[role][s])
                    w=fresh_word(rng,L,used)
                    if w is not None: break
                if w is None: raise RuntimeError("codeword generation exhausted")
                cb[role][s][letter]=w
    return cb


def choose_chunk(text, seed):
    if len(text)<SAMPLE_CHARS: raise RuntimeError(f"source too short {len(text)}")
    rng=random.Random(seed); start=rng.randrange(0,len(text)-SAMPLE_CHARS+1)
    return text[start:start+SAMPLE_CHARS],start


def generate_verbose(plain, lengths, seed):
    rng=random.Random(seed); cb=build_codebook(lengths,rng)
    perm=list(ALPHABET);rng.shuffle(perm); enc={a:b for a,b in zip(ALPHABET,perm)}
    unigram_set={w for s in STATES for w in cb["unigram"][s].values()}
    tokens=[];meta=[];i=0
    while i<len(plain):
        width=1 if i==len(plain)-1 else rng.choice((1,2))
        if width==1:
            st=STATES[rng.choice(UNIGRAM_DECK)];t=cb["unigram"][st][enc[plain[i]]]
        else:
            for _ in range(1000):
                a=STATES[rng.randrange(6)];b=STATES[rng.randrange(6)]
                t=cb["prefix"][a][enc[plain[i]]]+cb["suffix"][b][enc[plain[i+1]]]
                if t not in unigram_set: break
            else: raise RuntimeError("unambiguous compound generation failed")
        tokens.append(t);meta.append((i,width));i+=width
    return tokens,meta


def random_string(rng,L): return "".join(rng.choice(ART) for _ in range(L))


def null_type_recode(tokens,seed):
    rng=random.Random(seed); mapping={};used=defaultdict(set);out=[]
    for t in tokens:
        if t not in mapping:
            for _ in range(10000):
                z=random_string(rng,len(t))
                if z not in used[len(t)]: used[len(t)].add(z);mapping[t]=z;break
            else: raise RuntimeError("type recode exhausted")
        out.append(mapping[t])
    return out


def null_global_shuffle(tokens,seed):
    rng=random.Random(seed);chars=list("".join(tokens));rng.shuffle(chars);out=[];i=0
    for t in tokens: out.append("".join(chars[i:i+len(t)]));i+=len(t)
    return out


def null_internal_shuffle(tokens,seed):
    rng=random.Random(seed);out=[]
    for t in tokens:
        z=list(t);rng.shuffle(z);out.append("".join(z))
    return out


def null_slot3(tokens,plain,meta,seed):
    rng=random.Random(seed); dictionaries={}
    def comp(slot,L,idx):
        key=(slot,L,idx%10)
        if key not in dictionaries: dictionaries[key]=random_string(rng,L)
        return dictionaries[key]
    to_i={c:i for i,c in enumerate(ALPHABET)};out=[]
    for t,(pos,width) in zip(tokens,meta):
        L=len(t);a=to_i[plain[pos]];b=to_i[plain[min(pos+1,len(plain)-1)]]
        if L==1: out.append(comp("core",1,(a+3*b)%10))
        elif L==2: out.append(comp("pre",1,a)+comp("suf",1,b))
        else: out.append(comp("pre",1,a)+comp("core",L-2,(a+3*b)%10)+comp("suf",1,b))
    return out


def canonical(tokens):
    mp={};n=0;out=[]
    for t in tokens:
        z=[]
        for c in t:
            if c not in mp:
                mp[c]=chr(0x100+n);n+=1
            z.append(mp[c])
        out.append("".join(z))
    return out


def mi(pairs):
    if not pairs:return 0.0
    xy=Counter(pairs);x=Counter(a for a,_ in pairs);y=Counter(b for _,b in pairs);n=len(pairs);v=0.0
    for (a,b),c in xy.items():
        p=c/n;v+=p*math.log2(p/((x[a]/n)*(y[b]/n)))
    return v


def entropy_counts(vals):
    c=Counter(vals);n=sum(c.values())
    if n<=1 or len(c)<=1:return 0.0
    return -sum((v/n)*math.log2(v/n) for v in c.values())


def features(raw_tokens):
    tokens=canonical(raw_tokens);cnt=Counter(tokens);types=list(cnt);nt=len(tokens);nv=len(types)
    f={"type_token":nv/nt if nt else 0.0,"hapax_type_share":sum(v==1 for v in cnt.values())/nv if nv else 0.0,"repeat_occurrence_share":1-sum(1 for t in tokens if cnt[t]==1)/nt if nt else 0.0}
    for k in (1,2,3):
        pc=Counter(t[:k] for t in types if len(t)>=k);sc=Counter(t[-k:] for t in types if len(t)>=k)
        den=max(1,nv-1);w=sum(cnt.values()) or 1
        f[f"prefix_support_{k}"]=sum(cnt[t]*((pc[t[:k]]-1)/den if len(t)>=k else 0) for t in types)/w
        f[f"suffix_support_{k}"]=sum(cnt[t]*((sc[t[-k:]]-1)/den if len(t)>=k else 0) for t in types)/w
    pc=Counter();sc=Counter()
    for t in types:
        for k in range(1,len(t)):pc[t[:k]]+=1;sc[t[k:]]+=1
    selected={};logsup=ge3=ge5=0.0;weight=sum(cnt.values()) or 1
    for t in types:
        best=None
        if len(t)>=2:
            for k in range(1,len(t)):
                p,s=t[:k],t[k:];sup=min(pc[p],sc[s]);cand=(sup,pc[p]*sc[s],-k,p,s)
                if best is None or cand>best:best=cand
        if best:
            sup,_,_,p,s=best;selected[t]=(p,s);logsup+=cnt[t]*math.log1p(sup);ge3+=cnt[t]*(sup>=3);ge5+=cnt[t]*(sup>=5)
        else:selected[t]=(t,"")
    f["best_split_log_support"]=logsup/weight;f["recomposable_ge3"]=ge3/weight;f["recomposable_ge5"]=ge5/weight
    flat=sum(len(t) for t in types) or 1
    P={p for p,s in selected.values() if s};S={s for p,s in selected.values() if s};atoms={t for t,(p,s) in selected.items() if not s}
    f["dictionary_cost_ratio"]=(sum(map(len,P))+sum(map(len,S))+sum(map(len,atoms)))/flat
    edges={(p,s) for p,s in selected.values() if s};pa=defaultdict(set);sa=defaultdict(set)
    for p,s in edges:pa[p].add(s);sa[s].add(p)
    participating=0
    for p,s in edges:
        hit=False
        for p2 in sa[s]-{p}:
            if (pa[p]-{s}) & (pa[p2]-{s}):hit=True;break
        participating+=hit
    f["rectangle_edge_fraction"]=participating/len(edges) if edges else 0.0
    pref_vals=[];suf_vals=[]
    for p,ss in pa.items():pref_vals.extend([s for t,(pp,s) in selected.items() if s and pp==p for _ in range(cnt[t])])
    for s,pp in sa.items():suf_vals.extend([p for t,(p,ss) in selected.items() if ss and ss==s for _ in range(cnt[t])])
    # Weighted mean conditional partner entropy.
    pe=[];pw=[]
    for p in pa:
        vals=[]
        for t,(pp,s) in selected.items():
            if s and pp==p:vals.extend([s]*cnt[t])
        pe.append(entropy_counts(vals));pw.append(len(vals))
    se=[];sw=[]
    for s in sa:
        vals=[]
        for t,(p,ss) in selected.items():
            if ss and ss==s:vals.extend([p]*cnt[t])
        se.append(entropy_counts(vals));sw.append(len(vals))
    f["suffix_partner_entropy"]=float(np.average(pe,weights=pw)) if pe and sum(pw) else 0.0
    f["prefix_partner_entropy"]=float(np.average(se,weights=sw)) if se and sum(sw) else 0.0
    inside=[(a,b) for t in tokens for a,b in zip(t,t[1:])];bound=[(tokens[i][-1],tokens[i+1][0]) for i in range(len(tokens)-1) if tokens[i] and tokens[i+1]]
    f["within_equal_rate"]=sum(a==b for a,b in inside)/len(inside) if inside else 0.0;f["boundary_equal_rate"]=sum(a==b for a,b in bound)/len(bound) if bound else 0.0
    f["within_bigram_mi"]=mi(inside);f["boundary_bigram_mi"]=mi(bound)
    pfx=Counter();sfx=Counter()
    for t in types:
        for k in range(1,len(t)+1):pfx[t[:k]]+=1;sfx[t[-k:]]+=1
    lcp=lcs=0.0
    for t in types:
        mp=ms=0
        for k in range(1,len(t)+1):
            if pfx[t[:k]]>=2:mp=k
            if sfx[t[-k:]]>=2:ms=k
        lcp+=cnt[t]*(mp/len(t));lcs+=cnt[t]*(ms/len(t))
    f["best_lcp_fraction"]=lcp/weight;f["best_lcs_fraction"]=lcs/weight
    return np.array([f[n] for n in FEATURE_NAMES],dtype=float),f


def make_sample(plain,lengths,seed):
    pos,meta=generate_verbose(plain,lengths,stable_seed(seed,"positive"))
    return {
      "positive":pos,
      "type_recode":null_type_recode(pos,stable_seed(seed,"n1")),
      "global_char_reshuffle":null_global_shuffle(pos,stable_seed(seed,"n2")),
      "token_internal_shuffle":null_internal_shuffle(pos,stable_seed(seed,"n3")),
      "dependent_slot3":null_slot3(pos,plain,meta,stable_seed(seed,"n4")),
    }


def source_chunks(text,n,tag):
    return [choose_chunk(text,stable_seed("u5b-chunk",tag,i)) for i in range(n)]


def metrics(y,prob,thr,fams):
    pred=prob>=thr;tp=int(np.sum(pred & (y==1)));fp=int(np.sum(pred & (y==0)));fn=int(np.sum((~pred)&(y==1)))
    precision=tp/(tp+fp) if tp+fp else 1.0;recall=tp/(tp+fn) if tp+fn else 0.0
    fprs={}
    for fam in NULL_FAMILIES:
        ix=np.array([x==fam for x in fams]);den=int(np.sum(ix));fprs[fam]=float(np.sum(pred & ix)/den) if den else 0.0
    return {"precision":precision,"recall":recall,"aggregate_fpr":fp/max(1,int(np.sum(y==0))),"per_family_fpr":fprs,"tp":tp,"fp":fp,"fn":fn}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--naibbe-repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lengths=parse_role_lengths(a.naibbe_repo/'references'/'naibbe_tables.csv');cache=a.out/'source_cache';cache.mkdir(exist_ok=True)
    devsrc={}
    for iso in ('la','it'):devsrc[iso]=normalize(strip_gutenberg(fetch(TRAIN_URLS[iso],cache/f'train_{iso}.txt')))
    locksrc={iso:normalize((a.naibbe_repo/TEST_FILES[iso]).read_text(encoding='utf-8',errors='ignore')) for iso in ('la','it')}
    provenance={"train_lengths":{k:len(v) for k,v in devsrc.items()},"locked_lengths":{k:len(v) for k,v in locksrc.items()},"feature_names":FEATURE_NAMES,"voynich_read":False,"target_opened":False}

    rows=[]
    # 60/source dev positives: first 40 fit, last 20 threshold calibration.
    for iso in ('la','it'):
        for i,(plain,start) in enumerate(source_chunks(devsrc[iso],60,f'dev-{iso}')):
            group='fit' if i<40 else 'calibration';surfs=make_sample(plain,lengths,stable_seed('u5b-dev',iso,i))
            for fam,toks in surfs.items():
                x,fd=features(toks);rows.append((group,1 if fam=='positive' else 0,fam,iso,i,x,fd,start))
    def matrix(group):
        z=[r for r in rows if r[0]==group];return np.vstack([r[5] for r in z]),np.array([r[1] for r in z]),[r[2] for r in z]
    Xf,yf,ff=matrix('fit');Xc,yc,fc=matrix('calibration')
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',random_state=20260814,max_iter=1000));clf.fit(Xf,yf);pc=clf.predict_proba(Xc)[:,1]
    candidates=sorted(set(pc.tolist()),reverse=True);valid=[]
    for t in candidates:
        m=metrics(yc,pc,t,fc)
        if m['recall']>0 and m['precision']>=0.95 and max(m['per_family_fpr'].values())<=0.05:valid.append((m['recall'],t,m))
    if not valid:
        result={"schema":"frontier-u5-b-v0.1","formal_verdict":"RECOVERABLE_NOT_IDENTIFIABLE","stage":"CALIBRATION_THRESHOLD_FAIL","target_opened":False,"voynich_read":False,"calibration":{"n":len(yc),"best_possible":None},"provenance":provenance}
        (a.out/'U5B_RECOGNITION_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8');print('U5B_FINAL',json.dumps(result,sort_keys=True));return
    valid.sort(key=lambda x:(x[0],x[1]),reverse=True);_,thr,cal=valid[0]

    locked=[]
    for iso in ('la','it'):
        for i,(plain,start) in enumerate(source_chunks(locksrc[iso],50,f'locked-{iso}')):
            surfs=make_sample(plain,lengths,stable_seed('u5b-locked',iso,i))
            for fam,toks in surfs.items():
                x,fd=features(toks);locked.append((1 if fam=='positive' else 0,fam,iso,i,x,start))
    Xt=np.vstack([r[4] for r in locked]);yt=np.array([r[0] for r in locked]);ft=[r[1] for r in locked];pt=clf.predict_proba(Xt)[:,1];tm=metrics(yt,pt,thr,ft)
    passed=tm['recall']>=0.80 and tm['precision']>=0.95 and max(tm['per_family_fpr'].values())<=0.05
    coef=clf.named_steps['logisticregression'].coef_[0];top=sorted(zip(FEATURE_NAMES,coef),key=lambda x:abs(x[1]),reverse=True)
    result={"schema":"frontier-u5-b-v0.1","formal_verdict":"PASS_RECOGNITION_CALIBRATION" if passed else "RECOVERABLE_NOT_IDENTIFIABLE","stage":"LOCKED_TEST","target_opened":False,"voynich_read":False,"threshold":float(thr),"calibration_metrics":cal,"locked_metrics":tm,"fit_n":len(yf),"calibration_n":len(yc),"locked_n":len(yt),"classifier":"StandardScaler + LogisticRegression(C=1,class_weight=balanced,liblinear)","top_coefficients":[[n,float(v)] for n,v in top],"provenance":provenance,"consequence":"Voynich recognition target may open" if passed else "U5 closes under v0.1; Voynich remains sealed"}
    (a.out/'U5B_RECOGNITION_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8')
    md=['# U5-B fresh-codebook verbose recognition result','',f'Formal verdict: **{result["formal_verdict"]}**','',f'- frozen threshold: {thr:.6f}',f'- locked recall: {tm["recall"]:.4f} (gate ≥0.80)',f'- locked precision: {tm["precision"]:.4f} (gate ≥0.95)',f'- locked aggregate FPR: {tm["aggregate_fpr"]:.4f}',f'- max matched-null family FPR: {max(tm["per_family_fpr"].values()):.4f} (gate ≤0.05)','', 'Voynich was not read during U5-B calibration.']
    for fam,v in tm['per_family_fpr'].items():md.append(f'- {fam}: FPR {v:.4f}')
    (a.out/'U5B_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8');print('U5B_FINAL',json.dumps({"formal_verdict":result['formal_verdict'],"threshold":thr,**tm},sort_keys=True))

if __name__=='__main__':main()
