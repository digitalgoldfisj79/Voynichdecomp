#!/usr/bin/env python3
import json, math, random, re, hashlib, urllib.request, urllib.parse, html
from collections import Counter, defaultdict
import numpy as np
from unidecode import unidecode

SEED0 = 20260808
ALPH = "abcdefghiklmnopqrstuxyz"
A2I = {c:i for i,c in enumerate(ALPH)}
N = len(ALPH)
LENGTHS = [88,176,352]
REPS = [0,1]
MODELS = ["T2","T3"]
TARGETS = ["latin","italian","german","hebrew"]
LANGS = ["latin","italian","german","french","greek","hebrew","arabic","spanish"]
STEPS = 3500
RESTARTS = 4
NULLS = 4

TABLES = {
    "F":[1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24],
    "M":[1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4],
    "G":[1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3],
    "L":[1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7],
    "H":[1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7],
}
PAIR_ORDER = ["FM","FG","FL","FH","MG","ML","MH","GL","GH","LH"]
GLOBAL_PAIRS = ["FM","FG","FL","ML"]

LM_URLS = {
    "latin":"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu",
    "italian":"https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu",
    "german":"https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu",
    "french":"https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu",
    "greek":"https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu",
    "hebrew":"https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu",
    "arabic":"https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu",
    "spanish":"https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu",
}
HIST_URLS = {
    "latin":"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-LLCT/master/la_llct-ud-train.conllu",
    "italian":"https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-Old/master/it_old-ud-train.conllu",
    "german":"https://www.gutenberg.org/cache/epub/22636/pg22636.txt",
}
SEFARIA_INDEX = "https://raw.githubusercontent.com/Sefaria/Sefaria-Export/master/books.json"

def stable_seed(*parts):
    h=hashlib.sha256(("::".join(map(str,parts))).encode()).digest()
    return (SEED0 + int.from_bytes(h[:8],"big")) & 0xffffffff

def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()

def fetch(url):
    q=urllib.parse.quote(url, safe=":/?=&%")
    req=urllib.request.Request(q, headers={"User-Agent":"Mozilla/5.0 BnF-control/0.2"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8","replace")

def conllu_sentences(txt):
    out=[]; cur=[]
    for line in txt.splitlines():
        if not line:
            if cur:
                out.append(" ".join(cur)); cur=[]
            continue
        if line.startswith("#"): continue
        cols=line.split("\t")
        if len(cols)>=2 and cols[0].isdigit():
            cur.append(cols[1])
    if cur: out.append(" ".join(cur))
    return out

def normalize_common(s):
    s=html.unescape(s)
    s=re.sub(r"<[^>]+>"," ",s)
    s=unidecode(s).lower()
    s=s.replace("j","i").replace("v","u").replace("w","u")
    return "".join(c for c in s if c in A2I)

def normalize_mhg_ascii(s):
    def repl(m):
        z=re.sub(r"[^A-Za-z]","",m.group(1))
        return z
    s=re.sub(r"\[([^\]]+)\]", repl, s)
    s=s.replace("{","").replace("}","").replace("_","")
    return normalize_common(s)

def split_train_holdout(sents):
    train=[s for i,s in enumerate(sents) if i%5!=0]
    hold=[s for i,s in enumerate(sents) if i%5==0]
    return train,hold

def normalized_concat(sents, limit=None):
    out=[]; n=0
    for s in sents:
        z=normalize_common(s)
        if z:
            out.append(z); n+=len(z)
            if limit and n>=limit: break
    x="".join(out)
    return x[:limit] if limit else x

def load_all_sources():
    lm_sents={}; hold_text={}; source_meta={}
    for lang,u in LM_URLS.items():
        txt=fetch(u); ss=conllu_sentences(txt)
        if lang in TARGETS:
            tr,ho=split_train_holdout(ss)
            lm_sents[lang]=tr
            hold_text[lang]=normalized_concat(ho)
            source_meta[f"p0_{lang}"]={"url":u,"sentences_total":len(ss),"train":len(tr),"hold":len(ho),"hold_chars":len(hold_text[lang])}
        else:
            lm_sents[lang]=ss
            source_meta[f"lm_{lang}"]={"url":u,"sentences_total":len(ss)}
        print("SOURCE",lang,len(ss),flush=True)

    hist={}
    t=fetch(HIST_URLS["latin"]); ss=conllu_sentences(t); hist["latin"]=normalized_concat(ss)
    source_meta["p1_latin"]={"url":HIST_URLS["latin"],"sentences":len(ss),"chars":len(hist["latin"])}
    t=fetch(HIST_URLS["italian"]); ss=conllu_sentences(t); hist["italian"]=normalized_concat(ss)
    source_meta["p1_italian"]={"url":HIST_URLS["italian"],"sentences":len(ss),"chars":len(hist["italian"])}
    t=fetch(HIST_URLS["german"])
    st=t.find("D[o] erbiten si der nahte")
    if st<0: raise RuntimeError("MHG start marker missing")
    en=t.find("\nVIII.",st)
    if en<0: en=t.find("PARZIVAL",st)
    if en<0: raise RuntimeError("MHG end marker missing")
    raw=t[st:en]
    hist["german"]=normalize_mhg_ascii(raw)
    source_meta["p1_german"]={"url":HIST_URLS["german"],"start_marker":"D[o] erbiten si der nahte","chars":len(hist["german"]),"raw_chars":len(raw)}
    idx=json.loads(fetch(SEFARIA_INDEX)); books=idx["books"] if isinstance(idx,dict) else idx
    hits=[b for b in books if b.get("title")=="Mishneh Torah, Torah Study" and b.get("language")=="Hebrew" and b.get("versionTitle")=="Torat Emet 363"]
    if len(hits)!=1: raise RuntimeError(f"Hebrew source ambiguity: {len(hits)}")
    hu=hits[0]["json_url"]; obj=json.loads(fetch(hu)); chunks=[]
    def walk(x):
        if isinstance(x,str): chunks.append(x)
        elif isinstance(x,list):
            for y in x: walk(y)
    walk(obj.get("text",[]))
    hist["hebrew"]=normalize_common(" ".join(chunks))
    source_meta["p1_hebrew"]={"url":hu,"chars":len(hist["hebrew"]),"chunks":len(chunks)}
    return lm_sents,hold_text,hist,source_meta

def build_lm(sents, max_chars=2500000):
    V=N**4
    counts=np.zeros(V,dtype=np.float64)
    unig=np.ones(N,dtype=np.float64)*0.1
    chars=0
    for raw in sents:
        s=normalize_common(raw)
        if len(s)<4: continue
        a=np.fromiter((A2I[c] for c in s),dtype=np.int16,count=len(s))
        unig += np.bincount(a,minlength=N)
        idx=((a[:-3].astype(np.int64)*N+a[1:-2])*N+a[2:-1])*N+a[3:]
        counts += np.bincount(idx,minlength=V)
        chars += len(a)
        if chars>=max_chars: break
    alpha=.05
    logp=np.log((counts+alpha)/(counts.sum()+alpha*V))
    unig/=unig.sum()
    return logp,unig,chars

def lm_score_plain(text,logp):
    a=np.fromiter((A2I[c] for c in text),dtype=np.int16,count=len(text))
    if len(a)<4:return -1e99
    idx=((a[:-3].astype(np.int64)*N+a[1:-2])*N+a[2:-1])*N+a[3:]
    return float(logp[idx].mean())

def make_codebooks():
    vals_by_pair={}
    for p in PAIR_ORDER:
        a,b=p[0],p[1]
        vals=[(TABLES[a][i],TABLES[b][i]) for i in range(N)]
        vals_by_pair[p]=vals
    t2={c:[] for c in ALPH}; t3={c:[] for c in ALPH}
    for p in GLOBAL_PAIRS:
        vals=vals_by_pair[p]
        if len(set(vals))!=N: raise RuntimeError("global pair not injective "+p)
        for i,c in enumerate(ALPH):
            v=vals[i]; t2[c].append(f"{p}:{v[0]}:{v[1]}")
    for p,vals in vals_by_pair.items():
        ct=Counter(vals)
        for i,c in enumerate(ALPH):
            v=vals[i]
            if ct[v]==1: t3[c].append(f"{p}:{v[0]}:{v[1]}")
    cap2={c:len(t2[c]) for c in ALPH}; cap3={c:len(t3[c]) for c in ALPH}
    if set(cap2.values())!={4}: raise RuntimeError(cap2)
    if sum(cap3.values())!=199: raise RuntimeError(sum(cap3.values()))
    return {"T2":t2,"T3":t3},{"T2":cap2,"T3":cap3}

def choose_span(text,L,tier,lang,rep):
    if len(text)<L+20: raise RuntimeError((tier,lang,len(text),L))
    seed=stable_seed("span",tier,lang,L,rep)
    start=seed%(len(text)-L+1)
    return text[start:start+L],int(start)

def encrypt(plain,model,codebooks,tier,lang,L,rep):
    rng=np.random.default_rng(stable_seed("encrypt",tier,lang,L,rep,model))
    raw=[rng.choice(codebooks[model][c]) for c in plain]
    syms=sorted(set(raw))
    perm=np.arange(len(syms)); rng2=np.random.default_rng(stable_seed("opaque",tier,lang,L,rep,model)); rng2.shuffle(perm)
    code_to_id={s:int(perm[i]) for i,s in enumerate(syms)}
    seq=np.asarray([code_to_id[s] for s in raw],dtype=np.int32)
    true=np.full(len(syms),-1,dtype=np.int16)
    for s,i in code_to_id.items():
        owners=[c for c in ALPH if s in codebooks[model][c]]
        if len(owners)!=1: raise RuntimeError((model,s,owners))
        true[i]=A2I[owners[0]]
    dec="".join(ALPH[int(true[x])] for x in seq)
    if dec!=plain: raise RuntimeError("oracle decode failed")
    return seq,true,len(syms)

def score_mapping(seq,mapping,logp):
    p=mapping[seq]
    if len(p)<4:return -1e99
    idx=((p[:-3].astype(np.int64)*N+p[1:-2])*N+p[2:-1])*N+p[3:]
    return float(logp[idx].mean())

def init_mapping(nsym,cap,unig,seq,rng):
    caps=np.array([cap[c] for c in ALPH],dtype=int)
    if nsym>caps.sum(): return None
    cntsym=np.bincount(seq,minlength=nsym)
    order=np.argsort(-cntsym)
    remaining=caps.copy()
    m=np.full(nsym,-1,dtype=np.int16)
    for s in order:
        avail=np.flatnonzero(remaining>0)
        w=unig[avail]**0.8; w=w/w.sum()
        j=int(rng.choice(avail,p=w))
        m[s]=j; remaining[j]-=1
    return m

def optimize(seq,cap,unig,logp,cipher_hash_value,model,lang):
    nsym=int(seq.max())+1
    caps=np.array([cap[c] for c in ALPH],dtype=int)
    best_s=-1e99; best_m=None
    for rr in range(RESTARTS):
        rng=np.random.default_rng(stable_seed("opt",cipher_hash_value,model,lang,rr))
        m=init_mapping(nsym,cap,unig,seq,rng)
        if m is None:return None
        cnt=np.bincount(m,minlength=N).astype(int)
        s=score_mapping(seq,m,logp)
        if s>best_s:best_s,best_m=s,m.copy()
        for step in range(STEPS):
            frac=step/max(1,STEPS-1); T=.18*(1-frac)+.003
            si=int(rng.integers(nsym)); old=int(m[si]); new=int(rng.integers(N))
            if new==old: continue
            m2=m.copy()
            if cnt[new]<caps[new]:
                m2[si]=new
            else:
                cand=np.flatnonzero(m==new)
                if len(cand)==0:continue
                sj=int(rng.choice(cand)); m2[si]=new; m2[sj]=old
            s2=score_mapping(seq,m2,logp)
            if s2>=s or rng.random()<math.exp(max(-50,min(50,(s2-s)/T))):
                m,s=m2,s2
                cnt=np.bincount(m,minlength=N).astype(int)
                if s>best_s:best_s,best_m=s,m.copy()
    return best_s,best_m

def cipher_hash(seq):
    return hashlib.sha256(seq.astype("<i4").tobytes()).hexdigest()

def shuffle_seq(seq,key,j):
    rng=np.random.default_rng(stable_seed("null",key,j)); x=seq.copy(); rng.shuffle(x); return x

def evaluate_case(tier,lang,L,rep,model,plain,lms,unigs,codebooks,caps):
    oracle_scores={la:lm_score_plain(plain,lms[la]) for la in LANGS}
    oracle_rank=sorted(LANGS,key=lambda la:oracle_scores[la],reverse=True)
    seq,true_map,nsym=encrypt(plain,model,codebooks,tier,lang,L,rep)
    ch=cipher_hash(seq); scored={}; maps={}
    for la in LANGS:
        o=optimize(seq,caps[model],unigs[la],lms[la],ch,model,la)
        if o is None: raise RuntimeError(("infeasible",tier,lang,L,rep,model,nsym))
        scored[la]=o[0]; maps[la]=o[1]
    rank=sorted(LANGS,key=lambda la:scored[la],reverse=True)
    best_map=maps[rank[0]]; target_map=maps[lang]
    truth=true_map[seq]
    acc_top=float(np.mean(best_map[seq]==truth)); acc_target=float(np.mean(target_map[seq]==truth))
    map_acc_target=float(np.mean(target_map==true_map))
    null=[]
    for j in range(NULLS):
        sh=shuffle_seq(seq,ch,j); shh=cipher_hash(sh)
        so=optimize(sh,caps[model],unigs[lang],lms[lang],shh,model,lang)
        null.append(so[0])
    mu=float(np.mean(null)); sd=float(np.std(null,ddof=1)) if len(null)>1 else 0.0
    z=(scored[lang]-mu)/sd if sd>1e-12 else 0.0
    return {"tier":tier,"lang":lang,"L":L,"rep":rep,"model":model,"oracle_top":oracle_rank[0],"oracle_correct":oracle_rank[0]==lang,"blind_top":rank[0],"blind_correct":rank[0]==lang,"rank_target":rank.index(lang)+1,"char_acc_top":acc_top,"char_acc_target":acc_target,"map_acc_target":map_acc_target,"target_z":z,"nsym":nsym,"cipher_sha":ch[:16],"target_score":scored[lang],"best_score":scored[rank[0]]}

def aggregate(cases):
    out=[]
    for tier in ["P0","P1"]:
        for L in LENGTHS:
            cc=[x for x in cases if x["tier"]==tier and x["L"]==L]
            for model in MODELS:
                mm=[x for x in cc if x["model"]==model]
                per={la:sum(x["blind_correct"] for x in mm if x["lang"]==la) for la in TARGETS}
                out.append({"tier":tier,"L":L,"model":model,"n":len(mm),"oracle_correct":sum(x["oracle_correct"] for x in mm),"blind_correct":sum(x["blind_correct"] for x in mm),"per_lang":per,"median_char_acc_target":float(np.median([x["char_acc_target"] for x in mm])),"median_map_acc_target":float(np.median([x["map_acc_target"] for x in mm])),"median_z":float(np.median([x["target_z"] for x in mm])),"median_nsym":float(np.median([x["nsym"] for x in mm]))})
    return out

def gate(cases,tier):
    cc=[x for x in cases if x["tier"]==tier and x["L"]==88]
    oracle=sum(x["oracle_correct"] for x in cc); blind=sum(x["blind_correct"] for x in cc)
    per={la:sum(x["blind_correct"] for x in cc if x["lang"]==la) for la in TARGETS}
    med_acc=float(np.median([x["char_acc_target"] for x in cc])); med_z=float(np.median([x["target_z"] for x in cc]))
    q0=oracle>=14; q1=blind>=12 and all(v>=2 for v in per.values()); q2=med_acc>=.50; q3=med_z>=3.0
    return {"tier":tier,"n":len(cc),"oracle_correct":oracle,"blind_correct":blind,"per_lang":per,"median_char_acc_target":med_acc,"median_z":med_z,"Q0":q0,"Q1":q1,"Q2":q2,"Q3":q3,"pass":q0 and q1 and q2 and q3}

def main():
    lm_sents,p0,hist,meta=load_all_sources(); lms={}; unigs={}; lmmeta={}
    for la in LANGS:
        logp,unig,nc=build_lm(lm_sents[la]); lms[la]=logp; unigs[la]=unig; lmmeta[la]={"train_chars":nc,"train_sentences":len(lm_sents[la])}; print("LM",la,nc,flush=True)
    codebooks,caps=make_codebooks(); print("CODEBOOK",{"T2":sum(len(v) for v in codebooks["T2"].values()),"T3":sum(len(v) for v in codebooks["T3"].values())},flush=True)
    cases=[]
    for tier,sources in [("P0",p0),("P1",hist)]:
        for lang in TARGETS:
            text=sources[lang]; print("CONTROL",tier,lang,len(text),flush=True)
            for L in LENGTHS:
                for rep in REPS:
                    plain,start=choose_span(text,L,tier,lang,rep)
                    for model in MODELS:
                        r=evaluate_case(tier,lang,L,rep,model,plain,lms,unigs,codebooks,caps); r["span_start"]=start; cases.append(r)
                        print("CASE",tier,lang,L,rep,model,"oracle",r["oracle_top"],"blind",r["blind_top"],"rank",r["rank_target"],"acc",round(r["char_acc_target"],3),"z",round(r["target_z"],3),"nsym",r["nsym"],flush=True)
    agg=aggregate(cases); g0=gate(cases,"P0"); g1=gate(cases,"P1")
    if not g0["pass"]: verdict="NOT QUALIFIED AT VOYNICH-FOLIO SCALE"
    elif not g1["pass"]: verdict="QUALIFIED IN-DOMAIN / HISTORICALLY DOMAIN-LIMITED"
    else: verdict="QUALIFIED FOR THIS MECHANISM CLASS"
    summary={"protocol_version":"0.2","verdict":verdict,"gates":{"P0":g0,"P1":g1},"aggregate":agg,"source_meta":meta,"lm_meta":lmmeta,"params":{"lengths":LENGTHS,"reps":REPS,"models":MODELS,"steps":STEPS,"restarts":RESTARTS,"nulls":NULLS},"case_count":len(cases)}
    print("SUMMARY_JSON="+json.dumps(summary,separators=(",",":"),ensure_ascii=False),flush=True)
    print("CASES_JSON="+json.dumps(cases,separators=(",",":"),ensure_ascii=False),flush=True)

if __name__=="__main__": main()
