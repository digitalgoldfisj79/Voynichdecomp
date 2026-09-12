#!/usr/bin/env python3
"""Historical Yiddish-German L recovery competition v0.1.

Development only. Voynich is never loaded. Candidate-1/2 direct classifier family
is not extended. This experiment asks whether equally budgeted historical-language
models discriminate through blind M0 key recovery and fit->audit transfer.

Modes:
  preflight: acquire/verify source representations, freeze opaque encrypted cases,
             write separate PUBLIC and PRIVATE artifacts before solver outcomes.
  solve:     receives PUBLIC artifact only; runs both anonymous models on one opaque
             relationship group. No plaintext, true key, family name or truth language.
  aggregate: receives frozen solver outputs plus PRIVATE truth; scores recovery and
             the preregistered deployable language rule, nuisance arms and exact nulls.
"""
from __future__ import annotations
import argparse, hashlib, hmac, itertools, json, math, os, pickle, random, re, statistics, sys, tempfile, time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "yiddish_qualification_v1b"
sys.path.insert(0, str(PARENT))
import finite_panel_r_v02 as base  # noqa: E402
from encoder_v02 import encode_words  # noqa: E402
from independent_decoder_v02 import invert_permutation, decode_words, assert_roundtrip_non_erased  # noqa: E402

VERSION = "yiddish_german_l_recovery_v01_20260912"
PPCHY_COMMIT = "b5864bd02a315c1d436a82553667bbf81eab6537"
TRAIN_BUDGET = 10000
FIT_N = 512
BUFFER = 32
AUDIT_N = 512
N_KEYS = 32
NULL_MAPS = 64
ALPHA = 0.25
CALL_Z = 1.0
ATOM_PASS = 0.90
WORD_PASS = 0.80
CELL_PASS = 29
SOLVER_CFG = {"id":"S1_INHERITED_R_V03","restarts":8,"steps":5000,"greedy_passes":60}
GROUP_ALIASES = tuple(f"G{i:02d}" for i in range(10))

YID_BUILD = {
    "shir_1579": ["1579e-shir-preface.psd", "1579e-shir.psd"],
    "ester_1589": ["1589e-ester-preface.psd", "1589e-ester.psd"],
}
YID_DEV = {
    "bovo_1507": ["1507w-bovo.psd"],
    "cracow_letters_1588": ["1588e-letters-cracow.psd"],
    "sam_hayyim_1590": ["1590e-sam-hayyim.psd"],
    "lev_tov_1620": ["1620e-lev-tov-1-preface.psd", "1620e-lev-tov-1.psd"],
    "kine_1648": ["1648w-kine.psd"],
}
GER_BUILD = ("F014", "F015")
GER_DEV = ("F016", "F018", "F034", "F037", "F148")
LEAF_RE = re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)")


def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def pseed(*parts) -> int:
    return int.from_bytes(hashlib.sha256('|'.join([VERSION,*map(str,parts)]).encode()).digest()[:8],'big')
def sseed(root: bytes,*parts) -> int:
    return int.from_bytes(hmac.new(root,'|'.join(map(str,parts)).encode(),hashlib.sha256).digest()[:8],'big')
def atomic_pickle(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def atomic_json(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=str(path.parent),text=True)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def append_jsonl(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(obj,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())


def normalize_leaf(raw):
    if raw.startswith('*') or raw in {'0','-NONE-'}: return []
    raw=raw.replace('@','').split('^',1)[0]; out=[]
    for part in raw.split('_'):
        w=''.join(c for c in part.lower() if 'a'<=c<='z')
        if w: out.append(w)
    return out

def penn_words(p:Path):
    out=[]
    for tag,raw in LEAF_RE.findall(p.read_text(encoding='utf-8',errors='replace')):
        if tag.startswith(('ID','CODE','PUNC')): continue
        out.extend(normalize_leaf(raw))
    return out

def penn_family(data:Path,spec):
    return {fam:sum((penn_words(data/f) for f in files),[]) for fam,files in spec.items()}

def normalize_latin(s):
    w=''.join(c for c in s.lower() if 'a'<=c<='z'); return w or None

def ref_words(root:Path,wid:str):
    cand=[p for p in root.rglob('*.xml') if wid.lower() in p.name.lower() or wid.lower() in str(p.parent).lower()]
    parsed=[]
    for p in cand:
        try:
            tree=ET.parse(p).getroot(); out=[]
            for el in tree.iter():
                if el.tag.rsplit('}',1)[-1]=='tok_dipl':
                    w=normalize_latin(el.attrib.get('utf') or (el.text or ''))
                    if w: out.append(w)
            if out: parsed.append((len(out),p,out))
        except Exception:
            pass
    if not parsed: raise RuntimeError(f'ReF diplomatic source {wid} not found')
    parsed.sort(key=lambda x:(x[0],str(x[1])),reverse=True)
    n,p,out=parsed[0]
    return out,p

def work_balanced(pool:dict[str,list[str]],budget:int):
    names=sorted(pool); idx={n:0 for n in names}; out=[]; src=[]
    while len(out)<budget:
        progressed=False
        for n in names:
            i=idx[n]
            if i<len(pool[n]):
                out.append(pool[n][i]); src.append(n); idx[n]+=1; progressed=True
                if len(out)>=budget: break
        if not progressed: break
    if len(out)!=budget: raise RuntimeError(f'balanced pool exhausted at {len(out)}')
    return out,dict(Counter(src))
def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}
def overlap_rows(build,dev):
    rows=[]
    for bn,bw in build.items():
        b8=ngrams(bw,8); b5=ngrams(bw,5)
        for dn,dw in dev.items():
            rows.append({'build':bn,'development':dn,'shared_8gram_types':len(b8&ngrams(dw,8)),'shared_5gram_types':len(b5&ngrams(dw,5))})
    return rows

def choose_span(words,family):
    need=FIT_N+BUFFER+AUDIT_N
    if len(words)<need: raise RuntimeError(f'{family}: {len(words)} < {need}')
    start=pseed('span',family)%(len(words)-need+1)
    return words[start:start+FIT_N],words[start+FIT_N+BUFFER:start+need],start

def shuffle_words(words,family,side):
    out=[]
    for i,w in enumerate(words):
        a=list(w); random.Random(pseed('within_word_shuffle',family,side,i)).shuffle(a); out.append(''.join(a))
    return out


def make_key(root,*parts):
    r=random.Random(sseed(root,'key',*parts)); p=list(range(base.A)); r.shuffle(p); return p

def encrypt(words,key): return encode_words(words,key,0.0,random.Random(0))[0]


def blind_solve(fit_cipher,logp,train_uni,search_seed):
    C,cuni=base.cipher_counts(fit_cipher); initial=base.frequency_initial(cuni,train_uni)
    rng=random.Random(search_seed); best=None; bs=-1e100
    for r in range(SOLVER_CFG['restarts']):
        m=initial.copy(); rr=random.Random(pseed('restart',search_seed,r))
        for _ in range(8*r): a,b=rr.sample(range(base.A),2); m[a],m[b]=m[b],m[a]
        s=base.mapping_score(C,logp,m)
        if s>bs: bs=s; best=m.copy()
        for step in range(SOLVER_CFG['steps']):
            a,b=rng.sample(range(base.A),2); d=base.swap_delta(C,logp,m,a,b)
            frac=step/max(1,SOLVER_CFG['steps']-1); temp=.006*(1-frac)+.00003
            if d>=0 or rng.random()<math.exp(max(-50.0,d/temp)):
                m[a],m[b]=m[b],m[a]; s+=d
                if s>bs: bs=s; best=m.copy()
        m=best.copy(); s=bs
        for _ in range(SOLVER_CFG['greedy_passes']):
            bd=0.0; bp=None
            for a in range(base.A):
                for b in range(a+1,base.A):
                    d=base.swap_delta(C,logp,m,a,b)
                    if d>bd+1e-12: bd=d; bp=(a,b)
            if bp is None: break
            a,b=bp; m[a],m[b]=m[b],m[a]; s+=bd
            if s>bs: bs=s; best=m.copy()
    return best,bs

def random_maps(tag,group,key_index):
    maps=[]
    for i in range(NULL_MAPS):
        r=random.Random(pseed('random_map',tag,group,key_index,i)); m=list(range(base.A)); r.shuffle(m); maps.append(m)
    return maps

def z_bigram(audit_cipher,mapping,logp,maps):
    C,_=base.cipher_counts(audit_cipher); x=base.mapping_score(C,logp,mapping)
    null=[base.mapping_score(C,logp,m) for m in maps]; mu=statistics.mean(null); sd=statistics.pstdev(null)
    return {'score':x,'null_mean':mu,'null_sd':sd,'z':(x-mu)/sd if sd>0 and math.isfinite(sd) else None}
def fit_unigram(words):
    c=Counter(''.join(words)); total=sum(c.values()); return [c.get(chr(97+i),0)+ALPHA for i in range(base.A)],total+ALPHA*base.A
def z_unigram(audit_cipher,mapping,umodel,maps):
    probs,total=umodel
    def score(m):
        n=0; s=0.0
        for w in audit_cipher:
            for c in w:
                if c=='~': continue
                ci=ord(c)-97; pi=m[ci]; s+=math.log(probs[pi]/total); n+=1
        return s/n if n else float('nan')
    x=score(mapping); null=[score(m) for m in maps]; mu=statistics.mean(null); sd=statistics.pstdev(null)
    return {'score':x,'null_mean':mu,'null_sd':sd,'z':(x-mu)/sd if sd>0 and math.isfinite(sd) else None}

def call_from_margin(m,threshold=CALL_Z):
    if m is None or not math.isfinite(m): return 'abstain'
    if m>=threshold: return 'A'
    if m<=-threshold: return 'B'
    return 'abstain'


def preflight(args):
    out=Path(args.out); public=out/'public'; private=out/'private'; public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
    ydata=Path(args.penn_data); ref=Path(args.ref_root)
    ybuild=penn_family(ydata,YID_BUILD); ydev=penn_family(ydata,YID_DEV)
    gpool={}; gpaths={}
    for wid in GER_BUILD+GER_DEV:
        w,p=ref_words(ref,wid); gpool[wid]=w; gpaths[wid]=str(p)
    gbuild={k:gpool[k] for k in GER_BUILD}; gdev={k:gpool[k] for k in GER_DEV}
    ytrain,yalloc=work_balanced(ybuild,TRAIN_BUDGET); gtrain,galloc=work_balanced(gbuild,TRAIN_BUDGET)
    yo=overlap_rows(ybuild,ydev); go=overlap_rows(gbuild,gdev)
    if any(r['shared_8gram_types'] for r in yo+go): raise RuntimeError('BUILD-DEVELOPMENT exact 8-word overlap gate failed')
    for fam,ws in {**ydev,**gdev}.items():
        if len(ws)<FIT_N+BUFFER+AUDIT_N: raise RuntimeError(f'{fam} quantity gate failed')

    root=os.urandom(32)
    # Anonymous model identity and relationship-group aliases are frozen before scoring.
    model_langs=['yiddish','german']; random.Random(sseed(root,'model_alias')).shuffle(model_langs)
    model_truth={'A':model_langs[0],'B':model_langs[1]}
    allgroups=[('yiddish',k,ydev[k]) for k in sorted(ydev)]+[('german',k,gdev[k]) for k in sorted(gdev)]
    random.Random(sseed(root,'group_alias')).shuffle(allgroups)
    group_truth={alias:{'language':lang,'family':fam} for alias,(lang,fam,_) in zip(GROUP_ALIASES,allgroups)}

    lang_train={'yiddish':ytrain,'german':gtrain}
    public_models={}
    for mid,lang in model_truth.items():
        lp,uni=base.lm_from_words(lang_train[lang])
        public_models[mid]={'logp':lp,'train_uni':uni,'unigram':fit_unigram(lang_train[lang])}
    atomic_pickle(public_models,public/'models.pkl')

    private_truth={'version':VERSION,'model_truth':model_truth,'group_truth':group_truth,'plant_root_hex':root.hex(),'cases':{}}
    for alias,(lang,fam,words) in zip(GROUP_ALIASES,allgroups):
        fit,audit,start=choose_span(words,fam); n2fit=shuffle_words(fit,fam,'fit'); n2audit=shuffle_words(audit,fam,'audit')
        cases=[]; priv=[]
        for ki in range(N_KEYS):
            key=make_key(root,'plant',alias,ki); oracle=invert_permutation(key)
            fc=encrypt(fit,key); ac=encrypt(audit,key); sfc=encrypt(n2fit,key); sac=encrypt(n2audit,key)
            assert_roundtrip_non_erased(fit,decode_words(fc,oracle)); assert_roundtrip_non_erased(audit,decode_words(ac,oracle))
            assert_roundtrip_non_erased(n2fit,decode_words(sfc,oracle)); assert_roundtrip_non_erased(n2audit,decode_words(sac,oracle))
            cases.append({'group':alias,'key_index':ki,'fit_cipher':fc,'audit_cipher':ac,'n2_fit_cipher':sfc,'n2_audit_cipher':sac})
            priv.append({'key_index':ki,'fit_truth':fit,'audit_truth':audit,'n2_fit_truth':n2fit,'n2_audit_truth':n2audit,'oracle':oracle})
        atomic_json({'version':VERSION,'group':alias,'cases':cases},public/f'{alias}.json')
        private_truth['cases'][alias]={'segment_start':start,'rows':priv}

    source_hashes={'penn':{},'ref':{}}
    for fam,files in {**YID_BUILD,**YID_DEV}.items(): source_hashes['penn'][fam]=[{f:sha_file(ydata/f)} for f in files]
    for wid,pstr in gpaths.items(): source_hashes['ref'][wid]={'path':pstr,'sha256':sha_file(Path(pstr)),'words':len(gpool[wid])}
    manifest={
      'version':VERSION,'status':'PREFLIGHT_FROZEN_BEFORE_SOLVER_OUTCOMES','target_loaded':False,'voynich_access_allowed':False,
      'representation':'secondary normalized a-z only; primary bridge failed separately','ppchy_commit_expected':PPCHY_COMMIT,
      'build_budget_per_language':TRAIN_BUDGET,'yiddish_allocation':yalloc,'german_allocation':galloc,
      'development_counts':{'yiddish':{k:len(v) for k,v in ydev.items()},'german':{k:len(v) for k,v in gdev.items()}},
      'overlap_yiddish':yo,'overlap_german':go,'source_hashes':source_hashes,
      'plant_root_commitment_sha256':sha_bytes(root),'model_and_group_aliases_private_until_scoring':True,
      'solver_cfg':SOLVER_CFG,'keys_per_group':N_KEYS,'random_maps_per_model_case':NULL_MAPS,'call_z_threshold':CALL_Z,
      'protocol_sha256':sha_file(HERE/'PROTOCOL.md'),'runner_sha256':sha_file(Path(__file__))
    }
    atomic_json(manifest,public/'preflight_manifest.json'); atomic_pickle(private_truth,private/'truth.pkl')
    atomic_json({'plant_root_commitment_sha256':sha_bytes(root),'private_truth_sha256':sha_file(private/'truth.pkl')},private/'private_manifest.json')
    print(json.dumps({'status':'PREFLIGHT_OK','commitment':sha_bytes(root),'groups':len(allgroups)},indent=2))


def solve(args):
    pub=Path(args.public); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); group=args.group
    models=pickle.loads((pub/'models.pkl').read_bytes()); cases=json.loads((pub/f'{group}.json').read_text())['cases']
    rows=[]
    for case in cases:
        ki=case['key_index']; row={'group':group,'key_index':ki,'primary':{},'n1':{},'n2':{}}
        for arm,fc,ac in [('primary',case['fit_cipher'],case['audit_cipher']),('n2',case['n2_fit_cipher'],case['n2_audit_cipher'])]:
            maps=random_maps(arm,group,ki)
            zvals={}
            for mid in ('A','B'):
                t0=time.time(); returned,obj=blind_solve(fc,models[mid]['logp'],models[mid]['train_uni'],pseed('search',arm,group,ki))
                z=z_bigram(ac,returned,models[mid]['logp'],maps); z.update({'mapping':returned,'fit_objective':obj,'runtime_s':time.time()-t0})
                row[arm][mid]=z; zvals[mid]=z['z']
            margin=None if zvals['A'] is None or zvals['B'] is None else zvals['A']-zvals['B']
            row[arm]['margin_A_minus_B']=margin; row[arm]['call']=call_from_margin(margin)
        maps=random_maps('n1',group,ki); zvals={}
        _,cuni=base.cipher_counts(case['fit_cipher'])
        for mid in ('A','B'):
            m=base.frequency_initial(cuni,models[mid]['train_uni']); z=z_unigram(case['audit_cipher'],m,models[mid]['unigram'],maps); z['mapping']=m
            row['n1'][mid]=z; zvals[mid]=z['z']
        margin=None if zvals['A'] is None or zvals['B'] is None else zvals['A']-zvals['B']
        row['n1']['margin_A_minus_B']=margin; row['n1']['call']=call_from_margin(margin)
        rows.append(row); append_jsonl(row,out/'rows.jsonl')
    atomic_pickle({'version':VERSION,'group':group,'rows':rows},out/'checkpoint.pkl')
    atomic_json({'group':group,'rows':len(rows),'status':'SOLVER_COMPLETE_NO_TRUTH_ACCESSED'},out/'summary.json')


def rec_score(truth,cipher,mapping): return base.score_recovery(truth,decode_words(cipher,mapping))
def model_call_to_lang(call,model_truth): return model_truth.get(call,'abstain') if call!='abstain' else 'abstain'
def family_call(margins,model_truth,threshold=CALL_Z):
    med=statistics.median(margins); c=call_from_margin(med,threshold); return med,model_call_to_lang(c,model_truth)
def accuracy(fams): return sum(r['call']==r['truth'] for r in fams)/len(fams) if fams else float('nan')
def exact_label_null(fams,threshold=CALL_Z):
    vals=[r['median_margin'] for r in fams]; labels=[r['truth'] for r in fams]; ny=sum(x=='yiddish' for x in labels); n=len(labels)
    def acc(labs):
        ok=0
        for v,lab in zip(vals,labs):
            c='A' if v>=threshold else ('B' if v<=-threshold else 'abstain')
            # caller converts A/B externally before this function; values are oriented so + means Yiddish.
            pred='yiddish' if c=='A' else ('german' if c=='B' else 'abstain'); ok+=pred==lab
        return ok/n
    # values passed here are oriented Yiddish-minus-German z margins.
    obs=acc(labels); null=[]
    for comb in itertools.combinations(range(n),ny):
        S=set(comb); labs=['yiddish' if i in S else 'german' for i in range(n)]; null.append(acc(labs))
    mu=statistics.mean(null); sd=statistics.pstdev(null); p=sum(x>=obs-1e-15 for x in null)/len(null)
    return {'accuracy':obs,'effect_over_null_mean':obs-mu,'null_mean':mu,'null_sd':sd,'effect_over_null_sd':(obs-mu)/sd if sd else None,'exact_one_sided_p':p,'assignments':len(null)}


def aggregate(args):
    pub=Path(args.public); priv=Path(args.private); sol=Path(args.solutions); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    truth=pickle.loads((priv/'truth.pkl').read_bytes()); model_truth=truth['model_truth']; group_truth=truth['group_truth']
    # orient anonymous A-B margins so positive always means Yiddish model advantage.
    orient=1 if model_truth['A']=='yiddish' else -1
    allrows=[]
    for group in GROUP_ALIASES:
        p=sol/group/'rows.jsonl'
        if not p.exists():
            # merge-multiple artifacts may be flat with unique renamed directories absent.
            cand=list(sol.rglob(f'{group}/rows.jsonl'))
            if cand: p=cand[0]
        if not p.exists(): raise RuntimeError(f'missing solver rows for {group}')
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if len(rows)!=N_KEYS: raise RuntimeError(f'{group}: incomplete rows {len(rows)}')
        casepub={x['key_index']:x for x in json.loads((pub/f'{group}.json').read_text())['cases']}
        privrows={x['key_index']:x for x in truth['cases'][group]['rows']}
        for r in rows:
            ki=r['key_index']; pr=privrows[ki]; pc=casepub[ki]
            lang=group_truth[group]['language']; correct_mid=next(k for k,v in model_truth.items() if v==lang); wrong_mid='B' if correct_mid=='A' else 'A'
            for arm,cipherkey,truthkey in [('primary','audit_cipher','audit_truth'),('n2','n2_audit_cipher','n2_audit_truth')]:
                for mid in ('A','B'):
                    rr=rec_score(pr[truthkey],pc[cipherkey],r[arm][mid]['mapping']); r[arm][mid].update(rr); r[arm][mid]['m0_pass']=rr['atom_recovery']>=ATOM_PASS and rr['word_recovery']>=WORD_PASS
                r[arm]['oriented_margin_y_minus_g']=orient*r[arm]['margin_A_minus_B']
            r['n1']['oriented_margin_y_minus_g']=orient*r['n1']['margin_A_minus_B']
            r['truth_language']=lang; r['family']=group_truth[group]['family']; r['correct_model']=correct_mid; r['wrong_model']=wrong_mid
            allrows.append(r)
    atomic_pickle({'version':VERSION,'rows':allrows,'truth':truth},out/'checkpoint.pkl')
    with (out/'scored_rows.jsonl').open('w') as f:
        for r in allrows: f.write(json.dumps(r,sort_keys=True)+'\n')

    fams=[]
    for group in GROUP_ALIASES:
        rs=[r for r in allrows if r['group']==group]; lang=rs[0]['truth_language']; family=rs[0]['family']; cm=rs[0]['correct_model']; wm=rs[0]['wrong_model']
        fr={'group':group,'truth':lang,'family':family}
        for arm in ('primary','n1','n2'):
            margins=[r[arm]['oriented_margin_y_minus_g'] for r in rs]
            med=statistics.median(margins); call='yiddish' if med>=CALL_Z else ('german' if med<=-CALL_Z else 'abstain')
            fr[arm]={'median_margin':med,'call':call,'correct':call==lang}
        fr['primary']['correct_recovery_passes']=sum(r['primary'][cm]['m0_pass'] for r in rs)
        fr['primary']['wrong_recovery_passes']=sum(r['primary'][wm]['m0_pass'] for r in rs)
        fr['primary']['mean_correct_atom']=statistics.mean(r['primary'][cm]['atom_recovery'] for r in rs)
        fr['primary']['mean_wrong_atom']=statistics.mean(r['primary'][wm]['atom_recovery'] for r in rs)
        fams.append(fr)

    def arm_summary(arm,threshold=CALL_Z):
        ff=[]
        for x in fams:
            m=x[arm]['median_margin']; call='yiddish' if m>=threshold else ('german' if m<=-threshold else 'abstain')
            ff.append({'truth':x['truth'],'family':x['family'],'median_margin':m,'call':call})
        return {'accuracy':accuracy(ff),'errors':sum(r['call']!=r['truth'] for r in ff),'abstentions':sum(r['call']=='abstain' for r in ff),'families':ff,'null':exact_label_null(ff,threshold)}
    primary=arm_summary('primary'); n1=arm_summary('n1'); n2=arm_summary('n2')
    gates={
      'correct_language_recovery':all(x['primary']['correct_recovery_passes']>=CELL_PASS for x in fams),
      'wrong_language_selectivity':all(x['primary']['wrong_recovery_passes']<CELL_PASS for x in fams),
      'deployable_accuracy':primary['accuracy']>=0.80,
      'matched_null_2sd':primary['null']['effect_over_null_sd'] is not None and primary['null']['effect_over_null_sd']>=2.0,
      'beats_unigram_by_2_errors':primary['errors']+2<=n1['errors'],
      'beats_shuffle_by_2_errors':primary['errors']+2<=n2['errors'],
      'complete_non_degenerate':all(r['primary']['A']['z'] is not None and r['primary']['B']['z'] is not None for r in allrows),
    }
    # Leave-one-family-out: recompute the two controlling family-call gates on each 9-family subset.
    loo=[]
    for drop in range(len(fams)):
        subset=[x for i,x in enumerate(fams) if i!=drop]
        ff=[{'truth':x['truth'],'family':x['family'],'median_margin':x['primary']['median_margin'],'call':x['primary']['call']} for x in subset]
        nstat=exact_label_null(ff,CALL_Z); acc=accuracy(ff); ok=acc>=0.80 and nstat['effect_over_null_sd'] is not None and nstat['effect_over_null_sd']>=2.0
        loo.append({'dropped':fams[drop]['family'],'accuracy':acc,'null':nstat,'primary_call_gates_pass':ok})
    gates['leave_one_family_out_stable']=all(x['primary_call_gates_pass'] for x in loo) if gates['deployable_accuracy'] and gates['matched_null_2sd'] else False
    pass_all=all(gates.values()); status='L_RECOVERY_DEVELOPMENT_FEASIBLE__CONFIRMATION_REQUIRED' if pass_all else 'L_RECOVERY_DEVELOPMENT_NOT_RESOLVED'
    sensitivity={str(t):arm_summary('primary',t) for t in (0.5,1.0,1.5)}
    pre=json.loads((pub/'preflight_manifest.json').read_text())
    summary={'status':status,'gates':gates,'primary':primary,'nuisance_unigram':n1,'nuisance_within_word_shuffle':n2,'families':fams,'leave_one_out':loo,'sensitivity':sensitivity,
             'model_truth_revealed_after_solver_outputs':model_truth,'group_truth_revealed_after_solver_outputs':group_truth,
             'plant_root_commitment_sha256':pre['plant_root_commitment_sha256'],'plant_root_reveal_hex':truth['plant_root_hex'],
             'preflight_manifest_sha256':sha_file(pub/'preflight_manifest.json'),'protocol_sha256':pre['protocol_sha256'],'runner_sha256':pre['runner_sha256'],
             'scope':'DEVELOPMENT_ONLY_SECONDARY_NORMALIZED_REPRESENTATION__NO_HEBREW__NO_TARGET','voynich_loaded':False}
    atomic_json(summary,out/'summary.json')

    lines=['# Yiddish–German L recovery competition v0.1 — development result','', '## RETRACTED FINDINGS','', '- None at initial result write. Any later contradiction must be inserted here with equal prominence.','',f'**Status: `{status}`**','',
           'This is development-only on the secondary normalized representation. Voynich was never loaded. Primary-script qualification remains failed.','',
           '## Frozen-gate outcome','']
    for k,v in gates.items(): lines.append(f'- {k}: **{"PASS" if v else "FAIL"}**')
    ns=primary['null']; lines += ['', '## Primary headline','', f"Primary family accuracy = {primary['accuracy']:.3f}; effect over exact balanced-label null = {ns['effect_over_null_mean']:+.3f}, null SD = {ns['null_sd']:.6f}, effect/nullSD = {ns['effect_over_null_sd'] if ns['effect_over_null_sd'] is not None else 'NA'}; exact p = {ns['exact_one_sided_p']:.6f}.", '',
             f"Unigram nuisance accuracy = {n1['accuracy']:.3f}; within-word-shuffle nuisance accuracy = {n2['accuracy']:.3f}.", '', '## Family results','', '| truth | family | primary median z-margin | call | correct M0 passes | wrong-model M0 passes | unigram call | shuffle call |','|---|---|---:|---|---:|---:|---|---|']
    for x in sorted(fams,key=lambda z:(z['truth'],z['family'])):
        lines.append(f"| {x['truth']} | {x['family']} | {x['primary']['median_margin']:+.4f} | {x['primary']['call']} | {x['primary']['correct_recovery_passes']}/32 | {x['primary']['wrong_recovery_passes']}/32 | {x['n1']['call']} | {x['n2']['call']} |")
    lines += ['', '## Interpretation boundary','', 'A development pass would only justify engineering a fresh confirmation stage. It would not issue L. A failure leaves L unresolved. No result in this file licenses target/Voynich scoring.']
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':status,'gates':gates,'primary_accuracy':primary['accuracy'],'primary_null':primary['null'],'n1_accuracy':n1['accuracy'],'n2_accuracy':n2['accuracy']},indent=2))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('preflight'); p.add_argument('--penn-data',required=True); p.add_argument('--ref-root',required=True); p.add_argument('--out',required=True)
    s=sub.add_parser('solve'); s.add_argument('--public',required=True); s.add_argument('--group',choices=GROUP_ALIASES,required=True); s.add_argument('--out',required=True)
    a=sub.add_parser('aggregate'); a.add_argument('--public',required=True); a.add_argument('--private',required=True); a.add_argument('--solutions',required=True); a.add_argument('--out',required=True)
    args=ap.parse_args(); {'preflight':preflight,'solve':solve,'aggregate':aggregate}[args.mode](args)
if __name__=='__main__': main()
