#!/usr/bin/env python3
"""Development-only Yiddish-vs-German evaluator separability preflight.

This is NOT L qualification. It uses only previously consumed PPCHY Yiddish
families and a historical-German transport mirror of ReF *modernised* reading
texts. It never loads the Voynich target. Its purpose is to falsify the idea
that an evaluator can separate the closest registered comparator at all before
we spend effort on fresh confirmation corpus engineering.

Frozen before outcomes on 2026-09-12.
"""
from __future__ import annotations
import hashlib, itertools, json, math, random, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

PPCHY_COMMIT = "b5864bd02a315c1d436a82553667bbf81eab6537"
REF_MIRROR_COMMIT = "255bd632e45459230eaf2a994e0557ed6c682869"
TRAIN_BUDGET_WORDS = 10000
DEV_WORDS = 512
ALPHA = 0.25
PUBLIC_SEED = "yiddish-l-evaluator-preflight-v01-20260912"

# Whole relationship groups. Every listed Yiddish family was consumed before
# this L preflight; none can become fresh confirmation evidence later.
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

# ReF metadata verified against the official detailed inventory. Build and
# development are source-disjoint. The mirror contains the ReF 1.0.1
# MODERNISED reading text, so this arm is explicitly a representation/nuisance
# preflight, not the protocol's final diplomatic German comparator.
GER_BUILD = {
    "F014_1501_1524_north_bavarian": "f014.txt",
    "F015_early16_regensburg_north_bavarian": "f015.txt",
}
GER_DEV = {
    "F016_1530_regensburg_north_bavarian": "f016.txt",
    "F018_1550_regensburg_north_bavarian": "f018.txt",
    "F034_1524_munich_middle_bavarian": "f034.txt",
    "F037_1583_munich_east_bavarian": "f037.txt",
    "F148_1557_augsburg_swabian": "f148.txt",
}

LEAF_RE = re.compile(r"\(([A-Z][A-Z0-9$=*-]*)\s+([^()\s]+)\)")

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def normalize_ascii_token(raw: str):
    raw=raw.replace('@','').split('^',1)[0]
    out=[]
    for part in raw.split('_'):
        w=''.join(c for c in part.lower() if 'a' <= c <= 'z')
        if w: out.append(w)
    return out

def penn_words(p: Path):
    out=[]
    for tag,raw in LEAF_RE.findall(p.read_text(encoding='utf-8',errors='replace')):
        if tag.startswith(('ID','CODE','PUNC')) or raw.startswith('*') or raw in {'0','-NONE-'}: continue
        out.extend(normalize_ascii_token(raw))
    return out

def ref_words(p: Path):
    s=p.read_text(encoding='utf-8',errors='replace').replace('\n',' ')
    ident=p.stem.upper()
    # Reproduce the transport mirror's own extraction boundary without using
    # its derived CSV. Remove PDF header/footer and line reference labels.
    s=re.sub(rf'{ident}:.*?Modernisierter Lesetext', ' ', s, count=1, flags=re.S)
    s=re.sub(r'Referenzkorpus Frühneuhochdeutsch 1\.0\.1\s+\(.*?\)\s+\d+', ' ', s, flags=re.S)
    s=s.split('@H')[-1]
    s=re.sub(rf'{ident}-[^\s]+', ' ', s)
    out=[]
    for tok in s.split():
        w=''.join(c for c in tok.lower() if 'a' <= c <= 'z')
        if w: out.append(w)
    return out

def family_words(root: Path, spec: dict[str,list[str]]):
    return {fam:sum((penn_words(root/f) for f in files),[]) for fam,files in spec.items()}

def work_balanced(pool: dict[str,list[str]], budget: int):
    # Deterministic round-robin gives equal family opportunity; when a family
    # exhausts, residual budget is redistributed among remaining families.
    names=sorted(pool); idx={n:0 for n in names}; out=[]; source=[]
    while len(out)<budget:
        progressed=False
        for n in names:
            i=idx[n]
            if i < len(pool[n]):
                out.append(pool[n][i]); source.append(n); idx[n]+=1; progressed=True
                if len(out)>=budget: break
        if not progressed: break
    if len(out)<budget: raise RuntimeError(f'pool exhausted at {len(out)} < {budget}')
    return out, Counter(source)

def selected_segment(words: list[str], family: str, n=DEV_WORDS):
    if len(words)<n: raise RuntimeError(f'{family}: only {len(words)} words')
    span=len(words)-n+1
    seed=int.from_bytes(hashlib.sha256(f'{PUBLIC_SEED}|{family}|segment'.encode()).digest()[:8],'big')
    start=seed % span
    return words[start:start+n], start

# Character trigram evaluator. Boundaries influence probabilities but are not
# counted in the bits-per-source-atom denominator.
def fit_trigram(words):
    ctx=defaultdict(Counter); totals=Counter(); alphabet=list('abcdefghijklmnopqrstuvwxyz')+['$']
    for w in words:
        seq=['^','^']+list(w)+['$']
        for i in range(2,len(seq)):
            c=(seq[i-2],seq[i-1]); x=seq[i]; ctx[c][x]+=1; totals[c]+=1
    return ctx,totals,alphabet

def trigram_bpa(model, words):
    ctx,totals,alphabet=model; V=len(alphabet); nll=0.0; atoms=0
    for w in words:
        seq=['^','^']+list(w)+['$']; atoms+=len(w)
        for i in range(2,len(seq)):
            c=(seq[i-2],seq[i-1]); x=seq[i]
            p=(ctx[c][x]+ALPHA)/(totals[c]+ALPHA*V)
            nll-=math.log2(p)
    return nll/atoms

def fit_unigram(words):
    c=Counter(''.join(words)); total=sum(c.values()); return c,total

def unigram_bpa(model, words):
    c,total=model; V=26; nll=0.0; atoms=0
    for w in words:
        for x in w:
            nll-=math.log2((c[x]+ALPHA)/(total+ALPHA*V)); atoms+=1
    return nll/atoms

def fit_lengths(words):
    c=Counter(min(len(w),20) for w in words); return c,sum(c.values())

def length_bpw(model, words):
    c,total=model; V=20; nll=0.0
    for w in words:
        x=min(len(w),20); nll-=math.log2((c[x]+ALPHA)/(total+ALPHA*V))
    return nll/len(words)

def shuffle_inside(words, family):
    out=[]
    for i,w in enumerate(words):
        r=random.Random(int.from_bytes(hashlib.sha256(f'{PUBLIC_SEED}|{family}|shuffle|{i}'.encode()).digest()[:8],'big'))
        a=list(w); r.shuffle(a); out.append(''.join(a))
    return out

def ngrams(words,n): return {tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1))}

def overlap_audit(build, dev):
    out=[]
    for bf,bw in build.items():
        for df,dw in dev.items():
            out.append({'build':bf,'development':df,
                        'shared_8word_types':len(ngrams(bw,8)&ngrams(dw,8)),
                        'shared_5word_types':len(ngrams(bw,5)&ngrams(dw,5))})
    return out

def exact_null(rows, key='margin_trigram'):
    # Exact work-family label permutation, conditioned on 5/5 labels and fixed
    # model outputs. One-sided: how often is accuracy >= observed?
    vals=[r[key] for r in rows]; n=len(vals); k=n//2
    true=[r['truth'] for r in rows]
    def acc(labels):
        return sum(((v>0) == (lab=='yiddish')) for v,lab in zip(vals,labels))/n
    obs=acc(true); null=[]
    for comb in itertools.combinations(range(n),k):
        S=set(comb); labels=['yiddish' if i in S else 'german' for i in range(n)]
        null.append(acc(labels))
    mu=statistics.mean(null); sd=statistics.pstdev(null)
    p=sum(x>=obs-1e-15 for x in null)/len(null)
    return {'observed_accuracy':obs,'effect_over_chance':obs-0.5,
            'null_mean':mu,'null_sd':sd,'effect_over_null_sd':(obs-mu)/sd if sd else None,
            'exact_one_sided_p':p,'null_assignments':len(null)}

def main():
    if len(sys.argv)!=4:
        raise SystemExit('usage: script PPCHY_DATA REF_TEXTS OUTDIR')
    yp=Path(sys.argv[1]); gp=Path(sys.argv[2]); out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)

    ybuild=family_words(yp,YID_BUILD); ydev=family_words(yp,YID_DEV)
    gbuild={fam:ref_words(gp/fn) for fam,fn in GER_BUILD.items()}
    gdev={fam:ref_words(gp/fn) for fam,fn in GER_DEV.items()}
    ytrain,yalloc=work_balanced(ybuild,TRAIN_BUDGET_WORDS)
    gtrain,galloc=work_balanced(gbuild,TRAIN_BUDGET_WORDS)

    models={
      'tri_y':fit_trigram(ytrain),'tri_g':fit_trigram(gtrain),
      'uni_y':fit_unigram(ytrain),'uni_g':fit_unigram(gtrain),
      'len_y':fit_lengths(ytrain),'len_g':fit_lengths(gtrain),
    }
    rows=[]
    for truth,pool in [('yiddish',ydev),('german',gdev)]:
        for fam,allw in pool.items():
            seg,start=selected_segment(allw,fam)
            shuf=shuffle_inside(seg,fam)
            ty=trigram_bpa(models['tri_y'],seg); tg=trigram_bpa(models['tri_g'],seg)
            sy=trigram_bpa(models['tri_y'],shuf); sg=trigram_bpa(models['tri_g'],shuf)
            uy=unigram_bpa(models['uni_y'],seg); ug=unigram_bpa(models['uni_g'],seg)
            ly=length_bpw(models['len_y'],seg); lg=length_bpw(models['len_g'],seg)
            rows.append({'truth':truth,'family':fam,'available_words':len(allw),'segment_start':start,
                         'segment_words':len(seg),'bpa_yiddish_trigram':ty,'bpa_german_trigram':tg,
                         'margin_trigram':tg-ty,
                         'bpa_yiddish_trigram_shuffled':sy,'bpa_german_trigram_shuffled':sg,
                         'margin_trigram_shuffled':sg-sy,
                         'bpa_yiddish_unigram':uy,'bpa_german_unigram':ug,'margin_unigram':ug-uy,
                         'bpw_yiddish_length':ly,'bpw_german_length':lg,'margin_length':lg-ly})

    summary={
      'status':'DEVELOPMENT_EVALUATOR_PREFLIGHT_ONLY__NOT_L_QUALIFICATION',
      'target_loaded':False,
      'yiddish_representation':'PPCHY lossy a-z Romanisation; all source families previously consumed',
      'german_representation':'ReF 1.0.1 modernised reading texts via pinned transport mirror; NOT diplomatic',
      'ppchy_commit_expected':PPCHY_COMMIT,'ref_mirror_commit_expected':REF_MIRROR_COMMIT,
      'train_budget_words_per_language':TRAIN_BUDGET_WORDS,
      'yiddish_train_allocation':dict(yalloc),'german_train_allocation':dict(galloc),
      'development_work_families_per_language':5,
      'primary':exact_null(rows,'margin_trigram'),
      'nuisance_unigram':exact_null(rows,'margin_unigram'),
      'nuisance_length_only':exact_null(rows,'margin_length'),
      'nuisance_within_word_shuffle':exact_null(rows,'margin_trigram_shuffled'),
      'yiddish_overlap_audit':overlap_audit(ybuild,ydev),
      'german_overlap_audit':overlap_audit(gbuild,gdev),
      'source_hashes':{},
      'notes':[
        'Work family is the inferential unit; 512-word segments are not treated as independent works.',
        'Exact permutation null enumerates all 252 balanced 5-vs-5 work-label assignments.',
        'A positive here only establishes evaluator separability in a secondary representation.',
        'If unigram/length/shuffle nuisance arms perform similarly to trigram, editorial/orthographic confounding remains unresolved.',
        'No raw-score threshold, abstention rule, Hebrew comparator, fresh Yiddish confirmation, or target transfer is fitted here.'
      ]
    }
    for fam,files in {**YID_BUILD,**YID_DEV}.items():
        summary['source_hashes'][fam]=[{f:sha_file(yp/f)} for f in files]
    for fam,fn in {**GER_BUILD,**GER_DEV}.items(): summary['source_hashes'][fam]={fn:sha_file(gp/fn)}

    (out/'rows.json').write_text(json.dumps(rows,indent=2,sort_keys=True))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    # Compact human-readable audit.
    lines=['# Yiddish-German evaluator separability preflight','',f"Status: {summary['status']}",'',
           'This is development-only and does not qualify L. The Voynich target was not loaded.','',
           '## Headline']
    for k in ['primary','nuisance_unigram','nuisance_length_only','nuisance_within_word_shuffle']:
        x=summary[k]; lines.append(f"- {k}: accuracy={x['observed_accuracy']:.3f}; effect over chance={x['effect_over_chance']:+.3f}; exact balanced-label null SD={x['null_sd']:.6f}; z={x['effect_over_null_sd']:.3f}; exact one-sided p={x['exact_one_sided_p']:.6f} ({x['null_assignments']} assignments).")
    lines+=['','## Per-work primary margins','', '| truth | family | German BPA - Yiddish BPA | call |','|---|---|---:|---|']
    for r in rows:
        call='yiddish' if r['margin_trigram']>0 else 'german'
        lines.append(f"| {r['truth']} | {r['family']} | {r['margin_trigram']:+.6f} | {call} |")
    lines+=['','## Interpretation guardrails','',
            '- German input is historical ReF text but the available mirror is the modernised reading-text representation. It is therefore not the final registered diplomatic comparator.',
            '- Yiddish is Penn Romanisation, also secondary/lossy. A result may be representation-dependent.',
            '- Similar success of nuisance arms is evidence that the instrument may be using shallow editorial/orthographic cues rather than sequence structure.',
            '- No L certificate and no Voynich inference can be issued from this run.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,sort_keys=True))

if __name__=='__main__': main()
