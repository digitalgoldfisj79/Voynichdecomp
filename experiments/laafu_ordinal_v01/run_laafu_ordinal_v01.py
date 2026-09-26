#!/usr/bin/env python3
"""Representation-independent LAAFU ordinal-position test v0.1.

Primary question: after controlling for raw local token context and nuisance metadata,
does absolute distance from a physical line edge still predict current token form?
No PGCS or other morphological decomposition is used.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, math, pickle, random, re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold

SEED = 20260821
MIN_LINE_LEN = 10
MAX_ORD = 6
RIGHT_GUARD = 3
DEFAULT_NULL_REPS = 30
HASH_DIM = 2**16
ALPHA = 1e-4
EPS = 1e-12

@dataclass
class LineRec:
    line_id: str
    folio: str
    locus_code: str
    quire: str
    currier: str
    section: str
    para_start: bool
    tokens: list[str]


def save_ckpt(outdir: Path, stage: str, payload):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"checkpoint_{stage}.pkl"
    with p.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_section_map(path: Path):
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding='utf-8'))
    return obj.get('mapping', obj)


def clean_tokens(text: str) -> list[str]:
    text = re.sub(r'<!.*?>', '', text)
    for tag in ('<%>', '<$>', '<->'):
        text = text.replace(tag, '')
    text = re.sub(r'<[^>]*>', '', text)
    parts = re.split(r'[\s\.,]+', text.strip())
    out = []
    for x in parts:
        if not x:
            continue
        if any(c in x for c in "[]{}?@'/:;0123456789"):
            continue
        x = re.sub(r'[^a-z]', '', x.lower())
        if x and re.fullmatch(r'[a-z]+', x):
            out.append(x)
    return out


def parse_ivtff(path: Path, section_map: dict):
    cur_meta = {'folio': None, 'quire': 'UNK', 'currier': 'UNK'}
    lines = []
    page_count = raw_line_count = kept_paragraph_lines = 0
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if raw.startswith('#') or not raw.startswith('<'):
            continue
        mh = re.match(r'^<([^>]+)>\s*<!\s*(.*?)>\s*$', raw)
        if mh and '.' not in mh.group(1):
            folio = mh.group(1).strip(); meta = mh.group(2)
            q = re.search(r'\$Q=([^\s>]+)', meta); l = re.search(r'\$L=([^\s>]+)', meta)
            cur_meta = {'folio': folio, 'quire': q.group(1) if q else 'UNK', 'currier': l.group(1) if l else 'UNK'}
            page_count += 1; continue
        m = re.match(r'^<([^>]+)>\s*(.*)$', raw)
        if not m: continue
        locus, text = m.group(1), m.group(2)
        if ',' not in locus or '.' not in locus: continue
        raw_line_count += 1
        left, code = locus.rsplit(',', 1); folio = left.split('.', 1)[0]
        if 'P' not in code: continue
        toks = clean_tokens(text)
        if len(toks) < 2: continue
        kept_paragraph_lines += 1
        lines.append(LineRec(left, folio, code, cur_meta['quire'], cur_meta['currier'],
                             section_map.get(folio, 'UNK'),
                             code.startswith('@P') or code.startswith('*P'), toks))
    audit = {
        'pages_seen': page_count, 'raw_loci_seen': raw_line_count,
        'paragraph_lines_kept': kept_paragraph_lines,
        'tokens_kept': sum(len(x.tokens) for x in lines),
        'folios': len(set(x.folio for x in lines)),
        'quires': sorted(set(x.quire for x in lines)),
        'currier_counts': collections.Counter(x.currier for x in lines),
        'section_counts': collections.Counter(x.section for x in lines),
    }
    return lines, audit


def tok_shape(t: str):
    return [f"len={min(len(t),12)}", f"f1={t[0]}", f"l1={t[-1]}", f"f2={t[:2]}", f"l2={t[-2:]}"]


def add_tok_feats(fs, prefix, t):
    if not t:
        fs.append(f"{prefix}=BND"); return
    fs.append(f"{prefix}.tok={t}")
    for z in tok_shape(t): fs.append(f"{prefix}.{z}")


def meta_feats(line, n_total, anchor, reverse=False):
    return [f"section={line.section}", f"currier={line.currier}", f"n={min(n_total,25)}",
            f"para={int(line.para_start)}", f"orient={'R' if reverse else 'L'}",
            f"anchor_remaining={min(n_total-anchor,25)}"]


def make_edge_events(lines, reverse=False, anchor_mode='observed', rng=None):
    rng = rng or random.Random(SEED); evs=[]; eligible_lines=0
    for ln in lines:
        seq = list(reversed(ln.tokens)) if reverse else ln.tokens; n=len(seq)
        if n < MIN_LINE_LEN: continue
        max_anchor = n - (MAX_ORD + RIGHT_GUARD)
        if max_anchor < 1: continue
        anchor = 0 if anchor_mode == 'observed' else rng.randint(1, max_anchor)
        eligible_lines += 1; seed_tok=seq[anchor]
        for k in range(2, MAX_ORD+1):
            idx=anchor+k-1; cur=seq[idx]
            base=meta_feats(ln,n,anchor,reverse); add_tok_feats(base,'seed',seed_tok)
            for lag in (1,2,3):
                j=idx-lag; add_tok_feats(base,f"p{lag}",seq[j] if j>=anchor else None)
            if idx-2>=anchor: base.append(f"trans2={seq[idx-2]}>{seq[idx-1]}")
            evs.append({'line_id':ln.line_id,'folio':ln.folio,'quire':ln.quire,'currier':ln.currier,
                        'section':ln.section,'k':k,'token':cur,'base':base,'aug':base+[f"ORD={k}"]})
    return evs, eligible_lines


def make_phase_events(lines, rng=None, permute=False):
    rng=rng or random.Random(SEED); evs=[]
    for ln in lines:
        seq=ln.tokens; n=len(seq)
        if n<MIN_LINE_LEN: continue
        tmp=[]
        for idx in range(2,n-2):
            q=min(4,int(5.0*idx/max(1,n-1)))
            base=[f"section={ln.section}",f"currier={ln.currier}",f"n={min(n,25)}",f"para={int(ln.para_start)}"]
            add_tok_feats(base,'w1',seq[0]); add_tok_feats(base,'wn',seq[-1])
            add_tok_feats(base,'p1',seq[idx-1]); add_tok_feats(base,'p2',seq[idx-2])
            add_tok_feats(base,'n1',seq[idx+1]); add_tok_feats(base,'n2',seq[idx+2])
            tmp.append({'line_id':ln.line_id,'folio':ln.folio,'quire':ln.quire,'currier':ln.currier,
                        'section':ln.section,'k':q,'token':seq[idx],'base':base})
        if permute and len(tmp)>1:
            qs=[x['k'] for x in tmp]; shift=rng.randrange(1,len(qs)); qs=qs[shift:]+qs[:shift]
            for x,q in zip(tmp,qs): x['k']=q
        for x in tmp: x['aug']=x['base']+[f"PHASE={x['k']}"]
        evs.extend(tmp)
    return evs


def top_tokens(lines,n=64):
    c=collections.Counter(t for ln in lines for t in ln.tokens)
    return set(t for t,_ in c.most_common(n)),c


def task_label(ev,task,common):
    t=ev['token']
    if task=='first': return t[0]
    if task=='last': return t[-1]
    if task=='length': return str(min(len(t),12))
    if task=='lex64': return t if t in common else '__OTHER__'
    raise ValueError(task)


def safe_nll_bits(model,X,y):
    probs=model.predict_proba(X); classes={c:i for i,c in enumerate(model.classes_)}
    out=np.empty(len(y),dtype=float)
    for j,val in enumerate(y):
        i=classes.get(val); p=probs[j,i] if i is not None else EPS
        out[j]=-math.log(max(EPS,float(p)),2)
    return out


def cv_score(events,common,tasks=('first','last','length','lex64'),alpha=ALPHA,n_splits=5,seed=SEED):
    groups=np.array([e['quire'] for e in events],dtype=object); uniq=sorted(set(groups)); splits=min(n_splits,len(uniq))
    if splits<2: raise ValueError('need >=2 groups')
    gkf=GroupKFold(n_splits=splits); hasher=FeatureHasher(n_features=HASH_DIM,input_type='string',alternate_sign=False)
    Xb=hasher.transform([e['base'] for e in events]); Xa=hasher.transform([e['aug'] for e in events]); idx_all=np.arange(len(events))
    results={}; primary_event_gain=np.full(len(events),np.nan)
    for task in tasks:
        y=np.array([task_label(e,task,common) for e in events],dtype=object); nb=np.full(len(events),np.nan); na=np.full(len(events),np.nan)
        for fold,(tr,te) in enumerate(gkf.split(idx_all,y,groups)):
            kw=dict(loss='log_loss',penalty='l2',alpha=alpha,max_iter=400,tol=1e-3,average=True,random_state=seed+fold)
            mb=SGDClassifier(**kw); ma=SGDClassifier(**kw); mb.fit(Xb[tr],y[tr]); ma.fit(Xa[tr],y[tr])
            nb[te]=safe_nll_bits(mb,Xb[te],y[te]); na[te]=safe_nll_bits(ma,Xa[te],y[te])
        gain=nb-na
        results[task]={'base_nll_bits':float(np.nanmean(nb)),'aug_nll_bits':float(np.nanmean(na)),
                       'gain_bits_per_event':float(np.nanmean(gain)),'gain_sd_events':float(np.nanstd(gain,ddof=1))}
        if task=='first': primary_event_gain[:]=gain
    outs={}
    for name,extractor in [('primary_by_quire',lambda e:e['quire']),('primary_by_section',lambda e:e['section']),
                           ('primary_by_currier',lambda e:e['currier']),('primary_by_position',lambda e:str(e['k']))]:
        b=collections.defaultdict(list)
        for e,g in zip(events,primary_event_gain):
            if not np.isnan(g): b[extractor(e)].append(float(g))
        outs[name]={k:{'n':len(v),'mean_gain_bits':float(np.mean(v))} for k,v in b.items()}
    q=outs['primary_by_quire']; pos=sum(v['mean_gain_bits']>0 for v in q.values()); nq=len(q)
    sign_p=float(binomtest(pos,nq,0.5,alternative='greater').pvalue) if nq else None
    return {'n_events':len(events),'n_groups':len(uniq),'folds':splits,'tasks':results,**outs,
            'quire_positive':pos,'quire_total':nq,'quire_sign_p':sign_p}


def match_prob_by_k(events):
    by=collections.defaultdict(list)
    for e in events: by[e['k']].append(e['token'][0])
    out={}
    for k,vals in sorted(by.items()):
        c=collections.Counter(vals); n=len(vals); R=sum(v*(v-1) for v in c.values())/(n*(n-1)) if n>1 else float('nan')
        out[str(k)]={'n':n,'R_firstglyph':R,'modal':c.most_common(1)[0] if c else None}
    return out


def summarize_null(obs_score,null_scores,task='first'):
    obs=obs_score['tasks'][task]['gain_bits_per_event']; vals=np.array([x['tasks'][task]['gain_bits_per_event'] for x in null_scores])
    mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else float('nan'); z=(obs-mean)/sd if sd>0 else float('nan')
    return {'observed':obs,'null_mean':mean,'null_sd':sd,'delta':obs-mean,'z':z,'null_reps':len(vals),
            'null_min':float(vals.min()),'null_max':float(vals.max())}


def run_axis(lines,common,axis,null_reps,outdir):
    rng=random.Random(SEED+{'left':1,'right':2,'phase':3}[axis])
    if axis in ('left','right'):
        obs_events,nlines=make_edge_events(lines,reverse=(axis=='right'),anchor_mode='observed',rng=rng); obs=cv_score(obs_events,common); nulls=[]
        for r in range(null_reps):
            erng=random.Random(SEED+10000*({'left':1,'right':2}[axis])+r)
            ev,_=make_edge_events(lines,reverse=(axis=='right'),anchor_mode='internal',rng=erng); nulls.append(cv_score(ev,common))
            if (r+1)%5==0: save_ckpt(outdir,f'{axis}_null_{r+1:03d}',{'axis':axis,'obs':obs,'nulls':nulls})
        return {'axis':axis,'eligible_lines':nlines,'observed':obs,'nulls':nulls,'headline':summarize_null(obs,nulls),
                'descriptive_match_prob':match_prob_by_k(obs_events)}
    obs_events=make_phase_events(lines,rng=rng,permute=False); obs=cv_score(obs_events,common); nulls=[]
    for r in range(null_reps):
        ev=make_phase_events(lines,rng=random.Random(SEED+30000+r),permute=True); nulls.append(cv_score(ev,common))
        if (r+1)%5==0: save_ckpt(outdir,f'phase_null_{r+1:03d}',{'axis':'phase','obs':obs,'nulls':nulls})
    return {'axis':'phase','eligible_lines':len(set(e['line_id'] for e in obs_events)),'observed':obs,'nulls':nulls,
            'headline':summarize_null(obs,nulls),'descriptive_match_prob':match_prob_by_k(obs_events)}


def verdict(h,obs):
    z=h['z']; distributed=obs['quire_total']>0 and obs['quire_positive']/obs['quire_total']>=0.60
    if not math.isfinite(z) or z<2: return 'UNRESOLVED'
    return 'SURVIVES_PRIMARY_GATE' if distributed else 'LOCALIZED_NOT_CORPUS_WIDE'


def fmt(x,d=5):
    if x is None: return 'NA'
    try:
        if not math.isfinite(float(x)): return 'NA'
    except Exception: return str(x)
    return f"{float(x):.{d}f}"


def build_report(result):
    a=result['audit']; axes=result['axes']; L=[]
    L += ['# Running results — representation-independent LAAFU ordinal-position test v0.1','',
          '## RETRACTED FINDINGS','',
          'None in this run. PGCS-based interpretation is explicitly excluded from the test design.','',
          '## PRE-REGISTERED QUESTION AND GATES','',
          'Question: after controlling with raw-token local context plus section, Currier, paragraph state and line length, does physical line coordinate still predict token form?','',
          '- Primary response: first raw EVA character of the current token.',
          '- Primary effect size: held-out quire-blocked log-loss gain (bits/event) from adding ordinal/phase coordinate to the baseline.',
          '- LEFT/RIGHT matched null: move the six-token pseudo-edge to a later contiguous window in the same line; exact words and local transitions are preserved.',
          '- PHASE matched null: cyclically permute relative-position labels within each line while leaving tokens and baseline contexts fixed.',
          '- Gate: observed gain >=2 null SD above matched-null mean; corpus-wide claim additionally requires >=60% positive held-out quire groups.',
          '- If z<2, reporting begins: “the metric does not resolve this.”','',
          'No word decomposition, PGCS slot, morphological family, or manually defined Voynich grammar is used.','',
          '## DATA AUDIT','',
          f"Source: ZL3b-n.txt; paragraph loci only. Parsed {a['paragraph_lines_kept']} lines / {a['tokens_kept']} conservative raw EVA tokens across {a['folios']} folios.",
          f"Source SHA-256: `{a['source_sha256']}`",f"Currier counts: `{dict(a['currier_counts'])}`",f"Section counts: `{dict(a['section_counts'])}`",'']
    for name in ('left','right','phase'):
        x=axes[name]; h=x['headline']; o=x['observed']; v=verdict(h,o); L += [f"## {name.upper()} RESULT",'']
        if h['z']<2: L.append('**The metric does not resolve this.**')
        L.append(f"Headline: observed first-glyph predictive gain {fmt(h['observed'])} bits/event versus matched-null mean {fmt(h['null_mean'])} with null SD {fmt(h['null_sd'])}; delta {fmt(h['delta'])}, z={fmt(h['z'],2)}.")
        L.append(f"Distribution: positive OOF gain in {o['quire_positive']}/{o['quire_total']} quire groups; one-sided sign-test p={fmt(o['quire_sign_p'],4)}. Gate verdict: **{v}**.")
        L += ['','| task | baseline NLL bits | +position NLL bits | held-out gain bits/event |','|---|---:|---:|---:|']
        for task,d in o['tasks'].items(): L.append(f"| {task} | {fmt(d['base_nll_bits'])} | {fmt(d['aug_nll_bits'])} | {fmt(d['gain_bits_per_event'])} |")
        for title,key in [('Primary gain by Currier','primary_by_currier'),('Primary gain by section','primary_by_section'),('Primary gain by ordinal/phase bin','primary_by_position')]:
            L += ['',title+':']
            for k,d in sorted(o[key].items()): L.append(f"- {k}: n={d['n']}, gain={fmt(d['mean_gain_bits'])} bits/event")
        L.append('')
    L += ['## INTERPRETATION RULE','',
          'LEFT surviving while PHASE fails supports a decaying line-start/reset process rather than whole-line planning. RIGHT surviving independently supports a closure process. PHASE surviving after bidirectional local context supports a genuine whole-line coordinate. Failure against the matched null means the apparent ordinal curve is adequately reproduced by local sequence plus matched line geometry.','',
          '## AUDIT COMPLETENESS','',
          '- Circularity: no PGCS-derived feature enters predictor or response.',
          '- Leakage: evaluation is quire-group blocked; hashing is stateless.',
          '- Confounds: section, Currier, paragraph flag, line length and raw local token context are baseline predictors; LEFT/RIGHT also exclude the opposite edge with a 3-token guard.',
          '- Matched nulls: exact-line internal anchors for LEFT/RIGHT; within-line phase permutation for PHASE.',
          '- Measurement degeneracy: exact line length, exact left distance and exact right distance are never jointly conditioned; LEFT/RIGHT/PHASE are separate tests.',
          '- Representation dependence: primary first glyph; diagnostics last glyph, token length and common raw token identity.',
          '- Decision rule: fixed z>=2 plus >=60% positive-quire criterion for corpus-wide claim.',
          '- Full null arrays, parameters and source hash are retained in RESULTS.json.','']
    return '\n'.join(L)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--section-map',default='voynich_section_map.json')
    ap.add_argument('--outdir',default='results/laafu_ordinal_v01'); ap.add_argument('--null-reps',type=int,default=DEFAULT_NULL_REPS); args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True); source=Path(args.source); smap=load_section_map(Path(args.section_map))
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest(); lines,audit=parse_ivtff(source,smap); audit['source_sha256']=source_sha
    audit['parser_parameters']={'min_line_len':MIN_LINE_LEN,'max_ord':MAX_ORD,'right_guard':RIGHT_GUARD,'seed':SEED,'hash_dim':HASH_DIM,'alpha':ALPHA,'null_reps':args.null_reps}
    save_ckpt(outdir,'01_parsed',{'audit':audit,'lines':lines}); common,cnt=top_tokens(lines,64); audit['common64_coverage']=sum(cnt[t] for t in common)/sum(cnt.values())
    axes={}
    for axis in ('left','right','phase'):
        print(f'RUN {axis}',flush=True); axes[axis]=run_axis(lines,common,axis,args.null_reps,outdir); save_ckpt(outdir,f'02_{axis}_done',axes[axis])
    result={'version':'0.1','audit':audit,'axes':axes}; (outdir/'RESULTS.json').write_text(json.dumps(result,indent=2,default=lambda x:dict(x) if isinstance(x,collections.Counter) else str(x)),encoding='utf-8')
    (outdir/'RESULTS.md').write_text(build_report(result),encoding='utf-8'); save_ckpt(outdir,'99_complete',result); print(build_report(result))

if __name__=='__main__': main()
