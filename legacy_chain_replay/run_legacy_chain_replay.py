#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, re, statistics, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache

PROTOCOL_SHA = 'd464cdc717e55d4233e2e5700be85b14fa2bc62a7691ac024b9e9bf98949533f'
ELIGIBLE_SECTIONS = ['Herbal-A','Herbal-B','Astronomical','Cosmological','Zodiac','Rosettes','Balneological','Pharmaceutical','Stars']


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load_protocol(root: Path):
    p=root/'protocol.json'
    got=sha256_file(p)
    if got != PROTOCOL_SHA: raise SystemExit(f'protocol hash mismatch {got} != {PROTOCOL_SHA}')
    return json.loads(p.read_text())


@lru_cache(maxsize=2_000_000)
def ed1(a: str, b: str) -> bool:
    if a == b: return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1: return False
    if la==lb:
        return sum(x!=y for x,y in zip(a,b))==1
    if la>lb: a,b,la,lb=b,a,lb,la
    i=j=0; skipped=0
    while i<la and j<lb:
        if a[i]==b[j]: i+=1; j+=1
        else:
            skipped += 1; j += 1
            if skipped>1: return False
    return True


def metrics(lines):
    adj=edn=ex=leneq=lenclose=0; lendiff=0.0
    triples=chains=aba=ret=0; eligible_prev=0
    lag_counts={h:[0,0] for h in (1,2,3,4)}
    for line in lines:
        n=len(line)
        for h in (1,2,3,4):
            for i in range(n-h):
                lag_counts[h][1]+=1
                lag_counts[h][0]+=int(ed1(line[i],line[i+h]))
        for a,b in zip(line,line[1:]):
            adj+=1; e=ed1(a,b); edn+=int(e); ex+=int(a==b)
            leneq+=int(len(a)==len(b)); lenclose+=int(abs(len(a)-len(b))<=1); lendiff+=abs(len(a)-len(b))
        for i in range(n-2):
            a,b,c=line[i:i+3]; triples+=1
            e1=ed1(a,b); e2=ed1(b,c)
            if e1:
                eligible_prev+=1; chains+=int(e2)
            aba+=int(a==c and a!=b)
            ret+=int(e1 and a==c)
    base=edn/adj if adj else None
    cond=chains/eligible_prev if eligible_prev else None
    lift=(cond-base) if (cond is not None and base is not None) else None
    out={
      'n_lines':len(lines),'n_tokens':sum(map(len,lines)),'adjacent_pairs':adj,'triples':triples,
      'ed1_chain_lift':lift,'ed1_chain_rate':chains/triples if triples else None,
      'ed1_conditional_next':cond,'ed1_rate_l1':base,'exact_rate_l1':ex/adj if adj else None,
      'len_equal_l1':leneq/adj if adj else None,'len_close_l1':lenclose/adj if adj else None,
      'len_diff_mean_l1':lendiff/adj if adj else None,'aba_rate':aba/triples if triples else None,
      'ed1_return_rate':ret/triples if triples else None,'eligible_prev_ed1':eligible_prev,'ed1_chain_events':chains,
    }
    for h,(c,n) in lag_counts.items(): out[f'ed1_rate_l{h}']=c/n if n else None
    return out


def shuffle_lines(lines, rng):
    out=[]
    for line in lines:
        z=list(line); rng.shuffle(z); out.append(z)
    return out


def null_values(lines, reps, seed):
    rng=random.Random(seed); vals=[]
    for _ in range(reps):
        v=metrics(shuffle_lines(lines,rng))['ed1_chain_lift']
        if v is not None and math.isfinite(v): vals.append(v)
    return vals


def summarize(vals):
    vals=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not vals: return None
    q=sorted(vals)
    def pct(p):
        x=(len(q)-1)*p; lo=int(math.floor(x)); hi=int(math.ceil(x))
        return q[lo] if lo==hi else q[lo]*(hi-x)+q[hi]*(x-lo)
    return {'n':len(vals),'median':statistics.median(vals),'mean':statistics.mean(vals),'sd':statistics.pstdev(vals) if len(vals)>1 else 0.0,
            'q025':pct(.025),'q975':pct(.975),'min':min(vals),'max':max(vals)}


def clean_words(s):
    return re.findall(r'[a-z]+', (s or '').lower())


def load_vms(path: Path, section_map_path: Path):
    obj=json.loads(path.read_text())
    sm=json.loads(section_map_path.read_text()); sm=sm.get('mapping',sm)
    by_page=[]; by_section=defaultdict(list); pooled=[]
    pages=obj.get('pages',obj)
    for page, lns in pages.items():
        sec=sm.get(page)
        if sec not in ELIGIBLE_SECTIONS: continue
        page_lines=[]
        def line_key(k):
            m=re.match(r'(\d+)',str(k)); return int(m.group(1)) if m else 10**9
        for lnno in sorted(lns,key=line_key):
            rec=lns[lnno]
            txt=(rec.get('t') or {}).get('ZLZI','') if isinstance(rec,dict) else ''
            toks=clean_words(txt)
            if len(toks)>=2:
                page_lines.append(toks); by_section[sec].append(toks); pooled.append(toks)
        if page_lines: by_page.append({'page':page,'section':sec,'lines':page_lines})
    if sum(map(len,pooled))<20000: raise SystemExit('VMS parse unexpectedly small')
    return pooled, dict(by_section), by_page


def build_ed1_graph(vocab):
    vocab=set(vocab); graph={w:set() for w in vocab}
    buckets=defaultdict(list)
    for w in vocab:
        for i in range(len(w)):
            buckets[(len(w),i,w[:i],w[i+1:])].append(w)
    for ws in buckets.values():
        if len(ws)>1:
            for i,a in enumerate(ws):
                for b in ws[i+1:]: graph[a].add(b); graph[b].add(a)
    for w in vocab:
        if len(w)<=1: continue
        for i in range(len(w)):
            s=w[:i]+w[i+1:]
            if s in vocab and s!=w: graph[w].add(s); graph[s].add(w)
    return {w:tuple(sorted(v)) for w,v in graph.items()}


def gprime_generate(page_records, seed, R=64, beta=8.0, s=.186, mu=.0009):
    rng=random.Random(seed)
    real_tokens=[w for p in page_records for line in p['lines'] for w in line]
    cnt=Counter(real_tokens); vocab=sorted(cnt); base=[cnt[w] for w in vocab]
    graph=build_ed1_graph(vocab)
    outputs=[]; section_lines=defaultdict(list)
    for prec in page_records:
        lengths=[len(x) for x in prec['lines']]; n=sum(lengths); flat=[]
        for start in range(0,n,R):
            k=min(R,n-start)
            anchor=rng.choices(vocab,weights=base,k=1)[0]
            boost=set(graph.get(anchor,()))|{anchor}
            weights=[base[i]*(beta if w in boost else 1.0) for i,w in enumerate(vocab)]
            flat.extend(rng.choices(vocab,weights=weights,k=k))
        for i in range(1,len(flat)):
            if flat[i]==flat[i-1] and rng.random()<s:
                nb=graph.get(flat[i],())
                if nb: flat[i]=rng.choice(nb)
            if rng.random()<mu:
                nb=graph.get(flat[i-1],())
                if nb: flat[i]=rng.choice(nb)
        idx=0; lines=[]
        for L in lengths:
            line=flat[idx:idx+L]; idx+=L; lines.append(line); section_lines[prec['section']].append(line)
        outputs.extend(lines)
    return outputs,dict(section_lines), {'vocab':len(vocab),'ed1_graph_edges':sum(map(len,graph.values()))//2}


def load_timm_file(path):
    lines=[]
    for raw in Path(path).read_text(errors='ignore').splitlines():
        toks=clean_words(raw)
        if len(toks)>=2: lines.append(toks)
    if sum(map(len,lines))<5000: raise ValueError(f'Timm parse too small {path}')
    return lines


def score_model_seed(name, seed, lines, null_reps):
    met=metrics(lines); null=null_values(lines,null_reps,seed_of(name,seed,'null'))
    return {'seed':seed,'metrics':met,'null':summarize(null)}


def seed_of(*parts):
    h=hashlib.sha256('|'.join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],'big') & 0x7fffffff


def aggregate_arm(rows, target):
    vals=[r['metrics']['ed1_chain_lift'] for r in rows if r['metrics']['ed1_chain_lift'] is not None]
    sm=summarize(vals)
    nullvals=[]
    for r in rows:
        n=r.get('null')
        if n: nullvals.append(n['mean'])
    null_mean=statistics.mean(nullvals) if nullvals else float('nan')
    within=statistics.mean([r['null']['sd'] for r in rows if r.get('null')]) if rows else float('nan')
    between=statistics.pstdev(nullvals) if len(nullvals)>1 else 0.0
    null_sd=math.sqrt(within*within+between*between) if math.isfinite(within) else float('nan')
    z=(sm['median']-null_mean)/null_sd if sm and null_sd>0 else None
    t=target
    ratio=sm['median']/t if sm and t not in (None,0) else None
    same_sign=bool(sm and t and sm['median']*t>0)
    if same_sign and ratio is not None and abs(ratio-1)<=.25 and z is not None and z>=2:
        verdict='MODEL_MATCH'
    elif same_sign and ratio is not None and abs(ratio-1)<=.50:
        verdict='MODEL_PARTIAL'
    else:
        verdict='MODEL_FAIL'
    return {'chain_lift':sm,'null_mean_of_seed_means':null_mean,'null_combined_sd':null_sd,'null_z':z,'ratio_to_vms':ratio,'same_sign':same_sign,'verdict':verdict}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--vms',default='voynich_transcriptions_slim.json')
    ap.add_argument('--section-map',default='voynich_section_map.json')
    ap.add_argument('--timm-dir',required=True)
    ap.add_argument('--out',default='legacy_chain_results')
    args=ap.parse_args(); root=Path(args.root); P=load_protocol(root)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    vms_lines,vms_sections,vms_pages=load_vms(Path(args.vms),Path(args.section_map))
    target=metrics(vms_lines); target_null=null_values(vms_lines,P['controls']['within_line_shuffle']['per_target'],seed_of('VMS','null'))
    qa=[]; rng=random.Random(seed_of('line-order-qa'))
    for _ in range(P['controls']['line_order_shuffle']['repetitions']):
        z=list(vms_lines); rng.shuffle(z); qa.append(metrics(z)['ed1_chain_lift'])
    inv_max=max(abs(x-target['ed1_chain_lift']) for x in qa if x is not None)
    section_targets={s:metrics(lines) for s,lines in sorted(vms_sections.items())}

    arms={}
    gp=P['legacy_arms']['GPRIME']; rows=[]; gp_sec=[]; graph_diag=None
    for seed in gp['seeds']:
        lines,sec_lines,diag=gprime_generate(vms_pages,seed,**{
          'R':gp['parameters']['region_tokens'],'beta':gp['parameters']['beta'],
          's':gp['parameters']['double_substitution_s'],'mu':gp['parameters']['adjacent_nearforming_mu']})
        graph_diag=diag; rows.append(score_model_seed('GPRIME',seed,lines,P['controls']['within_line_shuffle']['per_model_seed']))
        gp_sec.append({'seed':seed,'sections':{s:metrics(x) for s,x in sec_lines.items()}})
    arms['GPRIME']={'status':'EXECUTED_SPECIFICATION_RECONSTRUCTION','rows':rows,'aggregate':aggregate_arm(rows,target['ed1_chain_lift']),'graph_diag':graph_diag,'section_rows':gp_sec}

    tdir=Path(args.timm_dir)
    for mode,key,seeds in [
      ('default','TIMM_DEFAULT',P['legacy_arms']['TIMM_DEFAULT']['seeds']),
      ('noreuse','TIMM_NOREUSE',P['legacy_arms']['TIMM_NOREUSE']['seeds']),
      ('random','TIMM_RANDOM_SOURCE',P['legacy_arms']['TIMM_RANDOM_SOURCE']['seeds']),
      ('position','TIMM_POSITION_SOURCE',P['legacy_arms']['TIMM_POSITION_SOURCE']['seeds'])]:
        rr=[]
        for seed in seeds:
            f=tdir/f'{mode}_seed{seed}.txt'; lines=load_timm_file(f)
            rr.append(score_model_seed(key,seed,lines,P['controls']['within_line_shuffle']['per_model_seed']))
        arms[key]={'status':'EXECUTED_EXACT_PINNED_OUTPUT','rows':rr,'aggregate':aggregate_arm(rr,target['ed1_chain_lift'])}

    q56=root/'legacy_chain_replay'/'q56_injective_anonymous_realiser.py'
    if q56.exists():
        arms['Q57B']={'status':'DEPENDENCY_PRESENT_BUT_RUNNER_NOT_BUNDLED_IN_V01','note':'v0.1 deliberately does not silently reimplement the Q57b stack.'}
    else:
        arms['Q57B']={'status':'NOT_EXACTLY_REPLAYABLE','missing':'q56_injective_anonymous_realiser.py','substitution_permitted':False}

    primary_matches=[k for k in ('GPRIME','TIMM_DEFAULT') if arms[k]['aggregate']['verdict']=='MODEL_MATCH']
    adjudication='CHAIN_LIFT_ALREADY_IMPLIED_BY_LEGACY_MECHANISM' if primary_matches else 'CHAIN_LIFT_NEW_MECHANISTIC_GAP_WITHIN_TESTED_LEGACY_ARMS'
    result={
      'experiment':P['experiment'],'protocol_sha256':PROTOCOL_SHA,
      'target':target,'target_null':summarize(target_null),'target_null_z':(target['ed1_chain_lift']-statistics.mean(target_null))/statistics.pstdev(target_null) if len(target_null)>1 and statistics.pstdev(target_null)>0 else None,
      'line_order_invariance_max_abs_delta':inv_max,'section_targets':section_targets,
      'arms':arms,'primary_matches':primary_matches,'adjudication':adjudication,
      'representation_note':P['upstream']['representation_qualification'],
      'q57b_note':'Exact Q57b is excluded from formal adjudication if its archived q56 dependency is absent; later generators are not substituted.'
    }
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
    md=[]
    md.append('# Voynich legacy ED1-chain replay v0.1 — results')
    md.append('')
    md.append(f"Protocol SHA-256: `{PROTOCOL_SHA}`")
    md.append(f"Adjudication: **{adjudication}**")
    md.append('')
    md.append(f"VMS native-EVA chain lift: **{target['ed1_chain_lift']:.8f}**; within-line-shuffle null mean {statistics.mean(target_null):.8f}, z {result['target_null_z']:.2f}.")
    md.append(f"Line-order invariance QA max |delta|: `{inv_max:.3g}`.")
    md.append('')
    md.append('| arm | median chain lift | ratio to VMS | null z | verdict/status |')
    md.append('|---|---:|---:|---:|---|')
    for k in ('GPRIME','TIMM_DEFAULT','TIMM_NOREUSE','TIMM_RANDOM_SOURCE','TIMM_POSITION_SOURCE'):
        a=arms[k]['aggregate']; md.append(f"| {k} | {a['chain_lift']['median']:.8f} | {a['ratio_to_vms']:.3f} | {a['null_z']:.2f} | {a['verdict']} |")
    md.append(f"| Q57B | — | — | — | {arms['Q57B']['status']} |")
    md.append('')
    md.append('Ablations are causal diagnostics only; they cannot determine the primary legacy-sufficiency verdict.')
    md.append('RF/STA/connected-aaa are used to qualify the Voynich residual as representation-robust; arbitrary legacy EVA strings are not pseudo-converted into STA/aaa.')
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__': main()
