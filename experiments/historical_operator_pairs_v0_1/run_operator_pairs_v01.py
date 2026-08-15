from __future__ import annotations

import argparse, bz2, hashlib, json, lzma, math, re, statistics, unicodedata, xml.etree.ElementTree as ET, zipfile, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SEED = 20260815
BOOT = 2000
PRIMARY = ("H0", "H1", "H0_minus_H1")
METRIC_NAMES = (
    "H0", "H1", "H2", "H0_minus_H1", "mean_outgoing_entropy",
    "deterministic_context_frac", "bigram_type_density", "tok_len_mean",
    "tok_len_std", "ttr", "hapax_rate", "one_edit_neighbor_rate",
    "adjacent_repeat_rate", "zlib_ratio", "bz2_ratio", "lzma_ratio",
)

def lname(tag): return tag.rsplit('}',1)[-1].lower()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def norm_text(s): return re.sub(r"\s+"," ",unicodedata.normalize("NFC",s or "")).strip()
def keep_char(ch): return bool(ch) and unicodedata.category(ch)[0] in {"L","M","N","S"}
def char_stream(text): return ''.join(ch for ch in unicodedata.normalize('NFC',text) if keep_char(ch))

def tokens(text):
    out=[]
    for raw in norm_text(text).split():
        t=''.join(ch for ch in raw if keep_char(ch))
        if t: out.append(t)
    return out

def entropy_counts(c):
    n=sum(c.values())
    if not n: return float('nan')
    return -sum((v/n)*math.log2(v/n) for v in c.values() if v)

def cond_entropy(s,order):
    if len(s)<=order:return float('nan')
    ctx=defaultdict(Counter)
    for i in range(order,len(s)):ctx[s[i-order:i]][s[i]]+=1
    total=sum(sum(c.values()) for c in ctx.values())
    if not total:return float('nan')
    return sum((sum(c.values())/total)*entropy_counts(c) for c in ctx.values())

def outgoing_stats(s):
    ctx=defaultdict(Counter)
    for a,b in zip(s,s[1:]):ctx[a][b]+=1
    if not ctx:return float('nan'),float('nan')
    return statistics.fmean(entropy_counts(c) for c in ctx.values()),sum(len(c)==1 for c in ctx.values())/len(ctx)

def one_edit_neighbor_rate(ts):
    vocab=set(ts)
    if len(vocab)<2:return 0.0
    hit=set()
    for w in vocab:
        for i in range(len(w)):
            v=w[:i]+w[i+1:]
            if v in vocab:hit.add(w);hit.add(v)
    buckets=defaultdict(list)
    for w in vocab:
        for i in range(len(w)):buckets[(len(w),i,w[:i],w[i+1:])].append(w)
    for vs in buckets.values():
        if len(vs)>1:hit.update(vs)
    return len(hit)/len(vocab)

def compression_ratios(s):
    b=s.encode('utf-8')
    if not b:return (float('nan'),)*3
    return len(zlib.compress(b,9))/len(b),len(bz2.compress(b,9))/len(b),len(lzma.compress(b,preset=9))/len(b)

def metrics(text):
    s=char_stream(text);ts=tokens(text);H0=entropy_counts(Counter(s));H1=cond_entropy(s,1);H2=cond_entropy(s,2);mo,det=outgoing_stats(s)
    lens=[len(t) for t in ts];tc=Counter(ts);zr,br,lr=compression_ratios(s)
    return {"H0":H0,"H1":H1,"H2":H2,"H0_minus_H1":H0-H1 if math.isfinite(H0) and math.isfinite(H1) else float('nan'),
    "mean_outgoing_entropy":mo,"deterministic_context_frac":det,"bigram_type_density":len(set(zip(s,s[1:])))/max(1,len(s)-1),
    "tok_len_mean":statistics.fmean(lens) if lens else float('nan'),"tok_len_std":statistics.pstdev(lens) if len(lens)>1 else 0.0 if lens else float('nan'),
    "ttr":len(tc)/len(ts) if ts else float('nan'),"hapax_rate":sum(v==1 for v in tc.values())/len(tc) if tc else float('nan'),
    "one_edit_neighbor_rate":one_edit_neighbor_rate(ts),"adjacent_repeat_rate":sum(a==b for a,b in zip(ts,ts[1:]))/max(1,len(ts)-1),
    "zlib_ratio":zr,"bz2_ratio":br,"lzma_ratio":lr,"n_chars":len(s),"n_tokens":len(ts),"n_types":len(tc)}

def render_unicode(node,expanded):
    parts=[]
    if node.text:parts.append(node.text)
    for child in list(node):
        nm=lname(child.tag)
        if nm=='del':pass
        elif nm=='ex' and not expanded:pass
        else:parts.append(render_unicode(child,expanded))
        if child.tail:parts.append(child.tail)
    return ''.join(parts)

def parse_nuremberg(zpath):
    docs=[];total_lines=abbr_lines=noabbr_lines=noabbr_equal=parse_fail=0
    with zipfile.ZipFile(zpath) as z:
        names=[n for n in z.namelist() if n.startswith('nuremberg_letterbooks/diplomatic-regularised/') and n.lower().endswith('.xml')]
        for name in sorted(names):
            try:root=ET.fromstring(z.read(name))
            except Exception:parse_fail+=1;continue
            a_lines=[];e_lines=[];writers=[];doc_abbr=0
            for tl in root.iter():
                if lname(tl.tag)!='textline':continue
                u=next((x for x in tl.iter() if lname(x.tag)=='unicode'),None)
                if u is None:continue
                has_ex=any(lname(x.tag)=='ex' for x in u.iter());a=norm_text(render_unicode(u,False));e=norm_text(render_unicode(u,True))
                if not a and not e:continue
                total_lines+=1
                if has_ex:abbr_lines+=1;doc_abbr+=1
                else:
                    noabbr_lines+=1
                    if a==e:noabbr_equal+=1
                a_lines.append(a);e_lines.append(e)
                if tl.attrib.get('writerID'):writers.extend(x.strip() for x in tl.attrib['writerID'].split(',') if x.strip())
            if a_lines:docs.append({'id':name,'abbr':norm_text(' '.join(a_lines)),'expanded':norm_text(' '.join(e_lines)),'abbr_lines':doc_abbr,'n_lines':len(a_lines),'writers':sorted(set(writers))})
    qc={'xml_candidates':len(names),'parsed_documents':len(docs),'parse_failures':parse_fail,'paired_lines':total_lines,'abbreviation_bearing_lines':abbr_lines,
        'no_abbreviation_lines':noabbr_lines,'no_abbreviation_identity_rate':noabbr_equal/max(1,noabbr_lines),'abbreviation_bearing_documents':sum(d['abbr_lines']>0 for d in docs)}
    return docs,qc

def alto_lines(xml_bytes):
    root=ET.fromstring(xml_bytes);out={};seq=0
    for tl in root.iter():
        if lname(tl.tag)!='textline':continue
        seq+=1;ident=tl.attrib.get('ID') or tl.attrib.get('id') or str(seq);vals=[]
        for x in tl.iter():
            if lname(x.tag)=='string':
                c=x.attrib.get('CONTENT') or x.attrib.get('content')
                if c:vals.append(c)
        if not vals:
            txt=''.join(tl.itertext())
            if txt.strip():vals.append(txt)
        out[ident]=norm_text(' '.join(vals))
    return out

def or_key(name):
    m=re.search(r'IRHT_P_\d+',name);return m.group(0) if m else None

def parse_oriflamms(zpath):
    docs=[]
    with zipfile.ZipFile(zpath) as z:
        abbr={};expan={}
        for n in z.namelist():
            low=n.lower()
            if not low.endswith('.xml') or '/without-norm/' not in low:continue
            k=or_key(n)
            if not k:continue
            if '/alto_abbr/' in low:abbr[k]=n
            elif '/alto_expan/' in low:expan[k]=n
        common_keys=sorted(set(abbr)&set(expan));total_a_lines=total_e_lines=common_lines=equal_lines=diff_lines=parse_fail=0;overlaps=[]
        for k in common_keys:
            try:aa=alto_lines(z.read(abbr[k]));ee=alto_lines(z.read(expan[k]))
            except Exception:parse_fail+=1;continue
            total_a_lines+=len(aa);total_e_lines+=len(ee);ids=sorted(set(aa)&set(ee));overlaps.append(len(ids)/max(1,max(len(aa),len(ee))))
            if not ids:continue
            av=[];ev=[];ndiff=0
            for i in ids:
                a,e=norm_text(aa[i]),norm_text(ee[i]);common_lines+=1
                if a==e:equal_lines+=1
                else:diff_lines+=1;ndiff+=1
                av.append(a);ev.append(e)
            docs.append({'id':k,'abbr':norm_text(' '.join(av)),'expanded':norm_text(' '.join(ev)),'abbr_lines':ndiff,'n_lines':len(ids),'writers':[]})
    qc={'abbr_files':len(abbr),'expanded_files':len(expan),'paired_manuscripts':len(common_keys),'parsed_documents':len(docs),'parse_failures':parse_fail,
        'abbr_lines_total':total_a_lines,'expanded_lines_total':total_e_lines,'paired_lines':common_lines,'abbreviation_bearing_lines':diff_lines,
        'identical_aligned_lines':equal_lines,'median_line_id_overlap':float(np.median(overlaps)) if overlaps else 0.0,'min_line_id_overlap':min(overlaps) if overlaps else 0.0}
    return docs,qc

def metric_docs(docs):
    out=[]
    for d in docs:
        ma=metrics(d['abbr']);me=metrics(d['expanded'])
        if min(ma['n_chars'],me['n_chars'])<30:continue
        delta={k:ma[k]-me[k] for k in METRIC_NAMES}
        out.append({'id':d['id'],'abbr_lines':d['abbr_lines'],'n_lines':d['n_lines'],'writers':d['writers'],'abbr':ma,'expanded':me,'delta':delta,'abbr_text':d['abbr'],'expanded_text':d['expanded']})
    return out

def bootstrap_summary(vals,rng):
    vals=vals[np.isfinite(vals)];n=len(vals)
    if not n:return {'n':0,'median':None,'ci95':[None,None],'q025':None,'q975':None}
    med=float(np.median(vals));boots=np.empty(BOOT)
    for i in range(BOOT):boots[i]=np.median(vals[rng.integers(0,n,size=n)])
    lo,hi=np.percentile(boots,[2.5,97.5]);qlo,qhi=np.percentile(vals,[2.5,97.5])
    return {'n':n,'median':med,'ci95':[float(lo),float(hi)],'q025':float(qlo),'q975':float(qhi),'mean':float(np.mean(vals)),'sd':float(np.std(vals))}

def length_matched_h01(record):
    a=char_stream(record['abbr_text']);e=char_stream(record['expanded_text']);n=len(a)
    if n<30 or len(e)<n:return float('nan'),float('nan')
    if len(e)==n:windows=[e]
    else:
        starts=np.unique(np.linspace(0,len(e)-n,num=min(21,len(e)-n+1),dtype=int));windows=[e[int(st):int(st)+n] for st in starts]
    e0=np.median([entropy_counts(Counter(x)) for x in windows]);e1=np.median([cond_entropy(x,1) for x in windows])
    return entropy_counts(Counter(a))-e0,cond_entropy(a,1)-e1

def corpus_analysis(records,label,qc):
    rng=np.random.default_rng(SEED+(1 if label=='Nuremberg' else 2));abr=[r for r in records if r['abbr_lines']>0];summaries={}
    for m in METRIC_NAMES:summaries[m]=bootstrap_summary(np.array([r['delta'][m] for r in abr],dtype=float),rng)
    lm=np.array([length_matched_h01(r) for r in abr],dtype=float);lm0=bootstrap_summary(lm[:,0] if len(lm) else np.array([]),rng);lm1=bootstrap_summary(lm[:,1] if len(lm) else np.array([]),rng)
    pa=metrics(' '.join(r['abbr_text'] for r in records));pe=metrics(' '.join(r['expanded_text'] for r in records));pd={m:pa[m]-pe[m] for m in METRIC_NAMES}
    h0,h1,gap=summaries['H0'],summaries['H1'],summaries['H0_minus_H1']
    qdir=h1['median'] is not None and h1['median']<0 and h1['ci95'][1] is not None and h1['ci95'][1]<0
    qsel=h0['median'] is not None and (abs(h0['median'])<abs(h1['median']) or (gap['median'] is not None and gap['median']>0))
    qlen=lm1['median'] is not None and lm1['median']<0;qual=bool(qdir and qsel and qlen)
    safe=[{k:r[k] for k in ['id','abbr_lines','n_lines','writers','abbr','expanded','delta']} for r in records]
    return {'label':label,'qc':qc,'documents_metric_eligible':len(records),'abbreviation_bearing_documents_metric_eligible':len(abr),'metric_shift_summaries':summaries,
            'length_matched':{'delta_H0':lm0,'delta_H1':lm1},'pooled':{'abbr':pa,'expanded':pe,'delta':pd},
            'qualification':{'H1_negative_ci':qdir,'selective_vs_H0_or_gap_positive':qsel,'length_matched_H1_direction_negative':qlen,'QUALIFIED':qual},'records':safe}

def qc_gate(nq,oq):
    fails=[]
    if nq.get('paired_lines',0)<5000:fails.append('Nuremberg paired lines <5000')
    if nq.get('abbreviation_bearing_lines',0)<500:fails.append('Nuremberg abbreviation lines <500')
    if nq.get('no_abbreviation_identity_rate',0)<0.98:fails.append('Nuremberg no-abbrev identity <98%')
    if oq.get('paired_manuscripts',0)<30:fails.append('ORIFLAMMS paired manuscripts <30')
    if oq.get('abbreviation_bearing_lines',0)<1000:fails.append('ORIFLAMMS abbreviation-bearing aligned units <1000')
    if oq.get('median_line_id_overlap',0)<0.98:fails.append('ORIFLAMMS median line-ID overlap <98%')
    return not fails,fails

def json_safe(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,list):return [json_safe(v) for v in x]
    return x

def canonical_hash(obj):return hashlib.sha256(json.dumps(json_safe(obj),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def load_voynich(path,transcriber='ZLZI'):
    d=json.loads(Path(path).read_text());lines=[];folios=0
    for fid,page in sorted(d['pages'].items()):
        used=False
        for lnum,line in sorted(page.items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
            txt=line.get('t',{}).get(transcriber,'')
            if txt:lines.append(txt);used=True
        if used:folios+=1
    return norm_text(' '.join(lines)),{'transcriber':transcriber,'folios':folios,'lines':len(lines)}

def target_bridge(tm,ext):
    covered={};residuals={};intervals={}
    for m in PRIMARY:
        r=tm[m]-ext['pooled']['expanded'][m];sm=ext['metric_shift_summaries'][m];lo,hi=sm['q025'],sm['q975'];residuals[m]=r;intervals[m]=[lo,hi];covered[m]=bool(lo is not None and hi is not None and lo<=r<=hi)
    return {'residual_vs_pooled_expanded':residuals,'operator_empirical_95_interval':intervals,'covered':covered,'covered_count':sum(covered.values()),'PASS_2_OF_3':sum(covered.values())>=2}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nuremberg',required=True);ap.add_argument('--oriflamms',required=True);ap.add_argument('--voynich',default='voynich_transcriptions_slim.json');ap.add_argument('--out',default='experiments/historical_operator_pairs_v0_1/results_v01');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('Parsing Nuremberg...',flush=True);ndocs,nqc=parse_nuremberg(args.nuremberg);print('Nuremberg QC',nqc,flush=True)
    print('Parsing ORIFLAMMS...',flush=True);odocs,oqc=parse_oriflamms(args.oriflamms);print('ORIFLAMMS QC',oqc,flush=True)
    gate,fail=qc_gate(nqc,oqc);schema={'nuremberg':nqc,'oriflamms':oqc,'gate':gate,'failures':fail,'sources':{'nuremberg_sha256':sha256_file(args.nuremberg),'oriflamms_sha256':sha256_file(args.oriflamms)}}
    (out/'schema_qc.json').write_text(json.dumps(json_safe(schema),ensure_ascii=False,indent=2))
    if not gate:
        result={'verdict':'SCHEMA_OR_PAIRING_FAILURE','schema':schema};(out/'full_result.json').write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2));print('VERDICT SCHEMA_OR_PAIRING_FAILURE',fail);return
    print('Computing external metrics...',flush=True);nr=metric_docs(ndocs);orr=metric_docs(odocs);na=corpus_analysis(nr,'Nuremberg',nqc);oa=corpus_analysis(orr,'ORIFLAMMS',oqc)
    freeze={'schema':schema,'nuremberg':na,'oriflamms':oa,'sign_convention':'ABBREVIATED_MINUS_EXPANDED','bootstrap_replicates':BOOT,'seed':SEED};fh=canonical_hash(freeze);freeze['freeze_sha256']=fh;(out/'external_freeze.json').write_text(json.dumps(json_safe(freeze),ensure_ascii=False,indent=2))
    print('EXTERNAL_FREEZE',fh,'N qualified',na['qualification']['QUALIFIED'],'O qualified',oa['qualification']['QUALIFIED'],flush=True)
    both=na['qualification']['QUALIFIED'] and oa['qualification']['QUALIFIED']
    if not both:
        result={'verdict':'HISTORICAL_ABBREVIATION_NOT_SUPPORTED','external_freeze_sha256':fh,'nuremberg_qualification':na['qualification'],'oriflamms_qualification':oa['qualification'],'voynich_opened':False};(out/'full_result.json').write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2));print('VERDICT HISTORICAL_ABBREVIATION_NOT_SUPPORTED; Voynich remains sealed');return
    print('External gate passed; opening pinned repository Voynich transcription...',flush=True);vtext,vmeta=load_voynich(args.voynich,'ZLZI');vm=metrics(vtext);nb=target_bridge(vm,na);ob=target_bridge(vm,oa);strong=nb['PASS_2_OF_3'] and ob['PASS_2_OF_3'];verdict='HISTORICAL_ABBREVIATION_MECHANISM_SUPPORTED' if strong else 'HISTORICAL_ABBREVIATION_DIRECTION_ONLY'
    result={'verdict':verdict,'external_freeze_sha256':fh,'voynich_opened':True,'voynich':{'meta':vmeta,'metrics':vm},'bridge':{'Nuremberg':nb,'ORIFLAMMS':ob},'nuremberg_qualification':na['qualification'],'oriflamms_qualification':oa['qualification']};result['result_sha256']=canonical_hash(result);(out/'full_result.json').write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2));print('VERDICT',verdict,'bridge',nb['covered_count'],ob['covered_count'])

if __name__=='__main__':main()
