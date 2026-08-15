#!/usr/bin/env python3
"""Post-hoc metric-driver audit for Medieval Magic Formula Discriminator v0.2.

Uses the frozen external qualification and A/C class model. No metrics, thresholds,
transforms, or class centroids are retuned. Decomposes folio A-vs-C squared-distance
advantage by metric and tests whether the apparent short-folio C affinity is driven
by length-sensitive metrics. Secondary/post-hoc; primary verdict unchanged.
"""
from __future__ import annotations
import csv, json, math, subprocess, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import run_full_v02 as m

OUT=m.ROOT/'results_metric_driver_audit_v02'; OUT.mkdir(exist_ok=True)
EXPECTED=['F1_H0','F2_oneedit_component_frac','F2_oneedit_degree','F2_shared_core_ratio','F2_tok_len_mean','F2_tok_len_std','F3_mutation_advantage','F3_nearcopy_lag10','F4_init_final_jsd','F4_line_medial_len','F5_local_global_gain','F7_char_bz2_ct_C','F7_tok_bz2_ct_A','F7_tok_bz2_ncd_A','F7_tok_lzma_ncd_A','F7_tok_lzma_ncd_B','F7_tok_lzma_ncd_C','F7_tok_zlib_ct_C']

def rankdata(x):
    x=np.asarray(x,float);o=np.argsort(x,kind='mergesort');r=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[o[j]]==x[o[i]]: j+=1
        rr=(i+1+j)/2.0;r[o[i:j]]=rr;i=j
    return r

def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3:return float('nan')
    a=rankdata(x[ok]);b=rankdata(y[ok])
    if np.std(a)<1e-12 or np.std(b)<1e-12:return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def normalize_section_map(raw):
    if not isinstance(raw,dict): return {}
    base=raw.get('mapping',raw);out={}
    if isinstance(base,dict):
        for k,v in base.items():
            if isinstance(v,str):out[str(k)]=v
            elif isinstance(v,dict):
                sec=v.get('section') or v.get('Section') or v.get('type')
                if sec:out[str(k)]=str(sec)
    return out

def write_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def family(metric): return metric.split('_',1)[0]

def dists(arr,cents,scale,mask=None):
    if mask is None: mask=np.ones(len(arr),dtype=bool)
    aa=((arr[mask]-cents['A'][mask])/scale[mask])**2
    cc=((arr[mask]-cents['C'][mask])/scale[mask])**2
    return float(np.sqrt(np.mean(aa))),float(np.sqrt(np.mean(cc)))

def quad_residual(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    out=np.full(len(y),np.nan)
    if ok.sum()<5:return out
    X=np.c_[np.ones(ok.sum()),x[ok],x[ok]**2]
    beta=np.linalg.lstsq(X,y[ok],rcond=None)[0]
    out[ok]=y[ok]-X@beta
    return out

def score_layer(name,layer,prim,feat,qualified,refs,secmap):
    folios=m.folio_samples(layer);vf=m.feature_table(folios);m.add_F7(folios,vf,refs)
    metrics=[mm for mm in qualified if all(np.isfinite(vf[s.sid].get(mm,np.nan)) for s in folios)]
    med,scale,cents,clsarr=m.standard_class_model(prim,feat,metrics)
    qrows=[];frows=[]
    for s in folios:
        arr=np.array([vf[s.sid][mm] for mm in metrics],float)
        dA,dC=dists(arr,cents,scale); delta=dA-dC
        f=s.meta['folio'];sec=secmap.get(f) or secmap.get(f.lstrip('f')) or 'UNMAPPED';n=len(s.tokens());ln=math.log1p(n)
        qA=((arr-cents['A'])/scale)**2;qC=((arr-cents['C'])/scale)**2;qdiff=qA-qC
        frows.append({'layer':name,'folio':f,'section':sec,'n_tokens':n,'log_tokens':ln,'d_A':dA,'d_C':dC,'delta_C':delta,'mean_qdiff':float(np.mean(qdiff))})
        for j,mm in enumerate(metrics):
            mask=np.ones(len(metrics),dtype=bool);mask[j]=False
            la,lc=dists(arr,cents,scale,mask);loo=la-lc
            qrows.append({'layer':name,'folio':f,'section':sec,'n_tokens':n,'log_tokens':ln,'metric':mm,'family':family(mm),'value':float(arr[j]),'z_to_A':float((arr[j]-cents['A'][j])/scale[j]),'z_to_C':float((arr[j]-cents['C'][j])/scale[j]),'qdiff_A_minus_C':float(qdiff[j]),'delta_C_full':delta,'delta_C_without_metric':loo,'loo_support':float(delta-loo)})
    # length quartiles per layer
    toks=np.array([r['n_tokens'] for r in frows],float);qs=np.quantile(toks,[.25,.75]);q1max,q4min=qs
    summary=[]
    for mm in metrics:
        rr=[r for r in qrows if r['metric']==mm];short=[r for r in rr if r['n_tokens']<=q1max];long=[r for r in rr if r['n_tokens']>=q4min]
        summary.append({'layer':name,'metric':mm,'family':family(mm),'n':len(rr),'q1_n':len(short),'q4_n':len(long),
            'median_qdiff_all':float(np.median([r['qdiff_A_minus_C'] for r in rr])),'median_qdiff_q1':float(np.median([r['qdiff_A_minus_C'] for r in short])),'median_qdiff_q4':float(np.median([r['qdiff_A_minus_C'] for r in long])),
            'q1_minus_q4':float(np.median([r['qdiff_A_minus_C'] for r in short])-np.median([r['qdiff_A_minus_C'] for r in long])),
            'rho_qdiff_log_tokens':spearman([r['log_tokens'] for r in rr],[r['qdiff_A_minus_C'] for r in rr]),'rho_rawvalue_log_tokens':spearman([r['log_tokens'] for r in rr],[r['value'] for r in rr]),
            'median_loo_support_q1':float(np.median([r['loo_support'] for r in short])),'median_loo_support_q4':float(np.median([r['loo_support'] for r in long])),
            'q1_positive_qdiff_fraction':float(np.mean([r['qdiff_A_minus_C']>0 for r in short])),'q4_positive_qdiff_fraction':float(np.mean([r['qdiff_A_minus_C']>0 for r in long]))})
    # family decomposition
    famrows=[]
    for fam in sorted(set(family(mm) for mm in metrics)):
        fm=[mm for mm in metrics if family(mm)==fam]
        for quart,label in [(lambda r:r['n_tokens']<=q1max,'Q1'),(lambda r:r['n_tokens']>=q4min,'Q4')]:
            vals=[]
            for fr in frows:
                if not quart(fr):continue
                vals.append(np.mean([next(q['qdiff_A_minus_C'] for q in qrows if q['folio']==fr['folio'] and q['metric']==mm) for mm in fm]))
            famrows.append({'layer':name,'family':fam,'quartile':label,'n_metrics':len(fm),'median_mean_qdiff':float(np.median(vals)),'positive_fraction':float(np.mean(np.array(vals)>0))})
    # section residuals after flexible length trend in delta_C
    x=[r['log_tokens'] for r in frows];y=[r['delta_C'] for r in frows];res=quad_residual(x,y)
    for r,v in zip(frows,res):r['delta_C_length_residual']=float(v)
    secrows=[]
    for sec in sorted(set(r['section'] for r in frows)):
        rr=[r for r in frows if r['section']==sec]; vals=[r['delta_C_length_residual'] for r in rr]
        secrows.append({'layer':name,'section':sec,'n_folios':len(rr),'median_delta_C':float(np.median([r['delta_C'] for r in rr])),'median_length_residual':float(np.median(vals)),'mean_length_residual':float(np.mean(vals)),'positive_residual_fraction':float(np.mean(np.array(vals)>0))})
    return qrows,frows,summary,famrows,secrows

def main():
    rows=m.load_lecouteux();docs=m.load_A_docs();asp=m.split_A_sources(docs)
    samples=m.A_samples(docs,asp)+m.formula_samples(rows);blocks=m.aggregate_blocks(samples)
    prim=[s for s in samples if s.cls=='A']+[s for s in blocks if s.cls in ('B','C')]
    feat=m.feature_table(prim);refs=m.add_F7(prim,feat);tests,qualified=m.qualify(prim,feat)
    if qualified!=EXPECTED: raise RuntimeError(f'freeze mismatch {qualified!r}')
    print('EXTERNAL_FREEZE_OK',len(qualified),flush=True)
    raw=m.load_section_map(Path('../..'));secmap=normalize_section_map(raw)
    allq=[];allf=[];alls=[];allfam=[];allsec=[];meta={}
    with tempfile.TemporaryDirectory() as td0:
        td=Path(td0)
        for key,(url,want) in m.VOYNICH_SRC.items():
            fn=td/(key if '.' in key else key+'.txt');meta[key]=m.fetch_checked(url,want,fn)
        exe=td/'bitrans';subprocess.run(['gcc','-O2','-o',str(exe),str(td/'bitrans.c')],check=True)
        aa=td/'RF.aaa.txt';p=subprocess.run([str(exe),'-1','-m2','-f',str(td/'STA-aaa.bit'),str(td/'RF.txt'),str(aa)],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr[-1000:])
        layers=m.parse_rf_layers((td/'RF.txt').read_text(errors='replace'),aa.read_text(errors='replace'))
        for name,layer in layers.items():
            q,f,s,fam,sec=score_layer(name,layer,prim,feat,qualified,refs,secmap)
            allq+=q;allf+=f;alls+=s;allfam+=fam;allsec+=sec
            print('LAYER',name,'folios',len(f),'metrics',len(s),flush=True)
    write_csv(OUT/'metric_folio_contributions.csv',allq);write_csv(OUT/'folio_length_residuals.csv',allf);write_csv(OUT/'metric_driver_summary.csv',alls);write_csv(OUT/'family_driver_summary.csv',allfam);write_csv(OUT/'section_length_residuals.csv',allsec)
    # cross-layer metric stability
    cross=[]
    for mm in EXPECTED:
        rr=[r for r in alls if r['metric']==mm]
        cross.append({'metric':mm,'family':family(mm),'layers':len(rr),'q1_C_support_layers':sum(r['median_qdiff_q1']>0 for r in rr),'q4_C_support_layers':sum(r['median_qdiff_q4']>0 for r in rr),'length_negative_layers':sum(r['rho_qdiff_log_tokens']<-.3 for r in rr),'median_q1_minus_q4':float(np.median([r['q1_minus_q4'] for r in rr])),'median_rho_qdiff_log_tokens':float(np.median([r['rho_qdiff_log_tokens'] for r in rr])),'median_loo_support_q1':float(np.median([r['median_loo_support_q1'] for r in rr]))})
    write_csv(OUT/'metric_cross_layer_stability.csv',cross)
    top=sorted(cross,key=lambda r:(r['q1_C_support_layers'],r['median_loo_support_q1'],r['median_q1_minus_q4']),reverse=True)
    result={'status':'METRIC_DRIVER_AUDIT_COMPLETE','primary_verdict_unchanged':'NO_ROBUST_MAGIC_AFFINITY','qualified_metrics':EXPECTED,'source_hashes':meta,'top_short_folio_drivers':top,'interpretation_rule':'Positive qdiff means the metric is closer to C than A in squared standardized distance. Strong finite-size driver = Q1 C support across layers plus qdiff decreasing with log token count.','B_reference_warning':'A-vs-C only; B omitted from driver attribution because frozen B has one held-out test block.'}
    (OUT/'metric_driver_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True))
    print('DONE',OUT/'metric_driver_audit.json',flush=True)
if __name__=='__main__':main()
