#!/usr/bin/env python3
import argparse, json, math, unicodedata, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np

SEED=20260813
VMS={'ED1_N0':1.164213760,'ED1_N1':1.103860531,'ED1_N3':1.025668049,'E1_N0':1.01819,'E2_N0':1.217303154,'E2_N1':1.080333687,'E2_N3':1.064529935}
FEATURES=['ED1_N0','ED1_N1','ED1_N3','E1_N0','E2_N0','E2_N1','E2_N3']

def has_letter(s): return any(ch.isalpha() for ch in s)
def norm(s): return unicodedata.normalize('NFC',s).lower()
def ed1(a,b):
    if a==b:return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1:return False
    if la==lb:return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:a,b=b,a;la,lb=lb,la
    i=j=0;sk=0
    while i<la and j<lb:
        if a[i]==b[j]:i+=1;j+=1
        else:
            sk+=1;j+=1
            if sk>1:return False
    return True

def groups(n,null):
    if null=='N0': return [np.arange(n,dtype=np.int32)]
    if null=='N1':
        if n<5:return [g for g in (np.arange(min(2,n),dtype=np.int32),np.arange(min(2,n),n,dtype=np.int32)) if len(g)]
        return [np.arange(0,2,dtype=np.int32),np.arange(2,n-2,dtype=np.int32),np.arange(n-2,n,dtype=np.int32)]
    raise ValueError(null)

def parse_llct(path):
    root=ET.parse(path).getroot(); units=[]; duplicate_nodes=0; sentence_count=0; seg_counts=Counter(); run_counts=Counter()
    for e in root.iter():
        if e.tag.split('}')[-1]!='LM' or 'document_id' not in e.attrib: continue
        sentence_count+=1;seen={}
        for d in e.iter():
            if d is e or d.tag.split('}')[-1]!='LM' or 'form' not in d.attrib: continue
            a=dict(d.attrib)
            try:i=int(a['id'])
            except:continue
            if i in seen:
                duplicate_nodes+=1;prev=seen[i]
                for k in ('form','lemma','seg'):
                    if prev.get(k)!=a.get(k):raise RuntimeError(f'conflicting duplicate token id sentence={e.attrib.get("id")} id={i} {k}')
                continue
            seen[i]=a;seg_counts[a.get('seg','')]+=1
        ordered=[seen[i] for i in sorted(seen)];charter=f"{e.attrib.get('document_id','')}:{e.attrib.get('subdoc','')}";cur_seg=None;cur=[]
        def flush():
            nonlocal cur_seg,cur
            if cur_seg in ('formulaic','free') and cur:
                units.append({'group':'F' if cur_seg=='formulaic' else 'R','tokens':cur,'charter':charter});run_counts[cur_seg]+=1
            cur_seg=None;cur=[]
        for a in ordered:
            s=a.get('seg','')
            if s=='subs' or s not in ('formulaic','free'):flush();continue
            if cur_seg is not None and s!=cur_seg:flush()
            if cur_seg is None:cur_seg=s
            f=a.get('form','')
            if has_letter(f):cur.append(norm(f))
        flush()
    return units,{'sentences':sentence_count,'duplicate_xml_nodes_collapsed':duplicate_nodes,'seg_nodes':dict(seg_counts),'runs':dict(run_counts)}

def prep_unit(u):
    toks=u['tokens'];uniq=list(dict.fromkeys(toks));mp={t:i for i,t in enumerate(uniq)};ids=np.asarray([mp[t] for t in toks],dtype=np.int16 if len(uniq)<32767 else np.int32);R=np.zeros((len(uniq),len(uniq)),dtype=np.bool_)
    for i,a in enumerate(uniq):
        for j in range(i+1,len(uniq)):
            if ed1(a,uniq[j]):R[i,j]=R[j,i]=True
    return {'ids':ids,'R':R,'charter':u['charter'],'group':u['group'],'tokens':toks}

def score_matrix(mat,R,interior_ed1=False,interior_e2=False):
    P,n=mat.shape;out={}
    if n>=2:
        s=np.arange(2,n-3,dtype=np.int32) if interior_ed1 and n>=6 else (np.array([],dtype=np.int32) if interior_ed1 else np.arange(n-1,dtype=np.int32));out['ED1']=R[mat[:,s],mat[:,s+1]].sum(axis=1,dtype=np.int32) if len(s) else np.zeros(P,dtype=np.int32);out['E1']=(mat[:,s]==mat[:,s+1]).sum(axis=1,dtype=np.int32) if len(s) else np.zeros(P,dtype=np.int32)
    else:out['ED1']=out['E1']=np.zeros(P,dtype=np.int32)
    if n>=3:
        s2=np.arange(2,n-4,dtype=np.int32) if interior_e2 and n>=7 else (np.array([],dtype=np.int32) if interior_e2 else np.arange(n-2,dtype=np.int32));out['E2']=(mat[:,s2]==mat[:,s2+2]).sum(axis=1,dtype=np.int32) if len(s2) else np.zeros(P,dtype=np.int32)
    else:out['E2']=np.zeros(P,dtype=np.int32)
    return out

def simulate(units,null,P,seed):
    rng=np.random.default_rng(seed);totals={k:np.zeros(P,dtype=np.int64) for k in ['ED1','E1','E2','ED1_N3','E2_N3']}
    for u in units:
        ids,R=u['ids'],u['R'];n=len(ids)
        if n<2:continue
        mat=np.broadcast_to(ids,(P,n)).copy()
        for g in groups(n,null):
            if len(g)>1:
                order=np.argsort(rng.random((P,len(g))),axis=1);vals=mat[:,g].copy();mat[:,g]=np.take_along_axis(vals,order,axis=1)
        sc=score_matrix(mat,R)
        for k in ('ED1','E1','E2'):totals[k]+=sc[k]
        if null=='N1':
            sc3=score_matrix(mat,R,True,True);totals['ED1_N3']+=sc3['ED1'];totals['E2_N3']+=sc3['E2']
    return totals

def actual(units):
    totals=Counter();den=Counter()
    for u in units:
        ids,R=u['ids'],u['R'];n=len(ids)
        if n>=2:
            s=np.arange(n-1);totals['ED1']+=int(R[ids[s],ids[s+1]].sum());totals['E1']+=int((ids[s]==ids[s+1]).sum());den['adj']+=len(s);s3=np.arange(2,n-3) if n>=6 else np.array([],dtype=int);totals['ED1_N3']+=int(R[ids[s3],ids[s3+1]].sum()) if len(s3) else 0;den['adj_n3']+=len(s3)
        if n>=3:
            s2=np.arange(n-2);totals['E2']+=int((ids[s2]==ids[s2+2]).sum());den['e2']+=len(s2);s23=np.arange(2,n-4) if n>=7 else np.array([],dtype=int);totals['E2_N3']+=int((ids[s23]==ids[s23+2]).sum()) if len(s23) else 0;den['e2_n3']+=len(s23)
    return totals,den

def summ(obs,arr,denom):
    arr=np.asarray(arr,dtype=float);mu=arr.mean();sd=arr.std(ddof=1);return {'actual':int(obs),'denom':int(denom),'null_mean':float(mu),'null_sd':float(sd),'ratio':float(obs/mu) if mu else None,'z':float((obs-mu)/sd) if sd else None}
def primary(units,seed):
    obs,den=actual(units);n0=simulate(units,'N0',500,seed+1);n1=simulate(units,'N1',500,seed+2)
    return {'ED1_N0':summ(obs['ED1'],n0['ED1'],den['adj']),'ED1_N1':summ(obs['ED1'],n1['ED1'],den['adj']),'ED1_N3':summ(obs['ED1_N3'],n1['ED1_N3'],den['adj_n3']),'E1_N0':summ(obs['E1'],n0['E1'],den['adj']),'E2_N0':summ(obs['E2'],n0['E2'],den['e2']),'E2_N1':summ(obs['E2'],n1['E2'],den['e2']),'E2_N3':summ(obs['E2_N3'],n1['E2_N3'],den['e2_n3'])}

def relation_expectation(ids,R,kind,null,interior=False):
    n=len(ids)
    if kind in ('ED1','E1'):starts=np.arange(2,n-3) if interior and n>=6 else (np.array([],dtype=int) if interior else np.arange(max(0,n-1)));lag=1
    else:starts=np.arange(2,n-4) if interior and n>=7 else (np.array([],dtype=int) if interior else np.arange(max(0,n-2)));lag=2
    if not len(starts):return 0.0
    rel=R if kind=='ED1' else np.eye(R.shape[0],dtype=float);gs=groups(n,null);cmap=np.empty(n,dtype=int)
    for gi,g in enumerate(gs):cmap[g]=gi
    cc=[np.bincount(ids[g],minlength=R.shape[0]).astype(float) for g in gs];ans=0.0
    for p in starts:
        q=p+lag;a,b=cmap[p],cmap[q];ca,cb=cc[a],cc[b]
        if a==b:
            m=ca.sum()
            if m<2:continue
            ans+=float(ca@rel@ca-np.sum(ca*np.diag(rel)))/(m*(m-1))
        else:
            ma,mb=ca.sum(),cb.sum()
            if ma and mb:ans+=float(ca@rel@cb)/(ma*mb)
    return ans

def per_charter(units):
    rows=defaultdict(lambda:{g:{f:[0.0,0.0] for f in FEATURES} for g in ('F','R')})
    for u in units:
        ids,R=u['ids'],u['R'];g=u['group'];c=u['charter'];n=len(ids);obs_ed1=obs_e1=obs_ed13=obs_e2=obs_e23=0.0
        if n>=2:
            s=np.arange(n-1);obs_ed1=float(R[ids[s],ids[s+1]].sum());obs_e1=float((ids[s]==ids[s+1]).sum());s3=np.arange(2,n-3) if n>=6 else np.array([],dtype=int);obs_ed13=float(R[ids[s3],ids[s3+1]].sum()) if len(s3) else 0
        if n>=3:
            s2=np.arange(n-2);obs_e2=float((ids[s2]==ids[s2+2]).sum());s23=np.arange(2,n-4) if n>=7 else np.array([],dtype=int);obs_e23=float((ids[s23]==ids[s23+2]).sum()) if len(s23) else 0
        vals={'ED1_N0':(obs_ed1,relation_expectation(ids,R,'ED1','N0')),'ED1_N1':(obs_ed1,relation_expectation(ids,R,'ED1','N1')),'ED1_N3':(obs_ed13,relation_expectation(ids,R,'ED1','N1',True)),'E1_N0':(obs_e1,relation_expectation(ids,R,'E1','N0')),'E2_N0':(obs_e2,relation_expectation(ids,R,'E2','N0')),'E2_N1':(obs_e2,relation_expectation(ids,R,'E2','N1')),'E2_N3':(obs_e23,relation_expectation(ids,R,'E2','N1',True))}
        for f,(o,x) in vals.items():rows[c][g][f][0]+=o;rows[c][g][f][1]+=x
    return rows

def ratios_from_rows(rows,chosen=None):
    if chosen is None:chosen=list(rows)
    out={g:{} for g in ('F','R')}
    for g in ('F','R'):
        for f in FEATURES:
            o=e=0.0
            for c in chosen:o+=rows[c][g][f][0];e+=rows[c][g][f][1]
            out[g][f]=o/e if e>0 else float('nan')
    return out

def distance(v):return float(np.linalg.norm(np.array([math.log(VMS[f]) for f in FEATURES])-np.array([math.log(v[f]) for f in FEATURES])))
def bootstrap(rows,seed,nboot=1000):
    rng=np.random.default_rng(seed);cs=list(rows);point=ratios_from_rows(rows);dF=distance(point['F']);dR=distance(point['R']);deltas=[];less=0
    for _ in range(nboot):
        chosen=rng.choice(cs,size=len(cs),replace=True).tolist();r=ratios_from_rows(rows,chosen);df,dr=distance(r['F']),distance(r['R']);less+=df<dr
        if r['F']['ED1_N0']>0 and r['R']['ED1_N0']>0:deltas.append(math.log(r['F']['ED1_N0'])-math.log(r['R']['ED1_N0']))
    delta0=math.log(point['F']['ED1_N0'])-math.log(point['R']['ED1_N0']);lo,hi=np.quantile(deltas,[.025,.975]);H2='SUPPORT' if dF<=.9*dR and less/nboot>=.95 else ('OPPOSITE' if dR<=.9*dF and (1-less/nboot)>=.95 else 'UNRESOLVED');H3='SUPPORT' if delta0>0 and lo>0 else ('OPPOSITE' if delta0<0 and hi<0 else 'UNRESOLVED')
    return {'analytic_point_ratios':point,'d_F':dF,'d_R':dR,'boot_p_dF_lt_dR':less/nboot,'delta_ED1':delta0,'delta_ED1_ci95':[float(lo),float(hi)],'H2':H2,'H3':H3}

def calibrate(units,null,seed):
    sims=simulate(units,null,4020,seed);keys=['ED1'] if null=='N0' else ['ED1','ED1_N3'];out={}
    for key in keys:
        arr=sims[key].astype(float);reps=[]
        for r in range(20):
            x=arr[r];ref=arr[20+200*r:20+200*(r+1)];mu=ref.mean();sd=ref.std(ddof=1);reps.append((x/mu if mu else float('nan'),(x-mu)/sd if sd else float('nan')))
        mr=float(np.nanmean([x[0] for x in reps]));nz=sum(abs(x[1])>=2 for x in reps if math.isfinite(x[1]));out[key]={'mean_ratio':mr,'n_abs_z_ge2':nz,'pass':bool(.95<=mr<=1.05 and nz<=2)}
    return out

def permute_raw(raw,null,seed):
    rng=np.random.default_rng(seed);out=[]
    for u in raw:
        a=np.array(u['tokens'],dtype=object)
        for g in groups(len(a),null):
            if len(g)>1:a[g]=rng.permutation(a[g])
        out.append({'group':u['group'],'charter':u['charter'],'tokens':a.tolist()})
    return out

def inject_c1(raw,frac,seed):
    rng=np.random.default_rng(seed);out=[{'group':u['group'],'charter':u['charter'],'tokens':u['tokens'][:]} for u in raw];eligible=[i for i,u in enumerate(out) if len(u['tokens'])>=6];rng.shuffle(eligible);k=max(1,int(round(frac*len(eligible))));chosen=eligible[:k]
    for ui in chosen:
        t=out[ui]['tokens'];i=int(rng.integers(2,len(t)-3));t[i+1]=t[i]+'§'
    return out,k,len(eligible)
def c1_known_answer(raw,seed):
    pseudo=permute_raw(raw,'N1',seed);inj,k,eligible=inject_c1(pseudo,.02,seed+1);p=[prep_unit(u) for u in inj];obs,den=actual(p);sim=simulate(p,'N1',500,seed+2);st=summ(obs['ED1_N3'],sim['ED1_N3'],den['adj_n3']);return {'eligible_loci':eligible,'injected_loci':k,'stat':st,'pass':bool(st['ratio'] is not None and st['ratio']>=1.15 and st['z'] is not None and st['z']>=3)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('xml');ap.add_argument('--out',default='RESULTS_llct_formulaic_profile_v0_1.json');args=ap.parse_args();raw,meta=parse_llct(args.xml);prepared=[prep_unit(u) for u in raw];gs={g:[u for u in prepared if u['group']==g] for g in ('F','R')};meta['analysis_units']={g:len(gs[g]) for g in gs};meta['lexical_tokens']={g:int(sum(len(u['ids']) for u in gs[g])) for g in gs};meta['charters']=len(set(u['charter'] for u in prepared));print('META',meta,flush=True)
    controls={}
    for gi,g in enumerate(('F','R')):
        controls[g]={'N0':calibrate(gs[g],'N0',SEED+10000+gi*1000),'N1':calibrate(gs[g],'N1',SEED+20000+gi*1000)};print('CTRL',g,controls[g],flush=True)
    controls['C1']=c1_known_answer(raw,SEED+25000);print('C1',controls['C1'],flush=True);results={g:primary(gs[g],SEED+30000+i*1000) for i,g in enumerate(('F','R'))};print('PRIMARY',json.dumps(results),flush=True);rows=per_charter(prepared);boot=bootstrap(rows,SEED+40000,1000);print('BOOT',boot,flush=True);c0ok=all(x['pass'] for g in ('F','R') for n in controls[g].values() for x in n.values());valid=bool(c0ok and controls['C1']['pass']);boot['H2_literal']=boot['H2'] if valid else 'INVALID_CONTROL';boot['H3_literal']=boot['H3'] if valid else 'INVALID_CONTROL';out={'seed':SEED,'meta':meta,'controls':controls,'primary':results,'bootstrap':boot,'valid':valid,'vms_reference':VMS,'features':FEATURES};json.dump(out,open(args.out,'w'),indent=2);print('WROTE',args.out,flush=True)
if __name__=='__main__':main()
