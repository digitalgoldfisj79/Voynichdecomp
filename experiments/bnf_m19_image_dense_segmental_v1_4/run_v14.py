#!/usr/bin/env python3
import urllib.request,json,hashlib,os
import numpy as np
from scipy.optimize import linear_sum_assignment

U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7b97e064c1098d63158a9a406780999aca91103d/experiments/bnf_m19_image_bridge_v1_2/run_arm_b.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
ns={'__name__':'v14lib'};exec(compile(src,'run_arm_b.py','exec'),ns)
b=ns['b'];K=ns['K']

# Dense raw loader. Text/EVA fields are never retained.
def load_dense():
    sel=[];folios=set();p=os.path.join(b['DATA'],'corpus_crop_manifest.jsonl')
    with open(p) as h:
        for rowi,line in enumerate(h):
            r=json.loads(line)
            if r.get('kind')=='ccmerge' and r.get('view')=='norm' and not r.get('low_conf',False):
                sel.append((rowi,r['id'],r['folio'],int(r['word_index']),int(r['slot']),int(r['n_slots'])));folios.add(r['folio'])
    idx=np.array([q[0] for q in sel],dtype=np.int64);z=np.load(os.path.join(b['DATA'],'corpus_embeddings_full_dense.npz'),allow_pickle=False);ids=z['ids'];checks=np.linspace(0,len(sel)-1,min(1000,len(sel)),dtype=int)
    for j in checks:
        q=sel[j]
        if ids[q[0]]!=q[1]+'::norm':raise RuntimeError(('dense order',j))
    X=np.asarray(z['vectors'][idx],dtype=np.float32);del ids,z
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    rec={'folio':np.array([q[2] for q in sel],dtype=object),'word':np.array([q[3] for q in sel],np.int32),'slot':np.array([q[4] for q in sel],np.int16),'nslots':np.array([q[5] for q in sel],np.int16)}
    folios=sorted(folios,key=lambda f:hashlib.sha256(('M19IMAGEv12split::'+f).encode()).digest());nt=round(.5*len(folios));nh=round(.2*len(folios));T=folios[:nt];H=folios[nt:nt+nh];C=folios[nt+nh:]
    tv=sorted(T,key=lambda f:hashlib.sha256(('M19IMAGEv12vis::'+f).encode()).digest());cut=round(.8*len(tv));split={'T':set(T),'H':set(H),'C':set(C),'Tf':set(tv[:cut]),'Tv':set(tv[cut:])}
    print('V14_DENSE_CENSUS',json.dumps({'rows':len(sel),'folios':len(folios),'T':len(T),'H':len(H),'C':len(C),'Tfit':len(split['Tf']),'Tvis':len(split['Tv'])},separators=(',',':')),flush=True)
    return X,rec,split
b['load_image_data']=load_dense
b['folio_center']=lambda X,rec:X  # v1.4 prospectively fixes raw dense, not folio-centred.

# Enforce all frozen segmental qualification gates including numerical-map recovery.
def segmental_qualify_fixed(lam,lms,pools,comps):
    rows=[]
    for la in b['QUAL']:
        span=b['choose_span'](pools[la],b['QTRAIN']+b['QHOLD'],('image-v14-segqual',la));trtxt,hotxt=b['split_text_letters'](span,b['QTRAIN'])
        trw,truth,P=ns['synth_micro'](trtxt,la,lam);A,widx=ns['array_words'](trw);cent=ns['fit_segmental'](A,widx,K,lam,408);segtr=ns['segment_words'](A,widx,cent,lam);bf=ns['boundary_f1_truth'](segtr,truth)
        how,htruth,PH=ns['synth_micro'](hotxt,la,lam);assert np.allclose(P,PH);AH,hidx=ns['array_words'](how);segh=ns['segment_words'](AH,hidx,cent,lam);bfh=ns['boundary_f1_truth'](segh,htruth)
        rr,cc=linear_sum_assignment(-(cent@P.T));true=np.full(K,-1,np.int16)
        for r,c in zip(rr,cc):true[int(r)]=int(c)
        tw=[[int(x[3]) for x in ss] for _,_,ss in segtr];hw=[[int(x[3]) for x in ss] for _,_,ss in segh];S=b['sym_stats'](tw,K);rank=[];fits={}
        for cand in b['LANGS']:
            sc,m=b['optimize'](S,comps[cand],K,('v14segqual',la,cand,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);fw,_=b['forward_sequences'](hw,m,lms[cand]);rank.append((cand,fw));fits[cand]=m
        rank.sort(key=lambda x:x[1],reverse=True);_,m2=b['optimize'](S,comps[la],K,('v14segqual2',la,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);agr=b['agreement'](S['freq'],fits[la],m2);acc=b['weighted_acc'](S['freq'],fits[la],true)
        r={'lang':la,'top':rank[0][0],'margin':rank[0][1]-rank[1][1],'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'train_boundary_f1':bf,'hold_boundary_f1':bfh,'mapping_acc':acc,'agreement':agr};rows.append(r);print('V14_QUAL',json.dumps(r,separators=(',',':')),flush=True)
    gate={'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'min_boundary_f1':min(min(r['train_boundary_f1'],r['hold_boundary_f1']) for r in rows),'median_acc':float(np.median([r['mapping_acc'] for r in rows])),'min_acc':min(r['mapping_acc'] for r in rows),'min_agreement':min(r['agreement'] for r in rows)}
    gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['min_boundary_f1']>=.85 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90
    print('V14_QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);return rows,gate
ns['segmental_qualify']=segmental_qualify_fixed
ns['main']()
