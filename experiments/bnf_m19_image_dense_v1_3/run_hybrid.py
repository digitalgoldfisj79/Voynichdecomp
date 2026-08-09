#!/usr/bin/env python3
import urllib.request,json,hashlib,os
import numpy as np

U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d4675efc01928ffac79ea605dc223628092cbbda/experiments/bnf_m19_image_bridge_v1_2/run_arm_a.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
ns={'__name__':'arm_a_hybrid_lib'};exec(compile(src,'run_arm_a.py','exec'),ns)

def load_hybrid():
    sel=[];folios=set();p=os.path.join(ns['DATA'],'corpus_crop_manifest.jsonl')
    with open(p) as h:
        for rowi,line in enumerate(h):
            r=json.loads(line)
            if r.get('kind')=='ccmerge' and r.get('view')=='norm' and not r.get('low_conf',False):
                sel.append((rowi,r['id'],r['folio'],int(r['word_index']),int(r['slot']),int(r['n_slots'])));folios.add(r['folio'])
    idx=np.array([q[0] for q in sel],dtype=np.int64);checks=np.linspace(0,len(sel)-1,min(1000,len(sel)),dtype=int)
    # Load selected dense then CLS vectors; textual/EVA fields were never retained.
    zd=np.load(os.path.join(ns['DATA'],'corpus_embeddings_full_dense.npz'),allow_pickle=False);idd=zd['ids']
    for j in checks:
        q=sel[j]
        if idd[q[0]]!=q[1]+'::norm':raise RuntimeError(('dense order',j))
    D=np.asarray(zd['vectors'][idx],dtype=np.float32);del idd,zd
    D/=np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-12)
    zc=np.load(os.path.join(ns['DATA'],'corpus_embeddings_full.npz'),allow_pickle=False);idc=zc['ids']
    for j in checks:
        q=sel[j]
        if idc[q[0]]!=q[1]+'::norm':raise RuntimeError(('cls order',j))
    C=np.asarray(zc['vectors'][idx],dtype=np.float32);del idc,zc
    C/=np.maximum(np.linalg.norm(C,axis=1,keepdims=True),1e-12)
    X=C+D;del C,D;X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    rec={'folio':np.array([q[2] for q in sel],dtype=object),'word':np.array([q[3] for q in sel],np.int32),'slot':np.array([q[4] for q in sel],np.int16),'nslots':np.array([q[5] for q in sel],np.int16)}
    folios=sorted(folios,key=lambda f:hashlib.sha256(('M19IMAGEv12split::'+f).encode()).digest());nt=round(.5*len(folios));nh=round(.2*len(folios));T=folios[:nt];H=folios[nt:nt+nh];Cset=folios[nt+nh:]
    tv=sorted(T,key=lambda f:hashlib.sha256(('M19IMAGEv12vis::'+f).encode()).digest());cut=round(.8*len(tv));split={'T':set(T),'H':set(H),'C':set(Cset),'Tf':set(tv[:cut]),'Tv':set(tv[cut:])}
    print('HYBRID_IMAGE_CENSUS',json.dumps({'rows':len(sel),'folios':len(folios),'T':len(T),'H':len(H),'C':len(Cset),'Tfit':len(split['Tf']),'Tvis':len(split['Tv'])},separators=(',',':')),flush=True)
    return X,rec,split

ns['load_image_data']=load_hybrid
ns['main']()
