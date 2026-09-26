#!/usr/bin/env python3
from __future__ import annotations
import json,re,importlib.util,math,random,statistics,sys
from pathlib import Path
import numpy as np

OUT=Path('/mnt/data/voynich_notation_v0_3');DATA=OUT/'gabc_pilot_corpus.json'
spec=importlib.util.spec_from_file_location('v02','/mnt/data/voynich_notation_falsification_v0_2.py');v=importlib.util.module_from_spec(spec);sys.modules['v02']=v;spec.loader.exec_module(v)
PITCH=set('abcdefghijklmABCDEFGHIJKLM')

def features(g):
    pitches=[c.lower() for c in g if c in PITCH]
    init=pitches[0] if pitches else 'NONE'
    dirs=[]
    for a,b in zip(pitches,pitches[1:]):
        d=ord(b)-ord(a);dirs.append('U' if d>0 else ('D' if d<0 else 'S'))
    contour=(dirs[0] if dirs else 'N')+('_'+str(min(4,len(pitches))))
    specials=''.join(sorted(set(c for c in g if c not in PITCH and c!='/')))
    mod=(specials or 'PLAIN')+'_N'+str(min(3,g.count('/')+1))
    core=''.join(pitches) or 'EMPTY'
    return init,contour,core,mod

def build():
    docs=json.load(open(DATA)); rec=[]; lines=[]
    for d in docs:
        groups=[g for g in re.findall(r'\(([^()]*)\)',d['body']) if not re.fullmatch(r'[cf]\d',g)]
        line=[]
        for i,g in enumerate(groups):
            p,ga,c,s=features(g)
            r={'token':g,'prefix':p,'gallows':ga,'m_core':c,'sfx_fam':s,'section':d['family'],'folio':d['id'],'line_no':1,'pos':i,'line_len':len(groups)}
            rec.append(r);line.append(len(rec)-1)
        lines.append(np.array(line,dtype=np.int32))
    return docs,rec,lines

def score_doc(d,train,test,K,seed):
    st=v.static_fit(d,train)
    iid=v.fit_iid(d,train,K,seed,max_iter=50,tol=1e-7)
    hmm=v.fit_hmm(d,train,K,seed,max_iter=50,tol=1e-7)
    li,n=v.score_latent(d,iid,test,'iid',st);lh,_=v.score_latent(d,hmm,test,'hmm',st)
    return {'iid_bpt':-li/math.log(2)/n,'hmm_bpt':-lh/math.log(2)/n,'hmm_gain':(lh-li)/math.log(2)/n,'n':n}

def main():
    docs,rec,all_lines=build();d=v.prepare(rec);rows=[]
    for K in (3,4,6):
      for j,doc in enumerate(docs):
        train=[l for i,l in enumerate(d.lines) if i!=j];test=[d.lines[j]];seed=7000+K*100+j
        st=v.static_fit(d,train);iid=v.fit_iid(d,train,K,seed,max_iter=20,tol=1e-5);hmm=v.fit_hmm(d,train,K,seed,max_iter=20,tol=1e-5)
        li,n=v.score_latent(d,iid,test,'iid',st);lh,_=v.score_latent(d,hmm,test,'hmm',st)
        z={'K':K,'heldout':doc['id'],'family':doc['family'],'shuffle':False,'iid_bpt':-li/math.log(2)/n,'hmm_bpt':-lh/math.log(2)/n,'hmm_gain':(lh-li)/math.log(2)/n,'n':n};rows.append(z)
        q=d.lines[j].copy();random.Random(9000+j).shuffle(q)
        li2,n2=v.score_latent(d,iid,[q],'iid',st);lh2,_=v.score_latent(d,hmm,[q],'hmm',st)
        rows.append({'K':K,'heldout':doc['id'],'family':doc['family'],'shuffle':True,'iid_bpt':-li2/math.log(2)/n2,'hmm_bpt':-lh2/math.log(2)/n2,'hmm_gain':(lh2-li2)/math.log(2)/n2,'n':n2})
        print(K,doc['id'],round(z['hmm_gain'],3),flush=True)
    summary=[]
    for K in (3,4,6):
      for sh in (False,True):
        rr=[x for x in rows if x['K']==K and x['shuffle']==sh]
        summary.append({'K':K,'shuffle':sh,'docs':len(rr),'tokens':sum(x['n'] for x in rr),'weighted_hmm_gain':sum(x['hmm_gain']*x['n'] for x in rr)/sum(x['n'] for x in rr),'mean_doc_gain':statistics.mean(x['hmm_gain'] for x in rr),'positive_docs':sum(x['hmm_gain']>0 for x in rr)})
    obj={'schema':'historical-gabc-notation-pilot-v0.3','source_docs':docs,'feature_definition':{'prefix':'initial pitch location','gallows':'first contour direction plus pitch-count bucket','core':'full pitch-location sequence','suffix_family':'graphical modifier signature plus neume-count bucket'},'rows':rows,'summary':summary}
    json.dump(obj,open(OUT/'historical_gabc_pilot_results_v0_3.json','w'),indent=2)
    print(summary)
if __name__=='__main__':main()
