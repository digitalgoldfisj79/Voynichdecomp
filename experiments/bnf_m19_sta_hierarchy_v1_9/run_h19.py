#!/usr/bin/env python3
import os,urllib.request,hashlib,math,json,tempfile
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/15e1cfa0e37119907d6a99ba6b2e2be1c4730fa6/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py'
b={'__name__':'v19hbase'}
exec(compile(urllib.request.urlopen(BASE,timeout=120).read().decode(),'run_v17.py','exec'),b)
REP=os.environ['M19_REP']
EXPECTED={'family':22,'aaa':26,'sta':36}
if REP not in EXPECTED: raise RuntimeError(('M19_REP',REP))
K=EXPECTED[REP]
NS='M19STAv19H1'
def sd(*p):return int.from_bytes(hashlib.sha256(('::'.join([NS]+list(map(str,p)))).encode()).digest()[:8],'big')&0xffffffff

def one_restart(S,comp,la,ens,rr):
    rng=np.random.default_rng(sd('opt',REP,la,ens,rr));m=b['init_map'](K,rng);s=b['score_num'](S,m,comp);ds=[]
    for _ in range(64):
        x,ch=b['proposal'](m,rng);ds.append(abs(b['score_num'](S,x,comp)-s))
    t0=max(1e-6,float(np.median(ds))*4);local=(s,m.copy())
    for k in range(100000):
        frac=k/99999.;temp=max(1e-8,t0*(0.003**frac));x,ch=b['proposal'](m,rng);s2=b['score_num'](S,x,comp);d=s2-s
        if d>=0 or rng.random()<math.exp(max(-60,d/temp)):
            m=x;s=s2
            if s>local[0]:local=(s,m.copy())
    m=local[1].copy();s=b['score_num'](S,m,comp)
    for rounds in range(30):
        bd=1e-14;bx=None;cnt=np.bincount(m,minlength=b['NV'])
        for a in range(K):
            for c in range(a+1,K):
                if m[a]==m[c]:continue
                x=m.copy();x[a],x[c]=x[c],x[a];s2=b['score_num'](S,x,comp);d=s2-s
                if d>bd:bd=d;bx=x
        if np.any(cnt==2) and np.any(cnt==1):
            for sv in np.flatnonzero(cnt==2):
                for dv in np.flatnonzero(cnt==1):
                    for i in np.flatnonzero(m==sv):
                        x=m.copy();x[i]=dv;s2=b['score_num'](S,x,comp);d=s2-s
                        if d>bd:bd=d;bx=x
        if bx is None:break
        m=bx;s=b['score_num'](S,m,comp)
    return s,m

def paired_fit(S,comp,la):
    best={'A':(-1e100,None),'B':(-1e100,None)};history=[]
    for batch in range(4):
        for ens in ['A','B']:
            for j in range(6):
                rr=batch*6+j;s,m=one_restart(S,comp,la,ens,rr)
                if s>best[ens][0]:best[ens]=(s,m.copy())
        gap=abs(best['A'][0]-best['B'][0]);agr=b['agreement'](S['freq'],best['A'][1],best['B'][1])
        row={'restarts_per_ensemble':(batch+1)*6,'scoreA':best['A'][0],'scoreB':best['B'][0],'score_gap':gap,'agreement':agr}
        history.append(row);print('H19_BATCH',REP,la,json.dumps(row,separators=(',',':')),flush=True)
        if gap<=1e-7 and agr>=.95:break
    conv=history[-1]['score_gap']<=1e-7 and history[-1]['agreement']>=.95
    winner=best['A'] if best['A'][0]>=best['B'][0] else best['B']
    return winner[0],winner[1],history[-1]['agreement'],conv,history

# Acquire exactly the frozen v1.7 sources and derive exactly the frozen RF split/vocab.
td=tempfile.mkdtemp(prefix='m19h19_');paths,source_meta=b['acquire_sources'](td)
st=open(paths['RF'],encoding='utf-8').read();aa=open(paths['RF_aaa'],encoding='utf-8').read()
raw={
 'family':b['parse_sta'](st,'family',False),
 'sta':b['parse_sta'](st,'sta',False),
 'aaa':b['parse_aaa'](aa,False),
}
T,H,C,allf=b['split_pages'](raw['sta'])
Cnt=b['count_tokens'](raw[REP],T,REP);v,selcov=b['choose_vocab'](Cnt,REP)
if len(v)!=K:raise RuntimeError(('K mismatch',REP,len(v),K,v))
Tw,Tby,Tcov=b['project'](raw[REP],T,REP,v);Hw,Hby,Hcov=b['project'](raw[REP],H,REP,v)
print('H19_META',json.dumps({'rep':REP,'K':K,'Tfolios':len(T),'Hfolios':len(H),'Cfolios_sealed':len(C),'selection_coverage':selcov,'Tcov':Tcov,'Hcov':Hcov,'vocab':v},separators=(',',':')),flush=True)

lms,lmmeta=b['build_lms']();comps={la:b['ns']['induced'](lms[la]) for la in b['LANGS']};S=b['stats'](Tw,v)
fits={};agreements={};converged={};histories={};train_scores={}
for la in b['LANGS']:
    s,m,agr,conv,hist=paired_fit(S,comps[la],la);fits[la]=m;agreements[la]=agr;converged[la]=conv;histories[la]=hist;train_scores[la]=s
    print('H19_FIT',REP,la,'score',s,'agreement',agr,'converged',conv,flush=True)
rank=[]
for la,m in fits.items():
    fw,n=b['forward'](Hw,m,v,lms[la]);rank.append((la,fw,n))
rank.sort(key=lambda x:x[1],reverse=True);top=rank[0][0];margin=rank[0][1]-rank[1][1]
gate=Hcov['coverage']>=.97 and margin>=.05 and agreements[top]>=.90 and converged[top] and all(converged.values())
out={'rep':REP,'K':K,'top':top,'margin':margin,'coverage':Hcov['coverage'],'word_coverage':Hcov['word_coverage'],'top_agreement':agreements[top],'top_converged':converged[top],'all_converged':all(converged.values()),'gate':gate,'ranking':rank,'agreements':agreements,'converged':converged,'train_scores':train_scores,'candidate_map_indices':[int(x) for x in fits[top]],'candidate_map_values':{v[i]:int(b['VALUES'][int(fits[top][i])]) for i in range(K)},'vocab':v,'Tfolios':T,'Hfolios':H,'Cfolios_sha256':hashlib.sha256(('\n'.join(C)).encode()).hexdigest(),'source_meta':source_meta,'lm_meta':lmmeta,'histories':histories}
print('H19_RESULT='+json.dumps(out,separators=(',',':')),flush=True)
