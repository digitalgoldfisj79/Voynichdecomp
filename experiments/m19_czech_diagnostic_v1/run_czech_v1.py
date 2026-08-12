#!/usr/bin/env python3
import argparse, collections, hashlib, json, math, os, tempfile, urllib.request
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/15e1cfa0e37119907d6a99ba6b2e2be1c4730fa6/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py'
CZECH_COMMIT='798f89716ae5a96e86042df7d394d56787e2e213'
CZ_TRAIN=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/{CZECH_COMMIT}/cs_cac-ud-train.conllu'
CZ_DEV=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/{CZECH_COMMIT}/cs_cac-ud-dev.conllu'
CZ_TEST=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-CAC/{CZECH_COMMIT}/cs_cac-ud-test.conllu'
NS='M19_CZECH_DIAGNOSTIC_V1_1'
ORIGINAL_RANK=[
 ('spanish',-2.5711088593),('french',-2.5934305936),('greek',-2.6259873716),
 ('german',-2.6319475217),('latin',-2.6527180797),('hebrew',-2.6835728413),
 ('italian',-2.7021448099),('arabic',-2.7801834577)]
ORIGINAL_COVERAGE=0.9983818770

def sd(*p):
    return int.from_bytes(hashlib.sha256(('::'.join([NS]+list(map(str,p)))).encode()).digest()[:8],'big')&0xffffffff

def load_base():
    b={'__name__':'m19_czech_base'}
    src=urllib.request.urlopen(BASE,timeout=120).read().decode()
    exec(compile(src,'run_v17.py','exec'),b)
    # Add Czech without changing any existing language source or rule.
    b['ns']['LM_URLS']['czech']=CZ_TRAIN
    b['LANGS']=list(b['LANGS'])+['czech']
    return b,hashlib.sha256(src.encode()).hexdigest()

def split_nonspace(span,n):
    out=[]
    for c in span:
        if c!=' ':
            out.append(c)
            if len(out)==n: break
    return out

def install_support_span(b,K):
    def support_span(pool,n,tag):
        la=tag[1]; pos=[i for i,c in enumerate(pool) if c!=' ']
        for attempt in range(2000):
            st=sd('span',la,K,attempt)%(len(pos)-n+1)
            span=pool[pos[st]:pos[st+n-1]+1].strip()
            tr=split_nonspace(span,b['TRAIN']); vals=set(); chars=set(tr)
            for c in chars: vals.update(b['V2I'][v] for v in b['LETTER_VALS'][b['A2I'][c]])
            if len(vals)==b['NV']:
                print('CZ_SUPPORT',la,K,attempt,''.join(sorted(chars)),flush=True)
                return span
        raise RuntimeError(('no support-complete span',la,K))
    b['ns']['choose_span']=support_span

def one_restart(b,K,S,comp,tag,ens,rr):
    rng=np.random.default_rng(sd('opt',K,tag,ens,rr));m=b['init_map'](K,rng);s=b['score_num'](S,m,comp);ds=[]
    for _ in range(64):
        x,ch=b['proposal'](m,rng);ds.append(abs(b['score_num'](S,x,comp)-s))
    t0=max(1e-6,float(np.median(ds))*4);local=(s,m.copy())
    for k in range(100000):
        frac=k/99999.;temp=max(1e-8,t0*(0.003**frac));x,ch=b['proposal'](m,rng);s2=b['score_num'](S,x,comp);d=s2-s
        if d>=0 or rng.random()<math.exp(max(-60,d/temp)):
            m=x;s=s2
            if s>local[0]:local=(s,m.copy())
    m=local[1].copy();s=b['score_num'](S,m,comp)
    for _ in range(30):
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

def paired_fit_qual(b,K,S,comp,true):
    best={'A':(-1e100,None),'B':(-1e100,None)};true_s=b['score_num'](S,true,comp);history=[]
    for batch in range(4):
        for ens in ['A','B']:
            for j in range(6):
                rr=batch*6+j;s,m=one_restart(b,K,S,comp,'czech-qual',ens,rr)
                if s>best[ens][0]:best[ens]=(s,m.copy())
        gap=abs(best['A'][0]-best['B'][0]);agr=b['agreement'](S['freq'],best['A'][1],best['B'][1]);oa=best['A'][0]-true_s;ob=best['B'][0]-true_s
        row={'restarts_per_ensemble':(batch+1)*6,'scoreA':best['A'][0],'scoreB':best['B'][0],'score_gap':gap,'agreement':agr,'oracleA':oa,'oracleB':ob}
        history.append(row);print('CZ_QUAL_BATCH',K,json.dumps(row,separators=(',',':')),flush=True)
        if gap<=1e-7 and agr>=.95 and oa>=-1e-6 and ob>=-1e-6:break
    conv=history[-1]['score_gap']<=1e-7 and history[-1]['agreement']>=.95 and history[-1]['oracleA']>=-1e-6 and history[-1]['oracleB']>=-1e-6
    winner=best['A'] if best['A'][0]>=best['B'][0] else best['B']
    return winner[0],winner[1],history[-1]['agreement'],conv,history,true_s

def paired_fit_target(b,K,S,comp):
    best={'A':(-1e100,None),'B':(-1e100,None)};history=[]
    for batch in range(4):
        for ens in ['A','B']:
            for j in range(6):
                rr=batch*6+j;s,m=one_restart(b,K,S,comp,'czech-h19-family',ens,rr)
                if s>best[ens][0]:best[ens]=(s,m.copy())
        gap=abs(best['A'][0]-best['B'][0]);agr=b['agreement'](S['freq'],best['A'][1],best['B'][1])
        row={'restarts_per_ensemble':(batch+1)*6,'scoreA':best['A'][0],'scoreB':best['B'][0],'score_gap':gap,'agreement':agr}
        history.append(row);print('CZ_H19_BATCH',json.dumps(row,separators=(',',':')),flush=True)
        if gap<=1e-7 and agr>=.95:break
    conv=history[-1]['score_gap']<=1e-7 and history[-1]['agreement']>=.95
    winner=best['A'] if best['A'][0]>=best['B'][0] else best['B']
    return winner[0],winner[1],history[-1]['agreement'],conv,history

def build_lms_and_czech_pool(b):
    lms,lmmeta=b['build_lms']()
    # b.build_lms now sees Czech because ns.LM_URLS was extended before invocation.
    ss=[]; blobs={}
    for label,url in [('dev',CZ_DEV),('test',CZ_TEST)]:
        raw=urllib.request.urlopen(url,timeout=120).read();blobs[label]=hashlib.sha256(raw).hexdigest();ss.extend(b['ns']['conllu'](raw.decode('utf-8','replace')))
    pool=b['ns']['pool_text'](ss);meta={'sentences':len(ss),'letters':sum(c!=' ' for c in pool),'sha256':blobs}
    if meta['letters']<b['TRAIN']+b['HOLD']:raise RuntimeError(('czech control pool short',meta))
    return lms,lmmeta,pool,meta

def run_qual(K):
    if K not in (22,26,36):raise RuntimeError(K)
    b,base_sha=load_base(); install_support_span(b,K)
    lms,lmmeta,pool,poolmeta=build_lms_and_czech_pool(b)
    comps={la:b['ns']['induced'](lms[la]) for la in b['LANGS']}
    tr,ho,syms,true,attempt=b['gen_control'](pool,'czech',K);S=b['stats'](tr,syms);H=b['stats'](ho,syms)
    s,m,agr,conv,hist,true_s=paired_fit_qual(b,K,S,comps['czech'],true)
    acc=b['map_acc'](H['freq'],m,true);rank=[]
    for cand in b['LANGS']:
        fw,n=b['forward'](ho,m,syms,lms[cand]);rank.append((cand,fw,n))
    rank.sort(key=lambda x:x[1],reverse=True);margin=rank[0][1]-rank[1][1]
    gate=(rank[0][0]=='czech' and margin>=.05 and acc>=.85 and agr>=.90 and conv and s-true_s>=-1e-6)
    out={'phase':'qualification','K':K,'gate':gate,'attempt':attempt,'top':rank[0][0],'czech_rank':1+next(i for i,x in enumerate(rank) if x[0]=='czech'),
         'margin':margin,'mapping_acc':acc,'fit_agreement':agr,'converged':conv,'best_minus_true_score':s-true_s,'ranking':rank,
         'history':hist,'lm_meta':lmmeta,'czech_pool_meta':poolmeta,'base_source_sha256':base_sha,'czech_commit':CZECH_COMMIT}
    print('CZ_QUAL_RESULT='+json.dumps(out,separators=(',',':')),flush=True)
    return out

def run_h19_family():
    K=22;b,base_sha=load_base();lms,lmmeta,_,poolmeta=build_lms_and_czech_pool(b);comp=b['ns']['induced'](lms['czech'])
    td=tempfile.mkdtemp(prefix='m19_czech_h19_');paths,source_meta=b['acquire_sources'](td)
    st=open(paths['RF'],encoding='utf-8').read();pages=b['parse_sta'](st,'family',False)
    T,H,C,allf=b['split_pages'](pages);Cnt=b['count_tokens'](pages,T,'family');v,selcov=b['choose_vocab'](Cnt,'family')
    if len(v)!=22:raise RuntimeError(('K mismatch',len(v),v))
    Tw,Tby,Tcov=b['project'](pages,T,'family',v);Hw,Hby,Hcov=b['project'](pages,H,'family',v);S=b['stats'](Tw,v)
    s,m,agr,conv,hist=paired_fit_target(b,K,S,comp)
    czech_score,n=b['forward'](Hw,m,v,lms['czech'])
    rank=[(la,sc) for la,sc in ORIGINAL_RANK]+[('czech',czech_score)];rank.sort(key=lambda x:x[1],reverse=True)
    margin=rank[0][1]-rank[1][1];top=rank[0][0]
    topagr=agr if top=='czech' else 1.0
    gate=(Hcov['coverage']>=.97 and margin>=.05 and topagr>=.90 and conv)
    out={'phase':'H19-family','K':22,'czech_score':czech_score,'czech_rank':1+next(i for i,x in enumerate(rank) if x[0]=='czech'),
         'ranking':rank,'top':top,'margin':margin,'coverage':Hcov['coverage'],'word_coverage':Hcov['word_coverage'],
         'czech_fit_agreement':agr,'czech_converged':conv,'gate':gate,'train_score':s,'history':hist,'vocab':v,
         'Tfolios':T,'Hfolios':H,'Cfolios_sha256':hashlib.sha256(('\n'.join(C)).encode()).hexdigest(),
         'source_meta':source_meta,'lm_meta':lmmeta,'base_source_sha256':base_sha,'czech_commit':CZECH_COMMIT,
         'original_comparator':ORIGINAL_RANK,'original_coverage':ORIGINAL_COVERAGE}
    print('CZ_H19_RESULT='+json.dumps(out,separators=(',',':')),flush=True)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['qual','h19-family'],required=True);ap.add_argument('--K',type=int,default=None);args=ap.parse_args()
    if args.phase=='qual':
        if args.K is None:raise SystemExit('--K required')
        run_qual(args.K)
    else:run_h19_family()
if __name__=='__main__':main()
