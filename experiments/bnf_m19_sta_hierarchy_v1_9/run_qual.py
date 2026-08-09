#!/usr/bin/env python3
import os,urllib.request,hashlib,math,json,collections
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/15e1cfa0e37119907d6a99ba6b2e2be1c4730fa6/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py'
b={'__name__':'v19base'}
exec(compile(urllib.request.urlopen(BASE,timeout=120).read().decode(),'run_v17.py','exec'),b)
NS='M19STAv19Q1'
def sd(*p):return int.from_bytes(hashlib.sha256(('::'.join([NS]+list(map(str,p)))).encode()).digest()[:8],'big')&0xffffffff
b['seed']=sd

def split_nonspace(span,n):
    out=[]
    for c in span:
        if c!=' ':
            out.append(c)
            if len(out)==n: break
    return out

def support_span(pool,n,tag):
    la,K=tag[1],int(tag[2]);pos=[i for i,c in enumerate(pool) if c!=' ']
    for attempt in range(2000):
        st=sd('span',la,K,attempt)%(len(pos)-n+1);span=pool[pos[st]:pos[st+n-1]+1].strip();tr=split_nonspace(span,b['TRAIN']);vals=set();chars=set(tr)
        for c in chars: vals.update(b['V2I'][v] for v in b['LETTER_VALS'][b['A2I'][c]])
        if len(vals)==b['NV']:
            print('V19_SUPPORT',la,K,attempt,''.join(sorted(chars)),flush=True);return span
    raise RuntimeError(('no support-complete span',la,K))
b['ns']['choose_span']=support_span

K=int(os.environ['M19_K'])
if K not in (22,26,36):raise RuntimeError(K)

def one_restart(S,comp,la,ens,rr):
    rng=np.random.default_rng(sd('opt',K,la,ens,rr));m=b['init_map'](K,rng);s=b['score_num'](S,m,comp);ds=[]
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

def paired_fit(S,comp,la,true):
    best={'A':(-1e100,None),'B':(-1e100,None)};true_s=b['score_num'](S,true,comp);history=[]
    for batch in range(4):
        for ens in ['A','B']:
            for j in range(6):
                rr=batch*6+j;s,m=one_restart(S,comp,la,ens,rr)
                if s>best[ens][0]:best[ens]=(s,m.copy())
        gap=abs(best['A'][0]-best['B'][0]);agr=b['agreement'](S['freq'],best['A'][1],best['B'][1]);oa=best['A'][0]-true_s;ob=best['B'][0]-true_s
        history.append({'restarts_per_ensemble':(batch+1)*6,'scoreA':best['A'][0],'scoreB':best['B'][0],'score_gap':gap,'agreement':agr,'oracleA':oa,'oracleB':ob})
        print('V19_BATCH',la,K,json.dumps(history[-1],separators=(',',':')),flush=True)
        if gap<=1e-7 and agr>=.95 and oa>=-1e-6 and ob>=-1e-6:break
    conv=history[-1]['score_gap']<=1e-7 and history[-1]['agreement']>=.95 and history[-1]['oracleA']>=-1e-6 and history[-1]['oracleB']>=-1e-6
    winner=best['A'] if best['A'][0]>=best['B'][0] else best['B']
    return winner[0],winner[1],history[-1]['agreement'],conv,history,true_s

lms,lmmeta=b['build_lms']();comps={la:b['ns']['induced'](lms[la]) for la in b['LANGS']};pools,poolmeta=b['control_pools']();rows=[]
for la in b['QUAL']:
    tr,ho,syms,true,attempt=b['gen_control'](pools[la],la,K);S=b['stats'](tr,syms);H=b['stats'](ho,syms);s,m,agr,conv,hist,true_s=paired_fit(S,comps[la],la,true);acc=b['map_acc'](H['freq'],m,true);rank=[]
    for cand in b['LANGS']:
        fw,n=b['forward'](ho,m,syms,lms[cand]);rank.append((cand,fw))
    rank.sort(key=lambda x:x[1],reverse=True);row={'lang':la,'K':K,'attempt':attempt,'top':rank[0][0],'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'margin':rank[0][1]-rank[1][1],'mapping_acc':acc,'fit_agreement':agr,'converged':conv,'best_minus_true_score':s-true_s,'history':hist,'ranking':rank};rows.append(row);print('V19_QUAL',json.dumps(row,separators=(',',':')),flush=True)
gate={'K':K,'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'median_acc':float(np.median([r['mapping_acc'] for r in rows])),'min_acc':min(r['mapping_acc'] for r in rows),'min_agreement':min(r['fit_agreement'] for r in rows),'all_converged':all(r['converged'] for r in rows),'min_oracle_gap':min(r['best_minus_true_score'] for r in rows)}
gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90 and gate['all_converged'] and gate['min_oracle_gap']>=-1e-6
print('V19_GATE='+json.dumps({'gate':gate,'rows':rows,'lm_meta':lmmeta,'pool_meta':poolmeta},separators=(',',':')),flush=True)
