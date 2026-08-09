#!/usr/bin/env python3
import urllib.request,json,math,hashlib
from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d4675efc01928ffac79ea605dc223628092cbbda/experiments/bnf_m19_image_bridge_v1_2/run_arm_a.py'
src=urllib.request.urlopen(BASE,timeout=120).read().decode('utf-8')
b={'__name__':'arm_a_lib'};exec(compile(src,'run_arm_a.py','exec'),b)
K=19; R='R1'; LAMBDAS=[.02,.04,.06,.08,.10,.12]


def word_index_lists(rec,F):
    d=defaultdict(list)
    for i,(f,w,sl) in enumerate(zip(rec['folio'],rec['word'],rec['slot'])):
        if f in F:d[(f,int(w))].append((int(sl),i))
    out=[]
    for key,a in sorted(d.items()):
        a.sort();out.append((key,[i for _,i in a]))
    return out

def segvec(A,inds):
    v=A[inds].mean(0);n=np.linalg.norm(v);return (v/max(n,1e-12)).astype(np.float32)

def segment_one(A,inds,cent,lam):
    m=len(inds);dp=[1e100]*(m+1);prev=[None]*(m+1);dp[0]=0.
    # cache candidate group embeddings/scores
    cache={}
    for i in range(1,m+1):
        for L in (1,2,3):
            j=i-L
            if j<0:continue
            key=(j,i);v=segvec(A,inds[j:i]);s=v@cent.T;k=int(np.argmax(s));dist=1-float(s[k]);cost=dp[j]+dist+lam
            cache[key]=(v,k,float(s[k]))
            if cost<dp[i]:dp[i]=cost;prev[i]=(j,key)
    spans=[];i=m
    while i>0:
        j,key=prev[i];spans.append((j,i,*cache[key]));i=j
    spans.reverse();return spans

def segment_words(A,words,cent,lam):
    out=[]
    for key,inds in words:out.append((key,inds,segment_one(A,inds,cent,lam)))
    return out

def collect_segments(seg):
    V=[];meta=[]
    for key,inds,ss in seg:
        for j,i,v,k,sim in ss:V.append(v);meta.append((key,j,i,k,sim))
    return np.asarray(V,np.float32),meta

def fit_segmental(A,words,K,lam,rs):
    # Initial centroids from components, then exactly three segment/refit alternations.
    allidx=np.array([i for _,ii in words for i in ii],np.int64);fit=b['stable_sample'](allidx,80000,('seg-init',lam,rs))
    km=MiniBatchKMeans(n_clusters=K,random_state=rs,batch_size=4096,n_init=3,max_iter=180,reassignment_ratio=.005).fit(A[fit]);cent=km.cluster_centers_.astype(np.float32);cent/=np.maximum(np.linalg.norm(cent,axis=1,keepdims=True),1e-12)
    for it in range(3):
        seg=segment_words(A,words,cent,lam);V,_=collect_segments(seg)
        if len(V)>80000:
            rng=np.random.default_rng(b['seed']('seg-refit',lam,rs,it));V=V[np.sort(rng.choice(len(V),80000,replace=False))]
        km=MiniBatchKMeans(n_clusters=K,random_state=rs+100+it,batch_size=4096,n_init=1,init=cent,max_iter=160,reassignment_ratio=.002).fit(V);cent=km.cluster_centers_.astype(np.float32);cent/=np.maximum(np.linalg.norm(cent,axis=1,keepdims=True),1e-12)
    return cent

def span_dict(seg):
    d={}
    for key,inds,ss in seg:d[key]=[(j,i,k,sim) for j,i,v,k,sim in ss]
    return d

def boundary_label_stability(seg0,seg1,c0,c1):
    row,col=linear_sum_assignment(-(c0@c1.T));map1=np.zeros(K,dtype=int);map1[col]=row
    d0=span_dict(seg0);d1=span_dict(seg1);tp=fp=fn=0;agree=common=0
    for key in d0.keys()&d1.keys():
        a=d0[key];bb=d1[key]
        # internal cut positions only
        ca={i for j,i,k,s in a[:-1]};cb={i for j,i,k,s in bb[:-1]};tp+=len(ca&cb);fp+=len(cb-ca);fn+=len(ca-cb)
        x={(j,i):k for j,i,k,s in a};y={(j,i):k for j,i,k,s in bb}
        for sp in x.keys()&y.keys():common+=1;agree+=int(x[sp]==map1[y[sp]])
    f1=2*tp/max(1,2*tp+fp+fn);la=agree/max(1,common);return float(f1),float(la),float(min(f1,la))
def eval_visual(A,rec,split,lam):
    wf=word_index_lists(rec,split['Tf']);wv=word_index_lists(rec,split['Tv']);c0=fit_segmental(A,wf,K,lam,408);c1=fit_segmental(A,wf,K,lam,409);s0=segment_words(A,wv,c0,lam);s1=segment_words(A,wv,c1,lam);bf,la,stab=boundary_label_stability(s0,s1,c0,c1)
    # Thresholds from Tfit segmentation under c0
    stf=segment_words(A,wf,c0,lam);Vf,mf=collect_segments(stf);labs=np.array([m[3] for m in mf]);sims=np.array([m[4] for m in mf]);thr=np.array([np.quantile(sims[labs==k],.05) if np.any(labs==k) else 1. for k in range(K)])
    Vv,mv=collect_segments(s0);lv=np.array([m[3] for m in mv]);sv=np.array([m[4] for m in mv]);acc=sv>=thr[lv];cov=float(acc.mean());counts=np.bincount(lv[acc],minlength=K);fsets=[set() for _ in range(K)]
    for m,ok in zip(mv,acc):
        if ok:fsets[m[3]].add(m[0][0])
    recmin=min(map(len,fsets));cntmin=int(counts.min());sample=np.arange(len(Vv))
    if len(sample)>3000:
        rng=np.random.default_rng(b['seed']('seg-sil',lam));sample=np.sort(rng.choice(sample,3000,replace=False))
    sil=float(silhouette_score(Vv[sample],lv[sample],metric='cosine')) if len(set(lv[sample]))>1 else -1.;meanw=float(np.mean([len(x[2]) for x in s0]))
    passed=stab>=.75 and cov>=.75 and recmin>=3 and cntmin>=25 and 2.0<=meanw<=10.0
    r={'lambda':lam,'boundary_f1':bf,'label_agreement':la,'stability':stab,'coverage':cov,'min_cluster_folios':recmin,'min_cluster_count':cntmin,'silhouette':sil,'mean_segments_word':meanw,'pass':passed};print('SEG_VISUAL',json.dumps(r,separators=(',',':')),flush=True);return r,c0,thr

def choose_lambda(rows):
    good=[r for r in rows if r['pass']]
    if not good:return None
    mx=max(r['silhouette'] for r in good);near=[r for r in good if mx-r['silhouette']<=.005];near.sort(key=lambda r:-r['lambda']);return near[0]

def fit_final(A,rec,split,lam):
    wt=word_index_lists(rec,split['T']);c=fit_segmental(A,wt,K,lam,408);st=segment_words(A,wt,c,lam);V,m=collect_segments(st);lab=np.array([x[3] for x in m]);sim=np.array([x[4] for x in m]);thr=np.array([np.quantile(sim[lab==k],.05) if np.any(lab==k) else 1. for k in range(K)]);return c,thr,st

def symbolic(seg,thr):
    words=[];total=acc=0
    for key,inds,ss in seg:
        run=[]
        for j,i,v,k,sim in ss:
            total+=1;ok=sim>=thr[k];acc+=int(ok)
            if not ok:
                if run:words.append(run);run=[]
            else:run.append(int(k))
        if run:words.append(run)
    return words,acc/max(1,total),total

# ---------- synthetic segmental positive control ----------
def synth_micro(plain,lang,lam):
    # K=NV=19, each value has a surface prototype. Return component words and true symbol-boundary sets.
    rng=np.random.default_rng(b['seed']('micro-proto',lang));D=64;P=rng.normal(size=(K,D)).astype(np.float32);P/=np.linalg.norm(P,axis=1,keepdims=True)
    rngv=np.random.default_rng(b['seed']('micro-values',lang));rngm=np.random.default_rng(b['seed']('micro-pieces',lang));words=[];truth=[]
    for w in plain.split():
        comps=[];cuts=[];pos=0
        for c in w:
            vi=b['V2I'][int(rngv.choice(b['LETTER_VALS'][b['A2I'][c]]))];L=int(rngm.choice([1,2,3],p=[.35,.45,.20]))
            for _ in range(L):
                x=P[vi]+.08*rngm.normal(size=D);x=x/max(np.linalg.norm(x),1e-12);comps.append(x.astype(np.float32));pos+=1
            cuts.append(pos)
        if comps:words.append(np.asarray(comps,np.float32));truth.append(set(cuts[:-1]))
    return words,truth,P

def array_words(words):
    A=np.concatenate(words,0);out=[];p=0
    for wi,w in enumerate(words):out.append((('s',wi),list(range(p,p+len(w)))));p+=len(w)
    return A,out

def boundary_f1_truth(seg,truth):
    tp=fp=fn=0
    for wi,(_,inds,ss) in enumerate(seg):
        pred={i for j,i,v,k,s in ss[:-1]};tru=truth[wi];tp+=len(pred&tru);fp+=len(pred-tru);fn+=len(tru-pred)
    return 2*tp/max(1,2*tp+fp+fn)
def segmental_qualify(lam,lms,pools,comps):
    rows=[]
    for la in b['QUAL']:
        span=b['choose_span'](pools[la],b['QTRAIN']+b['QHOLD'],('image-v12-segqual',la));trtxt,hotxt=b['split_text_letters'](span,b['QTRAIN'])
        trw,truth,_=synth_micro(trtxt,la,lam);A,widx=array_words(trw);cent=fit_segmental(A,widx,K,lam,408);segtr=segment_words(A,widx,cent,lam);bf=boundary_f1_truth(segtr,truth)
        how,htruth,_=synth_micro(hotxt,la,lam);AH,hidx=array_words(how);segh=segment_words(AH,hidx,cent,lam);bfh=boundary_f1_truth(segh,htruth)
        # no rejection in synthetic geometry; convert recovered cluster labels to symbolic words
        tw=[[int(x[3]) for x in ss] for _,_,ss in segtr];hw=[[int(x[3]) for x in ss] for _,_,ss in segh];S=b['sym_stats'](tw,K);rank=[];fits={}
        for cand in b['LANGS']:
            sc,m=b['optimize'](S,comps[cand],K,('segqual',la,cand,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);fw,_=b['forward_sequences'](hw,m,lms[cand]);rank.append((cand,fw));fits[cand]=m
        rank.sort(key=lambda x:x[1],reverse=True);_,m2=b['optimize'](S,comps[la],K,('segqual2',la,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);agr=b['agreement'](S['freq'],fits[la],m2);r={'lang':la,'top':rank[0][0],'margin':rank[0][1]-rank[1][1],'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'train_boundary_f1':bf,'hold_boundary_f1':bfh,'agreement':agr};rows.append(r);print('SEG_QUAL',json.dumps(r,separators=(',',':')),flush=True)
    gate={'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'min_boundary_f1':min(min(r['train_boundary_f1'],r['hold_boundary_f1']) for r in rows),'min_agreement':min(r['agreement'] for r in rows)};gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['min_boundary_f1']>=.85 and gate['min_agreement']>=.90;print('SEG_QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);return rows,gate

def fit_vms(Tw,Hw,lms,comps):return b['fit_voynich'](Tw,Hw,K,lms,comps)

def main():
    lms,pools,lmmeta=b['load_lms']();comps={la:b['induced'](lms[la]) for la in b['LANGS']};X,rec,split=b['load_image_data']();A=b['folio_center'](X,rec)
    rows=[]
    for lam in LAMBDAS:
        r,_,_=eval_visual(A,rec,split,lam);rows.append(r)
    choice=choose_lambda(rows);out={'protocol':'v1.2-armB','visual_rows':rows,'choice':choice,'lm_meta':lmmeta}
    if choice is None:
        out['verdict']='ARM B IMAGE-UNDERPOWERED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    lam=choice['lambda'];qrows,qgate=segmental_qualify(lam,lms,pools,comps);out['qualification']=qrows;out['qualification_gate']=qgate
    if not qgate['pass']:
        out['verdict']='IMAGE INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    cent,thr,segT=fit_final(A,rec,split,lam);Tw,Tcov,_=symbolic(segT,thr);segH=segment_words(A,word_index_lists(rec,split['H']),cent,lam);Hw,Hcov,_=symbolic(segH,thr);out['stream']={'T_words':len(Tw),'T_units':sum(map(len,Tw)),'T_coverage':Tcov,'H_words':len(Hw),'H_units':sum(map(len,Hw)),'H_coverage':Hcov,'lambda':lam,'K':K};print('SEG_STREAM',json.dumps(out['stream'],separators=(',',':')),flush=True)
    vrows,rank,maps,margin=fit_vms(Tw,Hw,lms,comps);top=rank[0];primary=top['agreement']>=.90 and margin>=.05 and Hcov>=.90;signal={'top':top['lang'],'second':rank[1]['lang'],'margin':margin,'agreement':top['agreement'],'Hcoverage':Hcov,'primary':primary};out['H12']=vrows;out['signal']=signal;print('SEG_H12_SIGNAL',json.dumps(signal,separators=(',',':')),flush=True)
    if not primary:
        out['verdict']='NO IMAGE-M19 SIGNAL';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    # C12 unlock only after H12 pass
    segC=segment_words(A,word_index_lists(rec,split['C']),cent,lam);Cw,Ccov,_=symbolic(segC,thr);cand=top['lang'];m=maps[cand];cr=[]
    for la in b['LANGS']:cr.append((la,b['forward_sequences'](Cw,m,lms[la])[0]))
    cr.sort(key=lambda x:x[1],reverse=True);cm=cr[0][1]-cr[1][1] if cr[0][0]==cand else None;buckets=[]
    for bi,B in enumerate(b['c_buckets'](rec,split['C'])):
        sg=segment_words(A,word_index_lists(rec,B),cent,lam);Bw,bc,_=symbolic(sg,thr);rr=[(la,b['forward_sequences'](Bw,m,lms[la])[0]) for la in b['LANGS']];rr.sort(key=lambda x:x[1],reverse=True);cs=next(x[1] for x in rr if x[0]==cand);bestother=max(x[1] for x in rr if x[0]!=cand);buckets.append({'bucket':bi,'folios':len(B),'units':sum(map(len,Bw)),'coverage':bc,'ranking':rr,'candidate_margin':cs-bestother})
    confirmed=cr[0][0]==cand and cm is not None and cm>=.05 and Ccov>=.90 and all(x['candidate_margin']>0 for x in buckets);out['C12']={'coverage':Ccov,'words':len(Cw),'units':sum(map(len,Cw)),'ranking':cr,'candidate':cand,'margin':cm,'buckets':buckets,'confirmed':confirmed};out['winning_map']={str(i):b['VALUES'][int(m[i])] for i in range(K)};out['verdict']=('CONFIRMED IMAGE-M19 SIGNAL '+cand) if confirmed else 'H12 IMAGE-M19 CANDIDATE / C12 FAILED';print('SEG_C12',json.dumps(out['C12'],separators=(',',':')),flush=True);print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
