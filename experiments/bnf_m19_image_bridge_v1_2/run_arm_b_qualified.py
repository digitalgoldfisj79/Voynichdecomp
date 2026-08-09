#!/usr/bin/env python3
import urllib.request,json
import numpy as np
from scipy.optimize import linear_sum_assignment

U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7b97e064c1098d63158a9a406780999aca91103d/experiments/bnf_m19_image_bridge_v1_2/run_arm_b.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
ns={'__name__':'arm_b_lib'};exec(compile(src,'run_arm_b.py','exec'),ns)
b=ns['b'];K=ns['K']

def segmental_qualify_fixed(lam,lms,pools,comps):
    rows=[]
    for la in b['QUAL']:
        span=b['choose_span'](pools[la],b['QTRAIN']+b['QHOLD'],('image-v12-segqual',la));trtxt,hotxt=b['split_text_letters'](span,b['QTRAIN'])
        trw,truth,P=ns['synth_micro'](trtxt,la,lam);A,widx=ns['array_words'](trw);cent=ns['fit_segmental'](A,widx,K,lam,408);segtr=ns['segment_words'](A,widx,cent,lam);bf=ns['boundary_f1_truth'](segtr,truth)
        how,htruth,PH=ns['synth_micro'](hotxt,la,lam);assert np.allclose(P,PH);AH,hidx=ns['array_words'](how);segh=ns['segment_words'](AH,hidx,cent,lam);bfh=ns['boundary_f1_truth'](segh,htruth)
        # Known synthetic centroid->M19-value map, recovered without using plaintext strings.
        rr,cc=linear_sum_assignment(-(cent@P.T));true=np.full(K,-1,np.int16)
        for r,c in zip(rr,cc):true[int(r)]=int(c)
        tw=[[int(x[3]) for x in ss] for _,_,ss in segtr];hw=[[int(x[3]) for x in ss] for _,_,ss in segh];S=b['sym_stats'](tw,K);rank=[];fits={}
        for cand in b['LANGS']:
            sc,m=b['optimize'](S,comps[cand],K,('segqual',la,cand,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);fw,_=b['forward_sequences'](hw,m,lms[cand]);rank.append((cand,fw));fits[cand]=m
        rank.sort(key=lambda x:x[1],reverse=True);_,m2=b['optimize'](S,comps[la],K,('segqual2',la,lam),b['CONTROL_STEPS'],b['CONTROL_RESTARTS']);agr=b['agreement'](S['freq'],fits[la],m2);acc=b['weighted_acc'](S['freq'],fits[la],true)
        r={'lang':la,'top':rank[0][0],'margin':rank[0][1]-rank[1][1],'rank':1+next(i for i,x in enumerate(rank) if x[0]==la),'train_boundary_f1':bf,'hold_boundary_f1':bfh,'mapping_acc':acc,'agreement':agr};rows.append(r);print('SEG_QUAL',json.dumps(r,separators=(',',':')),flush=True)
    gate={'correct':sum(r['top']==r['lang'] for r in rows),'min_margin':min(r['margin'] for r in rows),'min_boundary_f1':min(min(r['train_boundary_f1'],r['hold_boundary_f1']) for r in rows),'median_acc':float(np.median([r['mapping_acc'] for r in rows])),'min_acc':min(r['mapping_acc'] for r in rows),'min_agreement':min(r['agreement'] for r in rows)}
    gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['min_boundary_f1']>=.85 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90
    print('SEG_QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);return rows,gate

ns['segmental_qualify']=segmental_qualify_fixed
ns['main']()
