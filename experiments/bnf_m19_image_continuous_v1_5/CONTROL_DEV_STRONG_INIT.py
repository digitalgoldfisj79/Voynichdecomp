#!/usr/bin/env python3
import urllib.request,json
import numpy as np
U='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/90dd44f655844aa60bc5afbe50c10286faddba1f/experiments/bnf_m19_image_continuous_v1_5/run_v15.py'
src=urllib.request.urlopen(U,timeout=120).read().decode('utf-8')
ns={'__name__':'v15dev'};exec(compile(src,'run_v15.py','exec'),ns)
b=ns['b']; SIGMA=0.01021827930671254

# Only change under test: stronger hard permutation initializer, matching v0.9 budget.
def init_strong(words,comp,tag,rs):
    X=ns['flatten'](words);sample=ns['stable_sample'](X,80000,('strong-init',tag,rs));from sklearn.cluster import MiniBatchKMeans
    km=MiniBatchKMeans(n_clusters=19,random_state=rs,batch_size=4096,n_init=6,max_iter=240,reassignment_ratio=.003).fit(sample);cent=km.cluster_centers_.astype(np.float32);labs=ns['np_hard_labels'](words,cent);S=b['sym_stats'](labs,19);_,m=b['optimize'](S,comp,19,('v15-strong-map',tag,rs),24000,6);mu=np.zeros_like(cent)
    for c,v in enumerate(m):mu[int(v)]=cent[c]
    return mu
ns['init_means']=init_strong

def one(target,lms,pools,comps):
    trtxt,hotxt=ns['split_plain'](pools[target],('v15qual',target));tw,P,sig=ns['synth_words'](trtxt,target,64,SIGMA,'tr');hw,P2,sig2=ns['synth_words'](hotxt,target,64,SIGMA,'ho');tg=ns['group_tensors'](tw);hg=ns['group_tensors'](hw);rank=[];correct=None
    for cand in b['LANGS']:
        mu,trll,cnt,agr=ns['fit_language'](tw,tg,sig,comps[cand],('strongdev',target,cand));sc=ns['score_groups'](hg,mu,sig,comps[cand]);rank.append((cand,sc['visual_gain']));
        if cand==target:correct=(mu,agr)
    rank.sort(key=lambda z:z[1],reverse=True);row={'target':target,'top':rank[0][0],'margin':rank[0][1]-rank[1][1],'rank':1+next(i for i,x in enumerate(rank) if x[0]==target),'recovery':ns['mean_recovery'](correct[0],P),'agreement':correct[1],'ranking':rank};print('STRONG_DEV',json.dumps(row,separators=(',',':')),flush=True);return row

def main():
    lms,pools,_=b['load_lms']();comps={la:b['induced'](lms[la]) for la in b['LANGS']};rows=[one(x,lms,pools,comps) for x in ['latin','french','german']];print('RESULT_JSON='+json.dumps(rows,separators=(',',':')),flush=True)
if __name__=='__main__':main()
