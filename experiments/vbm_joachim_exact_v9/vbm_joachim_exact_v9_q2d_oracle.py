# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import json, urllib.request, numpy as np

SRC='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-joachim-exact-v9-20260901/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q2_synth_ident.py'
with urllib.request.urlopen(SRC,timeout=120) as r: code=r.read().decode('utf-8')
ns={'__name__':'q2mod'}
exec(compile(code,SRC,'exec'),ns)


def score_plain(lines,lm):
    ll=0.;n=0
    for L in lines:
        a,b=lm.score(L['plain']);ll+=a;n+=b
    return ll/max(1,n)


def random_key_scores(lines,asset,key,tag,nrand=200):
    out=[]
    for rr in range(nrand):
        rng=np.random.default_rng(ns['seed'](ns['NS'],'Q2D',tag,rr))
        bm=np.asarray(key['bmap'],dtype=np.int16).copy();nm=np.asarray(key['nmap'],dtype=np.int16).copy();rng.shuffle(bm);rng.shuffle(nm)
        out.append(ns['score_lines'](lines,asset,{'bmap':bm,'nmap':nm}))
    return np.asarray(out,float)


def local_truth_scan(lines,asset,key):
    bm=np.asarray(key['bmap'],dtype=np.int16).copy();nm=np.asarray(key['nmap'],dtype=np.int16).copy();lm=asset['lm'];runs=asset['runs'];bi,ni=ns['build_index'](lines)
    cache=[];totll=0.;totn=0
    for L in lines:
        ll,nn=lm.score(ns['decode_line'](L,bm,nm,runs));cache.append((ll,nn));totll+=ll;totn+=nn
    base=totll/max(1,totn);improving=[];best_delta=0.0;best_desc=None
    for typ,idxs,K in [('b',bi,ns['NV']),('n',ni,ns['KR'])]:
        for t,affected in enumerate(idxs):
            if not affected:continue
            old=int(bm[t] if typ=='b' else nm[t]);entry_best=base;entry_val=None
            for v in range(K):
                if v==old:continue
                if typ=='b':bm[t]=v
                else:nm[t]=v
                dll=0.;dn=0
                for j in affected:
                    ll0,n0=cache[j];ll1,n1=lm.score(ns['decode_line'](lines[j],bm,nm,runs));dll+=ll1-ll0;dn+=n1-n0
                rat=(totll+dll)/max(1,totn+dn)
                if rat>entry_best+1e-12:entry_best=rat;entry_val=v
            if typ=='b':bm[t]=old
            else:nm[t]=old
            if entry_val is not None:
                delta=entry_best-base;improving.append((typ,t,old,int(entry_val),float(delta)))
                if delta>best_delta:best_delta=delta;best_desc=improving[-1]
    occ_b=sum(bool(x) for x in bi);occ_n=sum(bool(x) for x in ni)
    return {'base_train_lm':float(base),'occurring_bridge_types':int(occ_b),'occurring_nucleus_types':int(occ_n),'entries_with_improving_single_change':len(improving),'fraction_entries_improvable':len(improving)/max(1,occ_b+occ_n),'best_single_change_delta':float(best_delta),'best_single_change':best_desc,'truth_coordinate_local_optimum':len(improving)==0}


def main():
    A=ns['assets']();rows=[]
    for lang in ['DE','IT']:
        other='IT' if lang=='DE' else 'DE'
        for rep in range(3):
            lines,key=ns['positive'](lang,'CAL',rep,A);train=lines[:80];hold=lines[80:]
            truth={'bmap':np.asarray(key['bmap'],dtype=np.int16),'nmap':np.asarray(key['nmap'],dtype=np.int16)}
            true_hold=float(ns['score_lines'](hold,A[lang],truth));rnd=random_key_scores(hold,A[lang],key,f'{lang}:R{rep}',200);med=float(np.median(rnd));q95=float(np.quantile(rnd,.95));mx=float(np.max(rnd));pct=float(np.mean(rnd<true_hold));native_plain=float(score_plain(hold,A[lang]['lm']));other_plain=float(score_plain(hold,A[other]['lm']));loc=local_truth_scan(train,A[lang],key)
            z={'language':lang,'rep':rep,'TRUE_HOLD_LM':true_hold,'RANDOM_MEDIAN':med,'RANDOM_Q95':q95,'RANDOM_MAX':mx,'ORACLE_ADV':true_hold-med,'TRUE_PERCENTILE_VS_RANDOM':pct,'PLAIN_NATIVE_LM':native_plain,'PLAIN_OTHER_LM':other_plain,'NATIVE_MINUS_OTHER':native_plain-other_plain,'RAW_LANGUAGE_DIRECTION_OK':native_plain>other_plain,'local_truth_scan':loc}
            rows.append(z);print('Q2DROW='+json.dumps(z,sort_keys=True,separators=(',',':')),flush=True)
    summary={'protocol':'VBM_JOACHIM_EXACT_V9_Q2D_ORACLE_PROTOCOL.md','rows':rows,'all_true_above_random_max':all(r['TRUE_HOLD_LM']>r['RANDOM_MAX'] for r in rows),'all_language_directions_ok':all(r['RAW_LANGUAGE_DIRECTION_OK'] for r in rows),'median_oracle_adv':float(np.median([r['ORACLE_ADV'] for r in rows])),'median_fraction_truth_entries_improvable':float(np.median([r['local_truth_scan']['fraction_entries_improvable'] for r in rows])),'Q2_remains_closed':True}
    print('VBM_V9_Q2D_RESULT='+json.dumps(summary,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
