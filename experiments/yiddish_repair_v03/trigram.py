#!/usr/bin/env python3
import argparse,collections,ctypes,hashlib,json,math,pickle,statistics,sys,time
from pathlib import Path
import numpy as np
from diagnose import c,v,read_pickle,HERE

def counts(words):
    out=collections.Counter()
    for w in words:
        seq=[26,26]+[ord(a)-97 for a in w]+[26]
        out.update(zip(seq,seq[1:],seq[2:]))
    ids=np.ascontiguousarray(sorted(out),dtype=np.int32)
    weights=np.ascontiguousarray([out[tuple(k)] for k in ids],dtype=np.float64);weights/=sum(weights)
    return ids,weights

def train(words):
    bigram,uni=c.base.lm_from_words(words)
    arr=np.zeros((27,27,27),dtype=np.float64)
    for w in words:
        seq=[26,26]+[ord(a)-97 for a in w]+[26]
        for a,b,d in zip(seq,seq[1:],seq[2:]):arr[a,b,d]+=1
    N=arr.sum(axis=2,keepdims=True);T=(arr>0).sum(axis=2,keepdims=True)
    back=np.broadcast_to(np.exp(np.array(bigram))[None,:,:],arr.shape)
    probs=np.divide(arr+T*back,N+T,out=back.copy(),where=N+T>0)
    assert np.allclose(probs.sum(axis=2),1)
    return {'lp':np.ascontiguousarray(np.log(probs)), 'train_uni':uni}

def score(events,lp,mapping):
    ids,w=events;m=np.array(list(mapping)+[26]);mi=m[ids]
    return float(w@lp[mi[:,0],mi[:,1],mi[:,2]])

def slow_score(words,lp,mapping):
    vals=[];m=list(mapping)+[26]
    for w in words:
        seq=[26,26]+[ord(a)-97 for a in w]+[26]
        vals.extend(lp[m[a],m[b],m[d]] for a,b,d in zip(seq,seq[1:],seq[2:]))
    return float(sum(vals)/len(vals))

_lib=None
def lib():
    global _lib
    if _lib is None:
        _lib=ctypes.CDLL(str(HERE/'native_trigram.so'))
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        _lib.trigram_solve.argtypes=[ctypes.c_int,ip,dp,dp,ip,ctypes.c_uint64,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip];_lib.trigram_solve.restype=ctypes.c_double
        _lib.trigram_delta.argtypes=[ctypes.c_int,ip,dp,dp,ip,ctypes.c_int,ctypes.c_int];_lib.trigram_delta.restype=ctypes.c_double
    return _lib

def solve(words,model,seed):
    events=counts(words);ids,w=events
    _,uni=c.base.cipher_counts(words);initial=np.array(c.base.frequency_initial(uni,model['train_uni']),dtype=np.int32);out=np.empty(26,dtype=np.int32)
    obj=lib().trigram_solve(len(w),ids,w,model['lp'],initial,seed,8,5000,60,out)
    assert abs(obj-score(events,model['lp'],out))<1e-10
    assert sorted(out.tolist())==list(range(26))
    return out.tolist(),obj

def prepare(data,penn,out):
    old=read_pickle(data/'public/models.pkl');truth=read_pickle(data/'result/checkpoint.pkl')['truth']
    ypool=c.penn_family(penn,v.YID_BUILD)
    for fam,files in v.PENN_EXPECTED.items():
        if fam in v.YID_BUILD:
            for name,sha in files.items():assert c.sha_file(penn/name)==sha
    gpool={name:v.ref_words_v02(data/'ref_extract',name)[0] for name in v.GER_BUILD}
    trains={'yiddish':c.work_balanced(ypool,10000)[0],'german':c.work_balanced(gpool,10000)[0]}
    models={}
    for mid,lang in truth['model_truth'].items():
        lp,uni=c.base.lm_from_words(trains[lang]);assert lp==old[mid]['logp'] and uni==old[mid]['train_uni']
        models[mid]=train(trains[lang])
    c.atomic_pickle(models,out/'models.pkl')
    c.atomic_json({'status':'BUILD_BIGRAM_REPRODUCED_EXACTLY','training_words_per_language':10000,'model_sha256':c.sha_file(out/'models.pkl'),'target_loaded':False},out/'model_manifest.json')
    # Oracle neighbourhood analysis: development diagnostic, never solver input.
    diagnostics=[]
    for group,meta in sorted(truth['group_truth'].items()):
        tr=truth['cases'][group]['rows'][0];mid=next(k for k,lang in truth['model_truth'].items() if lang==meta['language']);lp=models[mid]['lp'];identity=list(range(26))
        events=counts(tr['fit_truth']);base=score(events,lp,identity);assert abs(base-slow_score(tr['fit_truth'],lp,identity))<1e-10
        neigh=[]
        for a in range(26):
            for b in range(a+1,26):
                m=identity.copy();m[a],m[b]=m[b],m[a];delta=score(events,lp,m)-base
                native=lib().trigram_delta(len(events[1]),events[0],events[1],lp,np.array(identity,dtype=np.int32),a,b)
                assert abs(native-delta)<1e-10
                neigh.append((delta,a,b))
        best=max(neigh);sw=identity.copy();sw[best[1]],sw[best[2]]=sw[best[2]],sw[best[1]]
        rec=c.base.score_recovery(tr['audit_truth'],c.decode_words(tr['audit_truth'],sw))
        row={**meta,'true_key_best_single_swap_gain':best[0],'best_swap':[chr(97+best[1]),chr(97+best[2])],'audit_recovery_after_swap':rec}
        diagnostics.append(row)
    c.atomic_json(diagnostics,out/'oracle_neighbourhood.json');print(json.dumps(diagnostics,indent=2))

def run_group(data,modeldir,out,group):
    # This function deliberately has no PRIVATE/truth path access.
    with (modeldir/'models.pkl').open('rb') as f:models=pickle.load(f)
    old=read_pickle(data/'public/models.pkl')
    cases=json.loads((data/f'public/{group}.json').read_text())['cases'];rows=[]
    dest=out/group;dest.mkdir(parents=True,exist_ok=True)
    done=dest/'checkpoint.pkl'
    if done.exists():rows=read_pickle(done)['rows']
    for pc in cases[len(rows):]:
        ki=pc['key_index'];row={'group':group,'key_index':ki,'primary':{},'n1':{},'n2':{}}
        for arm,fc,ac in [('primary',pc['fit_cipher'],pc['audit_cipher']),('n2',pc['n2_fit_cipher'],pc['n2_audit_cipher'])]:
            events=counts(ac);maps=c.random_maps(arm,group,ki)
            for mid in ('A','B'):
                t=time.time();seed=int.from_bytes(hashlib.sha256(f'WB03|{arm}|{group}|{ki}'.encode()).digest()[:8],'big')
                m,obj=solve(fc,models[mid],seed);x=score(events,models[mid]['lp'],m);null=[score(events,models[mid]['lp'],z) for z in maps];mu=statistics.mean(null);sd=statistics.pstdev(null)
                row[arm][mid]={'score':x,'null_mean':mu,'null_sd':sd,'mapping':m,'fit_objective':obj,'runtime_s':time.time()-t}
        # Unigram nuisance is unchanged, including deterministic mapping/nulls.
        _,uni=c.base.cipher_counts(pc['fit_cipher'])
        for mid in ('A','B'):
            m=c.base.frequency_initial(uni,old[mid]['train_uni']);row['n1'][mid]=c.z_unigram(pc['audit_cipher'],m,old[mid]['unigram'],c.random_maps('n1',group,ki));row['n1'][mid]['mapping']=m
        rows.append(row);c.atomic_pickle({'group':group,'rows':rows,'target_loaded':False},done)
    (dest/'rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    print(json.dumps({'group':group,'rows':len(rows),'status':'BLIND_DEVELOPMENT_COMPLETE'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','solve']);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--penn',type=Path);ap.add_argument('--models',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--group');a=ap.parse_args()
    if a.mode=='prepare':prepare(a.data,a.penn,a.out)
    else:run_group(a.data,a.models,a.out,a.group)
