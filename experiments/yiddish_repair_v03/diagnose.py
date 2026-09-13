#!/usr/bin/env python3
import argparse,collections,importlib.util,json,math,pickle,statistics,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v02',HERE.parent/'yiddish_l_calibration_v02/run_l_calibration_v02.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);c=v.core

class BasicUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('class loading prohibited')
def read_pickle(path):
    with path.open('rb') as f:return BasicUnpickler(f).load()

def run(data,out):
    checkpoint=read_pickle(data/'result/checkpoint.pkl');truth=checkpoint['truth'];rows=checkpoint['rows']
    models=read_pickle(data/'public/models.pkl');res=[]
    y=next(k for k,t in truth['model_truth'].items() if t=='yiddish');g=next(k for k,t in truth['model_truth'].items() if t=='german')
    assert len(rows)==16*32
    for group,meta in sorted(truth['group_truth'].items()):
        pcs={r['key_index']:r for r in json.loads((data/f'public/{group}.json').read_text())['cases']}
        trs={r['key_index']:r for r in truth['cases'][group]['rows']}
        rs=[r for r in rows if r['group']==group];assert len(rs)==32
        cm=y if meta['language']=='yiddish' else g
        details=[]
        for r in rs:
            ki=r['key_index'];pc=pcs[ki];tr=trs[ki];oracle=tr['oracle'];lp=models[cm]['logp']
            assert c.decode_words(pc['audit_cipher'],oracle)==tr['audit_truth']
            Cf,_=c.base.cipher_counts(pc['fit_cipher']);Ca,_=c.base.cipher_counts(pc['audit_cipher'])
            fit_true=c.base.mapping_score(Cf,lp,oracle)
            fit_return=c.base.mapping_score(Cf,lp,r['primary'][cm]['mapping'])
            assert abs(fit_return-r['primary'][cm]['fit_objective'])<1e-9
            audit_true=c.base.mapping_score(Ca,lp,oracle)
            audit_return=c.base.mapping_score(Ca,lp,r['primary'][cm]['mapping'])
            assert abs(audit_return-r['primary'][cm]['score'])<1e-9
            own_null_sd=r['primary'][cm]['null_sd']
            neighbours=[]
            # Exhaust all elementary swaps; independent direct rescoring verifies delta.
            for a in range(26):
                for b in range(a+1,26):
                    m=oracle.copy();m[a],m[b]=m[b],m[a]
                    delta=c.base.swap_delta(Cf,lp,oracle,a,b)
                    if ki==0:assert abs(delta-(c.base.mapping_score(Cf,lp,m)-fit_true))<1e-10
                    neighbours.append((delta,a,b))
            best=max(neighbours);m=oracle.copy();m[best[1]],m[best[2]]=m[best[2]],m[best[1]]
            yz=c.z_bigram(pc['audit_cipher'],oracle,models[y]['logp'],c.random_maps('primary',group,ki))
            gz=c.z_bigram(pc['audit_cipher'],oracle,models[g]['logp'],c.random_maps('primary',group,ki))
            oracle_contrast=v.z_ind_from_arm({y:yz,g:gz},y,g)
            decoded=c.decode_words(pc['audit_cipher'],r['primary'][cm]['mapping']);conf=collections.Counter()
            lengths=collections.defaultdict(lambda:[0,0])
            for tw,dw in zip(tr['audit_truth'],decoded):
                assert len(tw)==len(dw)
                lengths[len(tw)][0]+=1;lengths[len(tw)][1]+=tw!=dw
                conf.update(a+'>'+b for a,b in zip(tw,dw) if a!=b)
            details.append({'key':ki,'fit_return_minus_truth':fit_return-fit_true,'audit_return_minus_truth':audit_return-audit_true,'audit_null_sd':own_null_sd,'oracle_contrast':oracle_contrast,'best_swap_fit_gain':best[0],'best_swap_plain_letters':[chr(97+oracle[best[1]]),chr(97+oracle[best[2]])],'best_swap_audit_gain':c.base.mapping_score(Ca,lp,m)-audit_true,'confusions':dict(conf),'word_length_counts_errors':dict(lengths)})
        row={**meta,'group':group,'n':len(details),'fit_return_beats_truth':sum(x['fit_return_minus_truth']>1e-10 for x in details),'audit_return_beats_truth':sum(x['audit_return_minus_truth']>1e-10 for x in details),'median_fit_return_minus_truth':statistics.median(x['fit_return_minus_truth'] for x in details),'median_audit_return_minus_truth':statistics.median(x['audit_return_minus_truth'] for x in details),'median_audit_random_map_null_sd':statistics.median(x['audit_null_sd'] for x in details),'oracle_median_D':statistics.median(x['oracle_contrast']['D'] for x in details),'oracle_median_independence_null_sd':statistics.median(x['oracle_contrast']['denom'] for x in details),'oracle_median_z':statistics.median(x['oracle_contrast']['z_ind'] for x in details),'best_swap_plain_letters':details[0]['best_swap_plain_letters'],'best_swap_fit_gain':details[0]['best_swap_fit_gain'],'confusions_key0':details[0]['confusions'],'details':details}
        res.append(row);c.atomic_pickle({'status':'DIAGNOSTIC_ONLY','target_loaded':False,'families':res},out/'checkpoint.pkl');c.atomic_json(res,out/'diagnostics.json')
        print(json.dumps({k:x for k,x in row.items() if k!='details'}),flush=True)
    c.atomic_json({'status':'KNOWN_KEY_DIAGNOSTIC_COMPLETE','families':len(res),'target_loaded':False},out/'status.json')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.data,a.out)
