#!/usr/bin/env python3
import argparse,json,math,statistics
from pathlib import Path
from diagnose import c,v,read_pickle

def run(data,solutions,out):
    truth=read_pickle(data/'result/checkpoint.pkl')['truth'];mt=truth['model_truth'];y=next(k for k,l in mt.items() if l=='yiddish');g=next(k for k,l in mt.items() if l=='german')
    families=[];allrows=[];hashes={}
    for group,meta in sorted(truth['group_truth'].items()):
        path=solutions/group/'rows.jsonl';hashes[group]=c.sha_file(path)
        rs=[json.loads(l) for l in path.read_text().splitlines()];assert len(rs)==32 and sorted(r['key_index'] for r in rs)==list(range(32))
        pcs={r['key_index']:r for r in json.loads((data/f'public/{group}.json').read_text())['cases']};trs={r['key_index']:r for r in truth['cases'][group]['rows']}
        cm=y if meta['language']=='yiddish' else g;wm=g if cm==y else y
        for r in rs:
            k=r['key_index']
            for arm,ck,tk in [('primary','audit_cipher','audit_truth'),('n2','n2_audit_cipher','n2_audit_truth')]:
                for mid in ('A','B'):
                    rec=c.rec_score(trs[k][tk],pcs[k][ck],r[arm][mid]['mapping']);r[arm][mid].update(rec);r[arm][mid]['m0_pass']=rec['atom_recovery']>=.9 and rec['word_recovery']>=.8
            for arm in ('primary','n1','n2'):
                r[arm]['contrast']=v.z_ind_from_arm(r[arm],y,g);assert r[arm]['contrast']['z_ind'] is not None
        fr={**meta,'truth':meta['language'],'group':group}
        for arm in ('primary','n1','n2'):
            vals=[r[arm]['contrast'] for r in rs];z=statistics.median(x['z_ind'] for x in vals)
            fr[arm]={'median_z_ind':z,'median_D':statistics.median(x['D'] for x in vals),'median_null_sd':statistics.median(x['denom'] for x in vals),'call':v.call(z)}
        fr['correct_m0_passes']=sum(r['primary'][cm]['m0_pass'] for r in rs);fr['wrong_m0_passes']=sum(r['primary'][wm]['m0_pass'] for r in rs)
        fr['median_atom_recovery']=statistics.median(r['primary'][cm]['atom_recovery'] for r in rs);fr['median_word_recovery']=statistics.median(r['primary'][cm]['word_recovery'] for r in rs)
        families.append(fr);allrows.extend(rs)
    summaries={}
    for arm in ('primary','n1','n2'):
        ff=[{'truth':f['truth'],'median_z_ind':f[arm]['median_z_ind']} for f in families];summaries[arm]=v.exact_label_null(ff)
    p=summaries['primary'];n=len(families)
    loo=[]
    for i in range(n):
        ff=[{'truth':f['truth'],'median_z_ind':f['primary']['median_z_ind']} for j,f in enumerate(families) if j!=i];loo.append(v.exact_label_null(ff))
    gates={'correct_language_recovery':all(f['correct_m0_passes']>=29 for f in families),'wrong_language_selectivity':all(f['wrong_m0_passes']<29 for f in families),'deployable_accuracy':p['accuracy']>=.8,'matched_label_null_2sd':p['effect_over_null_sd'] is not None and p['effect_over_null_sd']>=2,'beats_unigram_by_2_errors':round(n*p['accuracy'])>=round(n*summaries['n1']['accuracy'])+2,'beats_shuffle_by_2_errors':round(n*p['accuracy'])>=round(n*summaries['n2']['accuracy'])+2,'complete_non_degenerate':len(allrows)==512,'leave_one_family_out_stable':all(x['accuracy']>=.8 and x['effect_over_null_sd'] is not None and x['effect_over_null_sd']>=2 for x in loo)}
    status='WB03_DEVELOPMENT_PASS_CONFIRMATION_REQUIRED' if all(gates.values()) else 'WB03_DEVELOPMENT_NOT_RESOLVED'
    summary={'status':status,'gates':gates,'arms':summaries,'families':families,'solver_file_sha256':hashes,'target_loaded':False,'scope':'EXPOSED_FAMILIES_DEVELOPMENT_ONLY','leave_one_out':loo}
    c.atomic_json(summary,out/'summary.json');c.atomic_pickle({'summary':summary,'rows':allrows},out/'checkpoint.pkl')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--solutions',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.data,a.solutions,a.out)
