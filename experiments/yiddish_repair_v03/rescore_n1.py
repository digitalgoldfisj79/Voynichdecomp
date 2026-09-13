#!/usr/bin/env python3
"""Fair nuisance-only rescore; preserve original rows and every recovery result."""
import argparse,json,pickle,copy
from pathlib import Path
from diagnose import c

def run(data,models_path,source,out):
    models=pickle.load(models_path.open('rb'));manifest={}
    for p in sorted(source.glob('G*/rows.jsonl')):
        group=p.parent.name;pcs={r['key_index']:r for r in json.load((data/f'public/{group}.json').open())['cases']};rows=[json.loads(l) for l in p.read_text().splitlines()]
        for r in rows:
            unchanged=json.dumps([r['primary'],r['n2']],sort_keys=True);k=r['key_index'];pc=pcs[k];_,uni=c.base.cipher_counts(pc['fit_cipher'])
            for mid,model in models.items():
                m=c.base.frequency_initial(uni,model['train_uni']);um=(model['train_uni'],sum(model['train_uni']))
                r['n1'][mid]=c.z_unigram(pc['audit_cipher'],m,um,c.random_maps('n1',group,k));r['n1'][mid]['mapping']=m
            assert json.dumps([r['primary'],r['n2']],sort_keys=True)==unchanged
        dest=out/group;dest.mkdir(parents=True,exist_ok=True);f=dest/'rows.jsonl';f.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        manifest[group]={'original_sha256':c.sha_file(p),'corrected_sha256':c.sha_file(f),'primary_n2_unchanged':True}
    c.atomic_json(manifest,out/'n1_fairness_manifest.json')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--models',type=Path,required=True);a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();run(x.data,x.models,x.source,x.out)
