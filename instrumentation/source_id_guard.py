#!/usr/bin/env python3
import argparse, json, pathlib, sys
from typing import Any, Dict
import assay_guard as base


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def freeze(manifest: Dict[str, Any], out: str) -> Dict[str, Any]:
    required=['assay_id','version','scientific_question','sources','unknown_controls','target','decision','licensed_inference','prohibited_inference']
    miss=[k for k in required if k not in manifest]
    if miss: raise ValueError(f'manifest missing required fields: {miss}')
    if manifest['target'].get('sealed_before_qualification') is not True:
        raise ValueError('target.sealed_before_qualification must be true')
    names=[x['name'] for x in manifest['sources']]
    if len(names)<2 or len(names)!=len(set(names)): raise ValueError('need >=2 unique source names')
    d=manifest['decision']
    for k in ['min_per_source_recall','min_macro_recall','max_per_source_abstain_rate','min_unknown_rejection_rate','min_pairwise_effect_over_null_sd']:
        if k not in d: raise ValueError(f'decision missing {k}')
    rec={'kind':'SOURCE_ID_FREEZE','assay_id':manifest['assay_id'],'version':manifest['version'],
         'manifest_sha256':base.sha256_obj(manifest),'target_sealed':True,'lifecycle_stage':'FROZEN'}
    dump(out,rec); return rec


def qualify(manifest: Dict[str, Any], freeze_rec: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    msha=base.sha256_obj(manifest)
    if freeze_rec.get('manifest_sha256')!=msha: raise ValueError('manifest hash does not match freeze record')
    if summary.get('manifest_sha256')!=msha: raise ValueError('summary manifest_sha256 does not match frozen manifest')
    d=manifest['decision']; names=[x['name'] for x in manifest['sources']]
    rows={x['name']:x for x in summary.get('source_validation',[])}
    if set(rows)!=set(names): raise ValueError('source_validation set mismatch')
    src=[]; recalls=[]; all_src=True
    for name in names:
        x=rows[name]; n=int(x['n']); correct=int(x['correct']); abst=int(x.get('abstained',0))
        if not (0<=correct<=n and 0<=abst<=n): raise ValueError('invalid source counts')
        recall=correct/n; abst_rate=abst/n
        rp=recall>=float(d['min_per_source_recall']); ap=abst_rate<=float(d['max_per_source_abstain_rate'])
        all_src &= rp and ap; recalls.append(recall)
        src.append({'name':name,'n':n,'correct':correct,'recall':recall,'abstained':abst,'abstain_rate':abst_rate,
                    'recall_gate_pass':rp,'abstain_gate_pass':ap,'pass':bool(rp and ap)})
    macro=sum(recalls)/len(recalls); macro_pass=macro>=float(d['min_macro_recall'])

    expected_pairs={tuple(sorted((a,b))) for i,a in enumerate(names) for b in names[i+1:]}
    got={tuple(sorted((x['a'],x['b']))):x for x in summary.get('pairwise_separations',[])}
    if set(got)!=expected_pairs: raise ValueError(f'pairwise separation set mismatch: expected {sorted(expected_pairs)}, got {sorted(got)}')
    pairs=[]; pair_pass=True
    for key in sorted(expected_pairs):
        x=got[key]; eff=float(x['effect']); nsd=float(x['null_sd']); z=base.effect_over_null_sd(eff,nsd)
        p=z>=float(d['min_pairwise_effect_over_null_sd']); pair_pass &= p
        pairs.append({'a':key[0],'b':key[1],'effect':eff,'null_sd':nsd,'effect_over_null_sd':z,'pass':p,
                      'headline':('THE METRIC DOES NOT RESOLVE THIS SOURCE PAIR' if z<2 else 'source pair resolved at >=2 null SD')})

    unames=[x['name'] for x in manifest['unknown_controls']]
    urows={x['name']:x for x in summary.get('unknown_controls',[])}
    if set(urows)!=set(unames): raise ValueError('unknown control set mismatch')
    unknown=[]; unknown_pass=True
    for name in unames:
        x=urows[name]; n=int(x['n']); rejected=int(x['rejected_or_abstained']); rate=rejected/n
        eff=float(x['effect']); nsd=float(x['null_sd']); z=base.effect_over_null_sd(eff,nsd)
        rp=rate>=float(d['min_unknown_rejection_rate']); ep=z>=float(d['min_pairwise_effect_over_null_sd']); p=rp and ep
        unknown_pass &= p
        ci=base.wilson(rejected,n)
        unknown.append({'name':name,'n':n,'rejected_or_abstained':rejected,'rate':rate,'wilson95':[ci['lo'],ci['hi']],
                        'effect':eff,'null_sd':nsd,'effect_over_null_sd':z,'rate_gate_pass':rp,'effect_gate_pass':ep,'pass':p,
                        'headline':('THE METRIC DOES NOT RESOLVE THIS UNKNOWN CONTROL' if z<2 else 'unknown control resolved at >=2 null SD')})

    rep=summary.get('representation_checks',[]); rep_req=bool(d.get('require_representation_robustness',False))
    rep_pass=all(bool(x.get('pass')) for x in rep) if rep else not rep_req
    leak=summary.get('leakage_checks',{}); lk=['target_inaccessible_during_fit','disjoint_seed_namespaces','training_only_model_selection','grouped_trial_splits']
    leak_pass=all(leak.get(k) is True for k in lk)
    overall=bool(all_src and macro_pass and pair_pass and unknown_pass and rep_pass and leak_pass)
    out={'kind':'SOURCE_ID_QUALIFICATION','assay_id':manifest['assay_id'],'version':manifest['version'],'manifest_sha256':msha,
         'summary_sha256':base.sha256_obj(summary),'source_validation':src,'macro_recall':macro,'macro_recall_gate_pass':macro_pass,
         'pairwise_separations':pairs,'unknown_controls':unknown,'representation_gate_pass':rep_pass,'leakage_gate_pass':leak_pass,
         'overall_pass':overall,'target_open_permitted':overall,'lifecycle_stage':('ADVERSARIAL_POWERED' if overall else 'BLOCKED'),
         'licensed_inference':manifest['licensed_inference'] if overall else 'NO TARGET INFERENCE LICENSED',
         'prohibited_inference':manifest['prohibited_inference']}
    out['qualification_sha256']=base.sha256_obj(out); return out


def check_target(manifest, qualification):
    if qualification.get('manifest_sha256')!=base.sha256_obj(manifest): raise ValueError('qualification does not match current manifest')
    if qualification.get('overall_pass') is not True or qualification.get('target_open_permitted') is not True:
        raise PermissionError('TARGET SEALED: source-identification assay has not passed qualification')


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('freeze'); p.add_argument('manifest');p.add_argument('out')
    p=sp.add_parser('qualify');p.add_argument('manifest');p.add_argument('freeze_record');p.add_argument('summary');p.add_argument('out')
    p=sp.add_parser('check-target');p.add_argument('manifest');p.add_argument('qualification')
    a=ap.parse_args()
    try:
        if a.cmd=='freeze': print(json.dumps(freeze(load(a.manifest),a.out),sort_keys=True))
        elif a.cmd=='qualify':
            q=qualify(load(a.manifest),load(a.freeze_record),load(a.summary));dump(a.out,q);print(json.dumps(q,sort_keys=True))
        else: check_target(load(a.manifest),load(a.qualification));print('TARGET_OPEN_PERMITTED')
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr);raise SystemExit(2)
if __name__=='__main__': main()
