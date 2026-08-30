#!/usr/bin/env python3
import argparse, json, pickle
from pathlib import Path
import numpy as np
import taccola_calibration_v01b as core

ap=argparse.ArgumentParser(); ap.add_argument('--input',default='one_inputs'); args=ap.parse_args()
indir=Path(args.input); outdir=Path('taccola_complete_result'); outdir.mkdir(exist_ok=True)
feats={}; page_diag={}; errors=[]; provenance={}
files=sorted(indir.rglob('one_*.pkl'))
if not files: raise RuntimeError('no one-witness bundles')
for p in files:
    with open(p,'rb') as f: b=pickle.load(f)
    if b['panel_sha256'] != core.EXPECTED_PANEL_SHA256: raise RuntimeError(f'panel hash mismatch in {p}')
    mid=b['id']
    if mid in feats: raise RuntimeError(f'duplicate witness bundle: {mid}')
    feats[mid]=b['features']; page_diag[mid]=b.get('page_diag',[]); errors.extend(b.get('errors',[]))
    if 'transport_repair' in b: provenance[mid]=b['transport_repair']
missing_ids=[m for m in core.MANUSCRIPTS if m not in feats]
extra_ids=[m for m in feats if m not in core.MANUSCRIPTS]
if missing_ids or extra_ids: raise RuntimeError(f'coverage mismatch missing={missing_ids} extra={extra_ids}')

original_null=list(core.NULL_IDS)
required=['clm197','pal766']+core.TECH_IDS+core.ITALY_IDS
missing=[k for k in required if len(feats.get(k,[]))<12]
active=[k for k in original_null if len(feats.get(k,[]))>=12]
audit={'required_missing_or_lt12':missing,'original_n_null':len(original_null),'active_n_null':len(active),'failed_nulls':[k for k in original_null if k not in active],'image_errors':len([e for e in errors if e.get('stage')=='image']),'manifest_errors':len([e for e in errors if e.get('stage')=='manifest']),'executor':'37_independent_jobs','transport_repairs':provenance,'feature_counts':{k:len(v) for k,v in feats.items()}}
if missing or len(active)<30 or any(k not in active for k in (core.TECH_IDS+core.ITALY_IDS)):
    audit['calibration_allowed']=False
    result={'version':core.VERSION,'panel_sha256':core.EXPECTED_PANEL_SHA256,'audit':audit,'status':'CALIBRATION_BLOCKED_INCOMPLETE','errors':errors[:200]}
    (outdir/'calibration_complete.json').write_text(json.dumps(result,indent=2))
    raise SystemExit('Calibration blocked by completeness gate')
audit['calibration_allowed']=True
core.NULL_IDS=active
allgeom=np.vstack([f['geom'] for mid in feats for f in feats[mid] if mid in (['clm197','pal766','dresden_mixed']+core.NULL_IDS)])
geom_mu=allgeom.mean(axis=0); geom_sd=allgeom.std(axis=0); geom_sd=np.where(geom_sd<1e-6,1.0,geom_sd)
d1=core.build_direction('clm197','pal766',feats,geom_mu,geom_sd)
d2=core.build_direction('pal766','clm197',feats,geom_mu,geom_sd)
reps=['hog','chamfer','geometry']
rep_pass=[r for r in reps if min(d1['representation_stats'][r]['z'],d2['representation_stats'][r]['z'])>=2.0]
nondeg=False
for i,r1 in enumerate(rep_pass):
    for r2 in rep_pass[i+1:]:
        key=f'{r1}__{r2}' if f'{r1}__{r2}' in d1['null_rep_correlations'] else f'{r2}__{r1}'
        if abs(d1['null_rep_correlations'].get(key,1.0))<.90 and abs(d2['null_rep_correlations'].get(key,1.0))<.90: nondeg=True
composite_gate=all([d1['composite_stats']['z']>=2.0,d2['composite_stats']['z']>=2.0,d1['composite_stats']['empirical_block_p']<=.05,d2['composite_stats']['empirical_block_p']<=.05,d1['bootstrap']['fraction_composite_z_ge_2']>=.80,d2['bootstrap']['fraction_composite_z_ge_2']>=.80])
tech_gate=(d1['composite_stats']['technical_rank']==1 and d2['composite_stats']['technical_rank']==1)
fragility=all(v['z']>=1.5 for d in (d1,d2) for v in d['sensitivity_top_matches'].values())
passed=(len(rep_pass)>=2 and nondeg and composite_gate and tech_gate and fragility)
result={'version':core.VERSION,'release_id':core.RELEASE_ID,'panel_sha256':core.EXPECTED_PANEL_SHA256,'audit':audit,'direction_clm197_to_pal766':d1,'direction_pal766_to_clm197':d2,'gate':{'passing_representations_both_directions':rep_pass,'nondegenerate_passing_pair':nondeg,'composite_gate':composite_gate,'technical_rank_gate':tech_gate,'decision_rule_fragility_gate':fragility,'calibration_passed':passed,'q13_unseal_allowed':passed,'criteria':{'rep_z_min':2.0,'min_passing_reps':2,'max_abs_null_rep_corr':.90,'composite_z_min':2.0,'block_p_max':.05,'bootstrap_stability_min':.80,'technical_rank_required':1,'sensitivity_z_floor':1.5}},'errors':errors[:200]}
(outdir/'calibration_complete.json').write_text(json.dumps(result,indent=2))
with open(outdir/'checkpoint_03_complete.pkl','wb') as f: pickle.dump(result,f,protocol=5)
lines=['# Taccola × Q13 — Complete v0.1b Calibration','','## RETRACTED FINDINGS','','- **RETRACTED:** the raw Siena/Taccola concentration (4/12 vs 5/185) is not independent evidence. After family collapse it is 1/12 vs 5/185; effect +5.63 percentage points, exact conditional null SD 5.13 points, 1.10 null SD, Fisher p=0.318. The metric does not resolve this.','','Q13 remained sealed throughout calibration. Palatino 766 and LJS 419 used pre-score transport-only r15 fallbacks; all scientific scoring functions are the audited frozen v0.1 core.','']
for title,d in [('Clm197 → Pal766',d1),('Pal766 → Clm197',d2)]:
    c=d['composite_stats']; prefix='the metric does not resolve this — ' if abs(c['z'])<2 else ''
    lines.append(f"- **{title}:** {prefix}composite effect {c['effect']:.4f}, null SD {c['null_sd']:.4f}, {c['z']:.2f} null SD, block p={c['empirical_block_p']:.4f}; technical rank {c['technical_rank']}/{len(core.TECH_IDS)+1}; bootstrap stability={d['bootstrap']['fraction_composite_z_ge_2']:.3f}.")
    for r in reps:
        s=d['representation_stats'][r]; pfx='the metric does not resolve this — ' if abs(s['z'])<2 else ''
        lines.append(f"  - {r}: {pfx}effect {s['effect']:.4f}, null SD {s['null_sd']:.4f}, {s['z']:.2f} null SD, p={s['empirical_block_p']:.4f}.")
lines += ['',f"**Passing representations both directions:** `{rep_pass}`",f"**Nondegenerate passing pair:** `{nondeg}`",f"**Composite gate:** `{composite_gate}`",f"**Technical-rank gate:** `{tech_gate}`",f"**Decision-rule fragility gate:** `{fragility}`",f"**Calibration passed:** `{passed}`",f"**Q13 unseal allowed:** `{passed}`"]
(outdir/'RUNNING_RESULTS_COMPLETE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'event':'complete_calibration','calibration_passed':passed,'q13_unseal_allowed':passed,'active_nulls':len(core.NULL_IDS),'passing_reps':rep_pass,'d1_z':d1['composite_stats']['z'],'d2_z':d2['composite_stats']['z']}),flush=True)
