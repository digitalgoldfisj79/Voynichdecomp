from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trials',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    files=sorted(a.trials.glob('U5A_trial_*.json'))
    if len(files)!=20:
        raise SystemExit(f'expected exactly 20 locked trial files, got {len(files)}')
    rows=[];seen=set()
    for p in files:
        x=json.loads(p.read_text(encoding='utf-8'))
        if x.get('target_opened') is not False or x.get('voynich_read') is not False:
            raise SystemExit(f'contaminated trial metadata in {p}')
        key=(x['language'],int(x['trial']))
        if key in seen: raise SystemExit(f'duplicate trial {key}')
        seen.add(key);rows.append(x['result'])
    expected={(iso,i) for iso in ('la','it') for i in range(10)}
    if seen!=expected: raise SystemExit(f'locked trial set mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}')
    rows.sort(key=lambda r:(r['language'],r['trial']))
    mean=statistics.fmean(float(r['accuracy']) for r in rows)
    med=statistics.median(float(r['accuracy']) for r in rows)
    count=sum(float(r['accuracy'])>=0.75 for r in rows)
    base=statistics.fmean(float(r['baseline_accuracy']) for r in rows)
    verdict='PASS_RECOVERY_CALIBRATION' if mean>=0.85 and count>=16 else 'FAIL_RECOVERY_CALIBRATION'
    by={}
    for iso in ('la','it'):
        z=[r for r in rows if r['language']==iso]
        by[iso]={'trials':len(z),'mean_accuracy':statistics.fmean(r['accuracy'] for r in z),'median_accuracy':statistics.median(r['accuracy'] for r in z),'trials_ge_075':sum(r['accuracy']>=0.75 for r in z),'baseline_mean':statistics.fmean(r['baseline_accuracy'] for r in z)}
    out={
      'schema':'frontier-u5-a-v0.1','formal_verdict':verdict,'target_opened':False,'voynich_read':False,
      'trials':20,'mean_accuracy':mean,'median_accuracy':med,'trials_ge_075':count,'baseline_mean':base,
      'required_mean':0.85,'required_trials_ge_075':16,'iterations':700000,'restarts':50,'length':384,
      'by_language':by,'rows':rows,
      'consequence':'U5-B recognition may open; Voynich remains sealed' if verdict=='PASS_RECOVERY_CALIBRATION' else 'U5 closes under v0.1; Voynich remains sealed',
    }
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'U5A_RECOVERY_RESULT.json').write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
    md=['# U5-A fresh-key verbose recovery result','',f'Formal verdict: **{verdict}**','',f'- mean normalized recovery: **{mean:.4f}** (gate ≥0.85)',f'- trials ≥0.75: **{count}/20** (gate ≥16/20)',f'- median recovery: {med:.4f}',f'- frequency baseline mean: {base:.4f}','', 'Voynich target remained sealed throughout U5-A.']
    for iso,label in [('la','Latin / locked Pliny'),('it','Italian / locked Dante')]:
        q=by[iso];md.append(f'- {label}: mean {q["mean_accuracy"]:.4f}; ≥0.75 {q["trials_ge_075"]}/{q["trials"]}')
    (a.out/'U5A_RESULT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('U5A_AGGREGATE_FINAL',json.dumps({k:out[k] for k in ('formal_verdict','mean_accuracy','trials_ge_075','baseline_mean','target_opened')},sort_keys=True),flush=True)

if __name__=='__main__':main()
