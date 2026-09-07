#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from huggingface_hub import HfApi
import memory_recurrence_score_v1 as score

CLASSES=score.CLASSES
CONTROLS=score.CONTROLS

def fetch(api,jobs):
    texts=[]
    for jid in jobs:
        info=api.inspect_job(job_id=jid)
        if 'COMPLETED' not in str(info.status.stage).upper():
            raise RuntimeError(f'job not completed: {jid} {info.status.stage}')
        texts.append(''.join(api.fetch_job_logs(job_id=jid,follow=False)))
    return texts

def parse(texts):
    out=[]
    for text in texts: out.extend(score.parse_marker_text(text))
    return out

def exact_index(objs,split,sources,n):
    d=score.index_trials(objs,split,sources,n)
    expected={(s,t,None) for s in sources for t in range(n)}
    got=set(d)
    if got!=expected: raise RuntimeError(f'{split} exact set failure n={len(got)} missing={sorted(expected-got)[:10]} extra={sorted(got-expected)[:10]}')
    for o in d.values():
        if o.get('target_opened') is not False: raise RuntimeError(f'{split} target exposure flag')
        if len(o.get('folds',[]))!=5: raise RuntimeError(f'{split} fold count')
        for f in o['folds']:
            for rep in score.REPS:
                if len(f['features'][rep])!=20: raise RuntimeError(f'{split} feature count')
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--development',nargs='+',required=True)
    ap.add_argument('--calibration',nargs='+',required=True)
    ap.add_argument('--validation',nargs='+',required=True)
    ap.add_argument('--controls',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); api=HfApi()
    groups=[a.development,a.calibration,a.validation,a.controls]
    flat=[x for g in groups for x in g]
    if len(flat)!=len(set(flat)): raise RuntimeError('job id reused across registered splits')
    objs=parse(fetch(api,flat))
    dev=exact_index(objs,'development',CLASSES,80)
    cal=exact_index(objs,'calibration',CLASSES,40)
    val=exact_index(objs,'validation',CLASSES,50)
    ctrl=exact_index(objs,'control',CONTROLS,50)
    summary=score.summarize(dev,cal,val,ctrl,power_complete=False,power_report=None)
    Path(a.output).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    compact={'manifest_sha256':summary['manifest_sha256'],'source_validation':summary['source_validation'],
             'pairwise_separations':summary['pairwise_separations'],'unknown_controls':summary['unknown_controls'],
             'representation_checks':summary['representation_checks'],'leakage_checks':summary['leakage_checks'],
             'calibration_checks':summary['calibration_checks'],'power_reporting':summary['power_reporting'],
             'summary_sha256':summary['summary_sha256']}
    print('P1_PRIMARY='+json.dumps(compact,separators=(',',':')))
if __name__=='__main__': main()
