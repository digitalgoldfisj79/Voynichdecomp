from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import u5_naibbe_recovery as u5


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--legacy-repo',type=Path,required=True)
    ap.add_argument('--naibbe-repo',type=Path,required=True)
    ap.add_argument('--language',choices=('la','it'),required=True)
    ap.add_argument('--trial',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not (0 <= a.trial < u5.N_TRIALS_PER_LANGUAGE):
        raise SystemExit('trial index outside frozen 0..9 range')
    a.out.mkdir(parents=True,exist_ok=True)
    mono=u5.load_legacy_solver(a.legacy_repo)
    tables=u5.load_tables(a.naibbe_repo/'references'/'naibbe_tables.csv')
    cache=a.out/'source_cache';cache.mkdir(exist_ok=True)
    train=u5.fetch_text(u5.TRAIN_URLS[a.language],cache/f'train_{a.language}.txt')
    language,ntrain=u5.build_language(train)
    model=mono.build_language_model(language)
    locked=u5.normalize((a.naibbe_repo/u5.TEST_FILES[a.language]).read_text(encoding='utf-8',errors='ignore'))
    starts,chunks=u5.deterministic_chunks(locked,u5.N_TRIALS_PER_LANGUAGE,u5.LENGTH)

    # Compile once outside the measured trial.
    mono.anneal_mono(np.asarray([0,1,0,1,0,1],dtype=np.int32),np.arange(23,dtype=np.int32),model[0],model[1],2,1,1)
    row=u5.trial_job(a.language,a.trial,chunks[a.trial],language,model,mono,tables)
    payload={
        'schema':'frontier-u5-a-trial-v0.1',
        'protocol_commit_floor':'4e086132cfcca15b427fbb578b929efc6ae20fe2',
        'target_opened':False,'voynich_read':False,
        'language':a.language,'trial':a.trial,'locked_chunk_start':starts[a.trial],
        'training_normalized_chars':ntrain,
        'training_url':u5.TRAIN_URLS[a.language],
        'locked_test_file':u5.TEST_FILES[a.language],
        'iterations':u5.ITERATIONS,'restarts':u5.RESTARTS,'length':u5.LENGTH,
        'result':row,
    }
    out=a.out/f'U5A_trial_{a.language}_{a.trial:02d}.json'
    out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print('U5A_SINGLE_FINAL',json.dumps(payload,sort_keys=True),flush=True)

if __name__=='__main__':
    main()
