#!/usr/bin/env python3
"""Conservative source overlap screens, isolated from model training/scoring."""
import argparse
import collections
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path
from diagnose import c


def fold(w):
    # Deliberately lossy retrieval key, NEVER a solver representation.
    w = re.sub(r'sch|sh', 's', w)
    w = re.sub(r'kh|ch', 'k', w)
    w = re.sub(r'th', 't', w)
    w = re.sub(r'tz|ts|z|c', 's', w)
    w = re.sub(r'[fvw]', 'v', w)
    w = re.sub(r'[aeiouyj]', '', w)
    return re.sub(r'(.)\1+', r'\1', w) or '~'


def run(root, penn):
    source = pickle.loads(gzip.decompress((root/'edition_preflight.pkl.gz').read_bytes()))
    edition = {k: v['variants']['0.31']['words'] for k, v in source['sources'].items()}
    query = {name: {'literal': c.ngrams(words, 8),
                    'consonant': c.ngrams([fold(w) for w in words], 8)}
             for name, words in edition.items()}
    hits = {name: {'literal': [], 'consonant': []} for name in edition}
    files = {}
    for p in sorted(penn.glob('*.psd')):
        words = c.penn_words(p)
        files[p.name] = {'sha256': c.sha_file(p), 'words': len(words)}
        for mode, ws in [('literal', words), ('consonant', [fold(w) for w in words])]:
            grams = c.ngrams(ws, 8)
            for name in edition:
                overlap = query[name][mode] & grams
                if overlap:
                    hits[name][mode].append({'penn_file': p.name, 'shared_8gram_types': len(overlap),
                                             'grams': sorted(overlap)})
    # Positive retrieval control is drawn from each actual edition. This tests
    # code operation, not sensitivity to every possible historical respelling.
    controls = {}
    for name, words in edition.items():
        controls[name] = {}
        for mode, stream in [('literal', words), ('consonant', [fold(w) for w in words])]:
            planted = c.ngrams(stream[100:140], 8)
            assert len(planted) == len(planted & query[name][mode])
            controls[name][mode] = len(planted)
    result = {'status': 'SCREEN_COMPLETE_REQUIRES_MATERIALITY_REVIEW',
              'target_loaded': False, 'model_loaded': False,
              'penn_files': files, 'hits': hits, 'positive_self_retrieval_controls': controls,
              'limitations': ['Lossy consonant fold is only a near-overlap screen, not a phonological or YIVO conversion.',
                             'A negative screen cannot prove independence of translations with common biblical sources.',
                             'Edition footnotes and their Zene-u-Rene quotations are excluded from all candidate streams.'],
              'null_sd': None, 'null_sd_reason': 'Deterministic corpus-intersection census, not a random-null effect estimate.'}
    (root/'edition_overlap.json').write_text(json.dumps(result, indent=2)+'\n')
    (root/'edition_overlap.pkl.gz').write_bytes(gzip.compress(pickle.dumps(result), mtime=0))
    print(json.dumps({'files_screened': len(files), 'hits': hits, 'controls': controls}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--penn', type=Path, required=True)
    a = ap.parse_args()
    run(a.root, a.penn)
