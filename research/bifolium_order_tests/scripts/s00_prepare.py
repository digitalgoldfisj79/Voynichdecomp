#!/usr/bin/env python3
"""Stage 0: turn the shipped plain-text inputs into the working files the analysis scripts read.
 data/words_by_page.csv.gz -> data/s1_words.pkl  (dict with key 'meta': one row per word token)
 data/controls/*.txt.gz    -> data/controls/ci_corpus_parsed.pkl, greek_corpus_parsed.pkl  ({'all_words': [...]})
 optional: --slim downloads voynich_transcriptions_slim.json (12 transliterations) from the Voynichdecomp repo.
usage: python3 s00_prepare.py ROOT [--slim]"""
import gzip, os, pickle, sys, urllib.request
import pandas as pd
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data")
meta = pd.read_csv(os.path.join(D, "words_by_page.csv.gz"), dtype={"folio": str, "word": str})
pickle.dump({"meta": meta}, open(os.path.join(D, "s1_words.pkl"), "wb"))
for src, dst in (("circa_instans_words.txt.gz", "ci_corpus_parsed.pkl"), ("galen_words.txt.gz", "greek_corpus_parsed.pkl")):
    words = gzip.open(os.path.join(D, "controls", src), "rt").read().split("\n")
    pickle.dump({"all_words": words}, open(os.path.join(D, "controls", dst), "wb"))
print("tokens", len(meta), "pages", meta.folio.nunique())
if "--slim" in sys.argv:
    url = ("https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
           "b1a1a36589437093c6f7e7432dae20ab05947b69/voynich_transcriptions_slim.json")
    urllib.request.urlretrieve(url, os.path.join(D, "voynich_transcriptions_slim.json")); print("slim json downloaded")
