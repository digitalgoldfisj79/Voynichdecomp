#!/usr/bin/env python3
"""Deterministic smoke/unit tests for EDLC1 core invariants."""
import urllib.request, collections
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/2c9c1a848f1987d7b2dc8634449cfa5b010be9fe/research/edlc1_core_20261006.py"
m={"__name__":"edlc1_core"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)

assert m["glen"]("c̃")==1, m["graphemes"]("c̃")
assert m["clean_surface"]("(Abc,)")=="abc"

fc=collections.Counter({"cat":5,"bat":4,"cats":3,"dog":2})
g=m["ed_graph_from_counter"](fc,3,3)
assert g["V"]==3
assert g["pair_count_ed1"]==2, g
assert g["pair_count_ed2"]>=2

r=[
 {"form":"cat","block":"A"},{"form":"cat","block":"A"},
 {"form":"bat","block":"B"},{"form":"cats","block":"C"},
 {"form":"dog","block":"D"}
]
s=m["corpus_summary"](r)
assert s["tokens"]==5 and s["types"]==4
assert s["length_token"]["mean"]>0

morph=[
 {"form":"ein","block":"1","lemma":"ein","lemma_verified":True},
 {"form":"einem","block":"1","lemma":"ein","lemma_verified":True},
 {"form":"sein","block":"1","lemma":"sein","lemma_verified":True},
]*3
z=m["morphology_decomposition"](morph,1)
assert z["all_lemmas"]["2"]["edge_counts"].get("SAME_LEMMA",0)>=1


# Capacity-matched OOV routine should execute and retain finite common support.
A=[{"form":x,"block":b} for b,xs in {"A":["cat","bat","cats"],"B":["cat","rat","dog"],"C":["bat","dogs","fog"],"D":["cat","bog","rats"],"E":["bat","logs","cat"]}.items() for x in xs]
B=[{"form":x,"block":b} for b,xs in {"A":["sun","son","sons"],"B":["sun","run","day"],"C":["son","days","ray"],"D":["sun","say","runs"],"E":["son","rays","sun"]}.items() for x in xs]
mo=m["matched_oov_repair"](A,B,nrep=3,seed=17)
assert mo["nrep"]==3

print("EDLC1_TESTS_OK")
