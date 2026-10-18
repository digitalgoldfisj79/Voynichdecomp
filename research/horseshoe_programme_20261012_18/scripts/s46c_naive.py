#!/usr/bin/env python3
"""Rule-19 naive baseline for the s46b dependency check: MI(first_k, last_k) vs shuffled, k=1,2, same tokens."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "s46b_checks.py")).read().split("# positive control")[0].split('for nm, C_ in')[0])
for k in (1, 2):
    C_ = Counter()
    for w, c in tc.items():
        if len(w) >= 2 * k: C_[(w[:k], w[-k:])] += c
    m0 = mi(C_); mu, sd = mi_null(C_); print(f"naive first_{k}/last_{k} MI %.3f, shuffled %.3f (sd %.4f), z %.0f, excess %.3f" % (m0, mu, sd, (m0 - mu) / sd, m0 - mu))
