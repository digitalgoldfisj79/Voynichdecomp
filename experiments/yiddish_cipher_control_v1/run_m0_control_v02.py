#!/usr/bin/env python3
"""Yiddish cipher control v0.2: label-equivariant wrapper around frozen S1 search.

Development repair motivated by the non-qualifying v0.1 smoke MR1 failure.
No full qualification outcome and no Voynich target was exposed before this repair.

The only algorithmic change is a representation canonicalisation before search:
ciphertext symbols are renamed by order of first occurrence in the FIT ciphertext.
A global bijective relabelling of ciphertext symbols therefore produces the exact
same canonical fit instance and the exact same stochastic search trajectory.
The returned mapping is conjugated back to the caller's original symbol labels.

This wrapper does not change the Yiddish LM, objective, search budget, acceptance
thresholds, source roles, or target policy.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_m0_control as h  # noqa: E402

_RAW_SOLVER = h.v03.blind_solve_budget


def _canonicalise_fit(words):
    """Return canonical words and original-id -> canonical-id mapping.

    Seen symbols are assigned 0,1,... by first occurrence in the ciphertext stream.
    This ordering is invariant to any global renaming of the external symbol labels.
    Unseen symbols are assigned remaining canonical IDs only to complete the mapping;
    they carry no fit evidence and are reported implicitly through fit coverage.
    """
    seen_order=[]
    seen=set()
    for w in words:
        for c in w:
            if c == '~':
                continue
            oi=ord(c)-97
            if oi not in seen:
                seen.add(oi); seen_order.append(oi)
    remaining=[i for i in range(h.A) if i not in seen]
    order=seen_order+remaining
    orig_to_canon={orig:canon for canon,orig in enumerate(order)}
    canon_words=[]
    for w in words:
        chars=[]
        for c in w:
            if c == '~': chars.append(c)
            else: chars.append(h.ALPHABET[orig_to_canon[ord(c)-97]])
        canon_words.append(''.join(chars))
    return canon_words, orig_to_canon, len(seen_order)


def label_equivariant_blind_solve(fit_cipher, logp, train_uni, search_seed, cfg):
    canon_words, orig_to_canon, n_seen = _canonicalise_fit(fit_cipher)
    canon_map, score = _RAW_SOLVER(canon_words, logp, train_uni, search_seed, cfg)
    # Convert canonical-cipher -> plaintext mapping back to external-cipher -> plaintext.
    external_map=[0]*h.A
    for orig,canon in orig_to_canon.items():
        external_map[orig]=canon_map[canon]
    return external_map, score


# Monkey-patch only the solver entry point used by the v0.1 harness. All controls,
# scores and acceptance rules remain unchanged, so this isolates the repair axis.
h.v03.blind_solve_budget = label_equivariant_blind_solve
h.CONTROL_VERSION = "yiddish_cipher_instrument_control_v02_label_equivariant_20260912"
h.S1 = {**h.S1, "id":"S1E_FROZEN"}
h.SMOKE_CFG = {**h.SMOKE_CFG, "id":"SMOKE_EQUIVARIANT_NOT_QUALIFYING"}

if __name__ == '__main__':
    h.main()
