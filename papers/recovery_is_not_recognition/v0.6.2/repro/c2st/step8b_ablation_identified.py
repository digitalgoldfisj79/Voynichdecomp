"""Eight-feature ablation arm for the recovered step8b C2ST harness.
Subset identified 2026-08-05 by pre-registered exhaustive search over all C(13,8)=1287
subsets against the S3 ablation column (unique hit at max|delta|<=0.005; achieved 0.002;
next-best 0.009). Environment at identification: sklearn 1.8.0, numpy 2.4.4.

KEPT (8):    wl_mean, wl_std, wl_autocorr, ttr, hapax, H1, H2, chardist_max
DROPPED (5): digcov, within_tok_nextH, opener_gallows, charpos_gallows, adj_repeat

Requires candidates.pkl from the recovered step8a_candidates.py and the feats()/chunks_of()
definitions from the recovered step8b_eval_harness.py (imported here by exec to avoid
divergence). Run AFTER step8a; expects the step8b working directory layout.
Reference values (S3 eight-feature column): real_B 0.421, line-shuffle 0.844,
word-shuffle 0.955, gen_template_v10 0.988, delex_char3 0.970.
This file is a labelled reconstruction of the ablation ARM; the 13-feature harness itself
is the transcript-recovered original.
"""
import pickle, csv, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

WORK = '/tmp/vms/work'
KEEP = [0, 1, 2, 3, 4, 5, 6, 7]   # indices into the 13-feature vector of recovered step8b
FN = ['wl_mean','wl_std','wl_autocorr','ttr','hapax','H1','H2','chardist_max','digcov',
      'within_tok_nextH','opener_gallows','charpos_gallows','adj_repeat']
S3_REF = {'real_B':0.421,'line-shuffle':0.844,'word-shuffle':0.955,
          'gen_template_v10':0.988,'delex_char3':0.970}

# Reuse the recovered harness's own feats/chunks_of so the featurisation cannot drift.
src = open(f'{WORK}/step8b_eval_harness.py').read()
ns = {}
exec(src.split('ref=featmatrix')[0], ns)          # defs + candidate load only, no run
featmatrix = ns['featmatrix']

ref = featmatrix('real_A')
rng = np.random.default_rng(0)                     # same seed + same loop order as full run
rows = []
print(f"{'candidate':18s} {'8-feat AUC':>10s} {'S3':>7s} {'delta':>7s}")
for name in ['real_B','line-shuffle','word-shuffle','gen_template_v10','delex_char3']:
    Xc = featmatrix(name)
    n = min(len(ref), len(Xc))
    ri = rng.choice(len(ref), n, replace=False)
    ci = rng.choice(len(Xc), n, replace=False)
    X = np.vstack([ref[ri][:, KEEP], Xc[ci][:, KEEP]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    d = aucs.mean() - S3_REF[name]
    rows.append([name, f"{aucs.mean():.3f}", f"{aucs.std():.3f}", S3_REF[name], f"{d:+.3f}"])
    print(f"{name:18s} {aucs.mean():9.3f} {S3_REF[name]:7.3f} {d:+7.3f}")

with open(f'{WORK}/ablation_identified_results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['candidate','auc_8feat_mean','auc_8feat_sd','s3_reference','delta'])
    w.writerows(rows)
print("\nOK -> ablation_identified_results.csv")
print("Kept:", [FN[i] for i in KEEP])
print("Dropped:", [FN[i] for i in range(13) if i not in KEEP])
