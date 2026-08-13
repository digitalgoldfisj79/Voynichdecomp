#!/usr/bin/env python3
"""Post-target completion wrapper for LLCT v0.1.

The original runner produced all controls and primary F/R statistics, then failed
because some charter-block bootstrap replicates had zero observed exact repeats,
making a preregistered log-ratio distance undefined. This wrapper changes no
statistic, target, threshold, seed, or primary calculation. It only makes that
undefined state explicit: H2 is conservatively marked unresolved if ANY of the
1000 H2 bootstrap replicates is non-finite. H3 is computed independently from
its preregistered ED1 contrast with no continuity correction.
"""
import math
import numpy as np
import run_llct_formulaic_profile as base


def _finite_positive_profile(v):
    return all(math.isfinite(v[f]) and v[f] > 0 for f in base.FEATURES)


def safe_bootstrap(rows, seed, nboot=1000):
    rng = np.random.default_rng(seed)
    charters = list(rows)
    point = base.ratios_from_rows(rows)
    dF = base.distance(point['F'])
    dR = base.distance(point['R'])

    finite_h2 = 0
    less = 0
    invalid_h2 = 0
    deltas = []
    invalid_h3 = 0

    for _ in range(nboot):
        chosen = rng.choice(charters, size=len(charters), replace=True).tolist()
        r = base.ratios_from_rows(rows, chosen)

        if _finite_positive_profile(r['F']) and _finite_positive_profile(r['R']):
            df, dr = base.distance(r['F']), base.distance(r['R'])
            finite_h2 += 1
            less += int(df < dr)
        else:
            invalid_h2 += 1

        f_ed1, r_ed1 = r['F']['ED1_N0'], r['R']['ED1_N0']
        if (math.isfinite(f_ed1) and math.isfinite(r_ed1)
                and f_ed1 > 0 and r_ed1 > 0):
            deltas.append(math.log(f_ed1) - math.log(r_ed1))
        else:
            invalid_h3 += 1

    delta0 = math.log(point['F']['ED1_N0']) - math.log(point['R']['ED1_N0'])
    if not deltas:
        lo = hi = float('nan')
        H3 = 'UNRESOLVED_BOOTSTRAP_ZERO'
    else:
        lo, hi = np.quantile(deltas, [.025, .975])
        if invalid_h3:
            H3 = 'UNRESOLVED_BOOTSTRAP_ZERO'
        elif delta0 > 0 and lo > 0:
            H3 = 'SUPPORT'
        elif delta0 < 0 and hi < 0:
            H3 = 'OPPOSITE'
        else:
            H3 = 'UNRESOLVED'

    finite_p = (less / finite_h2) if finite_h2 else float('nan')
    conditional_H2 = (
        'SUPPORT' if (finite_h2 and dF <= .90*dR and finite_p >= .95)
        else ('OPPOSITE' if (finite_h2 and dR <= .90*dF and (1-finite_p) >= .95)
              else 'UNRESOLVED')
    )
    H2 = conditional_H2 if invalid_h2 == 0 else 'UNRESOLVED_ZERO_BOOTSTRAP'

    return {
        'analytic_point_ratios': point,
        'd_F': dF,
        'd_R': dR,
        'd_F_over_d_R': dF/dR,
        'h2_bootstrap_total': nboot,
        'h2_bootstrap_finite': finite_h2,
        'h2_bootstrap_invalid_zero_ratio': invalid_h2,
        'boot_p_dF_lt_dR_conditional_finite': finite_p,
        'H2_conditional_finite_descriptive': conditional_H2,
        'delta_ED1': delta0,
        'delta_ED1_ci95': [float(lo), float(hi)],
        'h3_bootstrap_total': nboot,
        'h3_bootstrap_valid': len(deltas),
        'h3_bootstrap_invalid_zero_ratio': invalid_h3,
        'H2': H2,
        'H3': H3,
        'post_target_handling': 'No continuity correction; H2 unresolved if any log-ratio bootstrap replicate is undefined.'
    }


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


_original_json_dump = base.json.dump

def _safe_json_dump(obj, fp, *args, **kwargs):
    kwargs.setdefault('default', _json_default)
    return _original_json_dump(obj, fp, *args, **kwargs)


base.bootstrap = safe_bootstrap
base.json.dump = _safe_json_dump
if __name__ == '__main__':
    base.main()
