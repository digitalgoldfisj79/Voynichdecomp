#!/usr/bin/env python3
"""Launch Family P with an execution-only branch-typing rewrite."""
from __future__ import annotations

import hashlib
from pathlib import Path

path = Path(__file__).with_name("v060_family_p_stage_a.py")
source = path.read_text(encoding="utf-8")
old = '''        for _ in range(iterations):
            state, move = _rng_int(state, 10)
            changed_shift = move < 3 and period > 1
            if changed_shift:
                state, slot_raw = _rng_int(state, period - 1)
                slot = slot_raw + 1
                old_shift = shifts[slot]
                state, proposal = _rng_int(state, a)
                if proposal == old_shift:
                    continue
                shifts[slot] = proposal
                first = second = -1
            else:
                state, first = _rng_int(state, a)
                state, second = _rng_int(state, a)
                if first == second:
                    continue
                temporary = inverse[first]
                inverse[first] = inverse[second]
                inverse[second] = temporary
                old_shift = -1
                slot = -1
            candidate = score_wheel(cipher, phase, inverse, shifts, trigram_logp, unigram_logp)
            delta = candidate - current_score
            accept = delta >= 0.0
            if not accept:
                state, uniform = _rng_float(state)
                accept = uniform < math.exp(delta / max(temperature, 1e-9))
            if accept:
                current_score = candidate
                if candidate > best_score:
                    best_score = candidate
                    best_inverse = inverse.copy()
                    best_shifts = shifts.copy()
            elif changed_shift:
                shifts[slot] = old_shift
            else:
                temporary = inverse[first]
                inverse[first] = inverse[second]
                inverse[second] = temporary
            temperature *= cooling
'''
new = '''        for _ in range(iterations):
            state, move = _rng_int(state, 10)
            if move < 3 and period > 1:
                state, slot_raw = _rng_int(state, period - 1)
                slot_idx = np.int64(slot_raw + 1)
                old_shift = shifts[slot_idx]
                state, proposal_raw = _rng_int(state, a)
                proposal = np.int64(proposal_raw)
                if proposal == old_shift:
                    continue
                shifts[slot_idx] = proposal
                candidate = score_wheel(cipher, phase, inverse, shifts, trigram_logp, unigram_logp)
                delta = candidate - current_score
                accept = delta >= 0.0
                if not accept:
                    state, uniform = _rng_float(state)
                    accept = uniform < math.exp(delta / max(temperature, 1e-9))
                if accept:
                    current_score = candidate
                    if candidate > best_score:
                        best_score = candidate
                        best_inverse = inverse.copy()
                        best_shifts = shifts.copy()
                else:
                    shifts[slot_idx] = old_shift
            else:
                state, first_raw = _rng_int(state, a)
                state, second_raw = _rng_int(state, a)
                first_idx = np.int64(first_raw)
                second_idx = np.int64(second_raw)
                if first_idx == second_idx:
                    continue
                temporary = inverse[first_idx]
                inverse[first_idx] = inverse[second_idx]
                inverse[second_idx] = temporary
                candidate = score_wheel(cipher, phase, inverse, shifts, trigram_logp, unigram_logp)
                delta = candidate - current_score
                accept = delta >= 0.0
                if not accept:
                    state, uniform = _rng_float(state)
                    accept = uniform < math.exp(delta / max(temperature, 1e-9))
                if accept:
                    current_score = candidate
                    if candidate > best_score:
                        best_score = candidate
                        best_inverse = inverse.copy()
                        best_shifts = shifts.copy()
                else:
                    temporary = inverse[first_idx]
                    inverse[first_idx] = inverse[second_idx]
                    inverse[second_idx] = temporary
            temperature *= cooling
'''
if source.count(old) != 1:
    raise RuntimeError("joint branch patch site mismatch")
patched = source.replace(old, new)
print(
    "V060_P_LAUNCHER2_PATCH",
    {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "scientific_search_changed": False,
        "reason": "separate Numba-typed shift and swap branches",
    },
    flush=True,
)
exec(compile(patched, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
