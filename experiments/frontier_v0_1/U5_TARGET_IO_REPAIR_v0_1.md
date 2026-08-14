# U5-C implementation repair — training-source prefetch

Date: 2026-08-14
Status: **FROZEN BEFORE TARGET OPENING**
Scientific repair budget consumed: **NO**

## Failure being repaired

The first U5-C workflow run (`31849736531`) aborted while reconstructing the already-qualified U5-B classifier. Project Gutenberg closed an HTTPS connection during the fetch of a development training text, raising `http.client.RemoteDisconnected` inside `fit_frozen_instrument`.

The traceback occurred before `U5C_FIREWALL_PASS`, before the canonical `enriched_records.pkl` SHA check, and before any target file was opened. Therefore no Voynich score was calculated and no target information is available to this repair.

## Repair

The workflow will prefetch the **same two frozen U5-B development URLs** with `curl --retry` into the exact cache filenames that `u5_target_score.py` already expects. Before target-scoring code is invoked, a preflight imports the frozen U5-B normalization functions and requires the cached sources to reproduce the exact normalized training lengths recorded by the successful locked U5-B run:

- Latin/Caesar: **123,354** normalized characters;
- Italian/Collodi: **194,755** normalized characters.

Only then is `u5_target_score.py` called. It must still independently reconstruct the exact U5-B calibration threshold `0.9997460219719421` and the 40/40-positive, 0/160-false-positive calibration result before it can hash/read the target.

## What does not change

No source work, model feature, codebook distribution, classifier parameter, random seed, calibration split, threshold, target representation, block size, null transformation or decision boundary changes.

This is an I/O/retry implementation repair after a demonstrably pre-target network abort, not the programme's bounded scientific repair. The latter was already consumed by U2 and remains unavailable.
