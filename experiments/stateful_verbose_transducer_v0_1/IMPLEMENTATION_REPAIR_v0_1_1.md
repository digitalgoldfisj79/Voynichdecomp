# SVT v0.1.1 — registered implementation repair

Date: 2026-08-21
Status: frozen before any binding Gate 0/1 run

The PR smoke exposed a mismatch between the frozen FSVT generator and the decoder.

The generator constructs each state alphabet as a **shared fresh base permutation plus a bounded number of state-local swaps** (`STATE_SWAP_FRACTION=0.12`). The original decoder instead fitted every state as an independent full permutation and charged an MDL penalty proportional to a full alphabet per state. On a non-binding two-trial oracle-head smoke this recovered only 30.47% plaintext and selected the correct state structure 0/2.

That is an instrument defect: the decoder discarded the factorisation defining the positive family.

v0.1.1 changes only the order-free head decoder:

1. initialise one shared inverse alphabet using the existing monoalphabetic solver on the complete head stream;
2. copy that inverse to every candidate state;
3. anneal two move types: a shared swap applied to all states, or one state-local swap;
4. charge complexity by the actual number of state-local deviations from the shared base plus the bounded schedule term, not by a full independent alphabet for every state;
5. retain the same periodic/line-reset modes, periods 2–12, language models, surface generator, segmentation lattice, hostile controls and target seal.

No Voynich data has been loaded. No binding synthetic gate has been run. The earlier smoke remains recorded and is not overwritten.

If the repaired oracle-head component still fails materially, the explicit FSVT construction is not recoverable by this solver and joint testing is blocked.