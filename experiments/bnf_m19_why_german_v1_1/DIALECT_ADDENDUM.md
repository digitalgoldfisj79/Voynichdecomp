# v1.1 Dialect Addendum — numerical n-gram qualification

Frozen 2026-08-09 after the preregistered bigram/HMM dialect panel proved unable to recover ReF dialect labels reliably, but before any numerical n-gram result is observed.

The original dialect-level result remains `UNDERPOWERED`; this addendum does not relax that gate. It asks a coarser binary question: **Bavarian vs ReF non-Bavarian**, where Bavarian is defined mechanically as ReF dialect metadata containing `bairisch` within the frozen 1350–1500 window.

## Data split

Within each macro-class, assign entire ReF documents deterministically by hash to training (~60%), development (~20%) and confirmation (~20%), while ensuring all three splits are non-empty. No document may cross splits.

## Channel representation

Do not infer plaintext. Generate numerical text directly through the exact BnF M19 channel: for every normalized plaintext letter, choose one of its five BnF table values uniformly by table occurrence. Word boundaries are retained. Surface homophones are irrelevant because the fixed Voynich key deterministically identifies the numerical value.

## Instrument development

Train equal-budget Bavarian and non-Bavarian numerical n-gram models at orders 2, 3, 4 and 5. Cap each macro training plaintext at the same number of letters before stochastic channel replication. Use identical smoothing and budgets.

On development documents generate 20 deterministic independent M19 encipherments per macro-class, each up to 25,000 letters. Select the **lowest n-gram order** achieving the highest balanced accuracy. Require development accuracy >= 0.85 or stop as underpowered.

## Fresh control gate

With the selected order frozen, evaluate 20 new deterministic encipherments per macro-class from confirmation documents. Require:

- balanced accuracy >= 0.85;
- accuracy >= 0.80 separately for Bavarian and non-Bavarian;
- median true-class log-likelihood advantage > 0.

If the gate fails, do not interpret a Voynich Bavarian/non-Bavarian score.

## Voynich diagnostic

If qualified, apply the unchanged v1.0 glyph→number map to C10 ZLZI without refitting. Score the complete stream and each of the four frozen C10 buckets. Report Bavarian-minus-non-Bavarian log-likelihood per predicted numerical symbol.

This is a macro-regional phonotactic/orthographic diagnostic, not a plaintext or provenance identification. Even a positive result cannot establish Bavarian without coherent lexical/morphological evidence and finer dialect controls.
