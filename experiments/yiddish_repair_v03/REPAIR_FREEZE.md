# Single constrained repair candidate: Witten–Bell trigram recovery

Frozen after the v0.2 known-key diagnostic, before this candidate is scored.

Keep exact BUILD works, 10,000-word allocations, extraction, fit/buffer/audit lengths, key count, 0.90 atom / 0.80 word recovery criteria, ±1 family call, independence-null formula, nuisance arms and family gates. All v0.2 families are now DEVELOPMENT, not untouched transfer.

Replace the word-boundary bigram objective with word-boundary trigrams: two initial boundary symbols and one final boundary symbol. For each two-symbol context, interpolate empirical trigram counts with the inherited bigram probabilities using Witten–Bell weight N/(N+T), where N is the context count and T the number of attested continuations. Unseen contexts back off completely. No tuned interpolation constant. Equal representation and smoothing for both languages.

Diagnose true-key single-swap preference on all development families first. Then run a blind, budget-matched development recovery panel, never injecting true keys or oracle results into solver input. Retain fresh-key blindness for any subsequent qualification. A candidate failure remains failure; no post-outcome threshold editing or source exclusions.

The implementation may accelerate exact objective evaluations with compiled code; verify scores and swap deltas against independent Python enumeration. Use 8 restarts x 5000 proposals and 60 greedy passes per model per arm. Initial key is the existing frequency rank mapping; no oracle starts. This candidate is one bounded repair attempt, not an unlimited objective search. Any residual failure is reported before another candidate is considered.

Source-derived trial checkpoints stay local. Publication is limited to executable code, protocol, and aggregate findings, with no source passages, planted keys, recovered mappings or per-key source-derived traces.
