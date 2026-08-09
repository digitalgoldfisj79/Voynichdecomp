# Amendment 003 — support-complete control spans and parallel execution

Date: 2026-08-09
Status: **prospective with respect to K=26/K=36 and all Voynich scoring; H17/C17 remain sealed.**

The first serial qualification run was cancelled during the K=22 Arabic control, after four completed K=22 controls (Latin, Italian, German, French), because a separate control-only diagnostic found a guaranteed later generator failure: the deterministically selected fresh K=26 Arabic 84,000-letter span contained no normalized `o`, and therefore could not emit BnF value 22. No K=26/K=36 qualification result and no RF H17/C17 language score had been generated.

## A. Support-aware fresh control span

Control plaintext remains exclusively the frozen UD dev+test pools. For every `(language,K)`, examine deterministic candidate 84,000-letter spans indexed `attempt=0,1,...` by SHA-256 namespace `M19STAv17::span-support::<language>::<K>::<attempt>`. Select the **first** span whose observed plaintext letters make all 19 BnF numerical values generatively possible. This is a source-validity condition only; it does not inspect generated ciphertext, key recovery or language score.

The old candidate spans are not reused selectively. All K qualification controls are rerun under this single support-aware rule and constitute the binding qualification evidence.

## B. Execution parallelism

Independent control-language fits and independent candidate-language fits may execute concurrently in separate processes. Seeds, objective, step counts, restart counts, numerical-map law and all gates are unchanged. Parallelism changes wall-clock execution only.

No threshold, representation, source, language panel, split, vocabulary, M19 law or success criterion is changed.
