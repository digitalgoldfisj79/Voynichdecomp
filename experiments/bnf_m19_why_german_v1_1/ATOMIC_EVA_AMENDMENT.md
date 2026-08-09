# Atomic EVA prospective amendment — Arabic qualification span

Date: 2026-08-09

The first atomic-EVA run stopped during synthetic control generation for Arabic, before any Voynich atomic fit or score was produced. Latin, Italian, German and French controls had already passed with exact map recovery and 100% independent-fit agreement.

Failure cause: the deterministic Arabic qualification span selected by the original atomic runner did not contain a plaintext-letter repertoire capable of emitting all 19 BnF M19 numerical values. Consequently no 31-surface-form legal synthetic ciphertext could be generated, regardless of stochastic emission attempts.

Prospective repair: for Arabic only, enumerate deterministic alternative qualification spans from the same frozen v0.9 Arabic qualification pool. Select the first span whose plaintext letters have a union of possible distinct BnF values equal to all 19 values. Then apply the unchanged synthetic generator and all unchanged qualification gates. Spanish and all other controls keep their original span rule.

No threshold, M19 mapping law, language model, Voynich panel, tokenization, or scoring rule is changed. The amended run reruns the entire six-language qualification gate before any Voynich atomic scoring.