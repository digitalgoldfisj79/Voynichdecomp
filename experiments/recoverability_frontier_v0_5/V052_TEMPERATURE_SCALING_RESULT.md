# Recoverability frontier v0.5.2 — temperature-scaling diagnostic

Date: 2026-07-15

Verdict: **REJECT LENGTH-PROPORTIONAL TEMPERATURE SCALING**

The search temperature and reheating floor were multiplied by ciphertext length divided by 96. The objective, search budget, inventory moves and language model were unchanged.

Development jobs:

- English: `Digitalgoldfish79/6a580be785d9643ce16d58f8`, SHA `4ba1988f9bf1ecc9f3c573975ba819dad26565de302dfc24011b6c7e086e654b`;
- Hebrew: `Digitalgoldfish79/6a580bf085d9643ce16d58fa`, SHA `c1ad1a851f613399dc90d4e9128085e784bd68f5241356c37082723aa4a0c8b3`.

## English development

- overall recovery: 35.5469%;
- 96 characters: 80.0781%;
- 192 characters: 17.5781%;
- 384 characters: 8.9844%.

## Hebrew development

- overall recovery: 63.5308%;
- 96 characters: 71.2240%;
- 192 characters: 84.1797%;
- 384 characters: 35.1888%.

The hotter schedule explores more broadly but does not settle into the high-scoring true basin within the fixed search budget. The fixed absolute temperature is therefore restored.

## Revised search diagnosis

The true key's large objective advantage, combined with failure even under the exact homophone inventory, indicates a barrier requiring coordinated changes. Single-symbol swaps and reassignments cannot efficiently traverse it because intermediate partial relabellings score poorly.

The next development algorithm may add pairwise block optimisation: for two plaintext labels, jointly reassign all currently associated cipher symbols under the multiplicity caps, evaluating the complete small assignment block. This changes the proposal mechanism, not the objective or target.
