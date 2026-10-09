# LANG2-CI 24-pair independent confirmatory GER2 replication (FROZEN before launch)

**Date:** 2026-10-09. **Panel size:** 24 NEW paired seeds, independent of pilot seeds 1234, 2026, 17, 73. **Purpose:** Resolve repeatability of real Latin Circa instans vocabulary structure relative to freshly scrambled vocabularies; NOT by itself a Voynich plaintext-language proof. Original four-seed GER2 qualification remains **FAILED** and will not be revised.

## Immutable science
- Latin CI source SHA `377daa2a9c2403e6b8e10146a9c67f9e6b8144e959f232ec9096cdd0a4ae81f1`; original real top-10k target vocab SHA `e5b62f9d76bbbae21218b2d5c18249b0976c30aedaed42484e1d7e029490a364`. One historical source, not an independently representative historical Latin panel.
- VMS raw SHA `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`; physical folds SHA `e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888`.
- Existing pinned, unchanged wrapper `10e8d498c745f833223b58ac2fdbfb66c7602371`, solver `e36031a9a6b8b67fcebb4d6f4af1c3753fad4287`, static GER2 cache `a4a133f0635e0408266bc185a1697821ab79bd28`, runtime `98b7324fc65aa9b14242920f6931a8d1256064df`.
- Frozen original 10×150 epochs, original optimizer/architecture and E-step, original validation-only checkpoint selection, fold4 validation with 4,103 target known vocab, folds0/1 strict-final with 10,000, fold2/3 discovery, 735 novel target types in each heldout set, physical +P0 with first two tokens on each line excluded, MCF demand221. Exact same within-token unique-anagram null generation, matched training model seeds, and unchanged evaluation. No tuning on fresh results.
- Model/scramble seeds are preselected SHA256-based 31-bit integers. **See frozen manifest below**. Do not replace selectively if a seed fares poorly; infrastructure errors may be retried under *identical* model/scramble seed and unchanged science, and log replacements.
- Execution packaging (three independently initialized replicas in one A10G-large job, stdout streamed per seed, 2h max instead of the failed earlier 30m): **engineering change only**, not scientific deviation. The GPU throughput does not change seed/arm composition.
- All 24 pairs will be analyzed exactly once as the confirmatory cohort, without interim effect peeking / sequential stopping. Scheduling in 8 packs of three per arm is merely engineering. Do not pool with the four exploratory pilot seeds for primary p-value.

## Frozen confirmatory tests
Define paired effect for seed i and endpoint k: D[i,k] = mean_edit_cost(null[i,k]) - mean_edit_cost(real[i,k]). Greater than 0 favors actual Latin. Strict-final folds0/1 is **primary**; fold4 selected-validation is predeclared **secondary reproducibility**.
1. Report 24 individual real and null costs on both endpoints; no changing the endpoint or excluded seeds. Report mean(D), median(D), 20% trimmed mean(D), sample SD of D and **sample SD of 24 null costs**, ratio mean(D)/null_sample_SD, seed-paired positive counts, and 95% percentile paired-bootstrap CI with 100,000 draws; fixed RNG seed 20261009. Report 95% percentile bootstrap CI for the mean, including negatives.
2. For each endpoint separately, test directional one-sided H0 of no mean paired advantage using **200,000 sign-flip Monte Carlo draws**, fixed RNG seed 20261009 (independent draw stream by endpoint); p=(extreme+1)/(B+1), H0 requires exchangeability under sign changes and must be stated. Use Holm familywise correction at overall alpha=0.05 for these two p-values.
3. **Promote only “reproducible Latin vocabulary-form advantage”** if BOTH endpoints have positive mean D, both Holm-adjusted one-sided p <0.05, strict-final >=18/24 paired wins, validation >=17/24 paired wins, and strict-final 20% trimmed mean D>0. If any condition fails: unresolved/failed replication; no wording that GER2 passed.
4. Report the exact binomial one-sided sign-test p for each paired win count as a robust additional diagnostic of sign consistency. No posthoc exclusions or retesting until significance. Compare directly with existing BAV/ALEM only as within-arm z and never as cross-language posterior odds.
5. All claims are limited to this one Latin CI source vocabulary and the current GER2 form-correspondence instrument. Any actual source-language identification additionally needs (i) power qualification on known encoded multilingual/monolingual synthetics, (ii) multiple independent Latin manuscript witnesses and matching genres, and (iii) robust physical/section/hand controls. Scrambled Latin is not a full historical other-language control.
6. The earlier frozen GER2 requirement z>2 using **null sample SD** on both endpoints remains unchanged; report its status separately; increasing n does not mathematically ensure z>2.

## Seed manifest (immutable; replicate index is 1–24)
```json
{
  "protocol": "LANG2_CI_24P_CONFIRM_20261009",
  "version": 1,
  "seed_derivation": "model_seed and scramble_seed each derive from 1+(uint64be(first 8 bytes of SHA256('LANG2-CI-GER2-CONFIRM-20261009-V1|pair-XX|model' or '|scramble')) mod (2^31-2))",
  "pairs": [
    {
      "pair": 1,
      "model_seed": 502261994,
      "scramble_seed": 1349709413
    },
    {
      "pair": 2,
      "model_seed": 1226010420,
      "scramble_seed": 1829303557
    },
    {
      "pair": 3,
      "model_seed": 857002309,
      "scramble_seed": 544413741
    },
    {
      "pair": 4,
      "model_seed": 1977088395,
      "scramble_seed": 1227931188
    },
    {
      "pair": 5,
      "model_seed": 1933132367,
      "scramble_seed": 658788334
    },
    {
      "pair": 6,
      "model_seed": 1863653119,
      "scramble_seed": 135037915
    },
    {
      "pair": 7,
      "model_seed": 1102886958,
      "scramble_seed": 171187306
    },
    {
      "pair": 8,
      "model_seed": 1501580245,
      "scramble_seed": 2011196939
    },
    {
      "pair": 9,
      "model_seed": 1524398923,
      "scramble_seed": 1650054274
    },
    {
      "pair": 10,
      "model_seed": 893803291,
      "scramble_seed": 1013911732
    },
    {
      "pair": 11,
      "model_seed": 1535304363,
      "scramble_seed": 111889380
    },
    {
      "pair": 12,
      "model_seed": 1255760146,
      "scramble_seed": 1678034778
    },
    {
      "pair": 13,
      "model_seed": 1005162895,
      "scramble_seed": 1932145794
    },
    {
      "pair": 14,
      "model_seed": 715032453,
      "scramble_seed": 489742399
    },
    {
      "pair": 15,
      "model_seed": 165467021,
      "scramble_seed": 1897080569
    },
    {
      "pair": 16,
      "model_seed": 1466174830,
      "scramble_seed": 2102051935
    },
    {
      "pair": 17,
      "model_seed": 979750596,
      "scramble_seed": 1346396645
    },
    {
      "pair": 18,
      "model_seed": 1865289073,
      "scramble_seed": 914128823
    },
    {
      "pair": 19,
      "model_seed": 90989445,
      "scramble_seed": 1395232844
    },
    {
      "pair": 20,
      "model_seed": 1448665307,
      "scramble_seed": 570046908
    },
    {
      "pair": 21,
      "model_seed": 283207752,
      "scramble_seed": 107427700
    },
    {
      "pair": 22,
      "model_seed": 490663375,
      "scramble_seed": 1631428037
    },
    {
      "pair": 23,
      "model_seed": 840557596,
      "scramble_seed": 1891112799
    },
    {
      "pair": 24,
      "model_seed": 235933997,
      "scramble_seed": 200089415
    }
  ]
}
```

## Infrastructure stop and audit rules
- GPU execution remains A10G-large; 3 independently trained replicas per job, no mixing of real and scrambled in the same job. Each replica streams its terminal VGER2_RESULT_JSON with its seed prefix. A job with missing terminal results is incomplete, not negative. Loss of GPU credit or timeout is infrastructure failure, never scientific exclusion.
- Compute pricing at submission from Hugging Face official page lists A10G-large at $1.50/hour. 16 packed jobs at 2h timeout yields a worst-case **$48** billed-running upper bound (normally lower due to early completion and minute billing). Failed/retried work can add charges. Do not state a guarantee on available account balance or run duration.
- Prior full four-seed results https://github.com/digitalgoldfisj79/Voynichdecomp/blob/f766600e0233671e6c891368116b0817ba052697/research/LANG2_LATIN_CI_GER2_FINAL_20261009.md
- This LANG2-CI fresh replication is isolated from the main R4 Voynich recovery ladder and separate proposed LANG2-B shared-renderer hypotheses.
