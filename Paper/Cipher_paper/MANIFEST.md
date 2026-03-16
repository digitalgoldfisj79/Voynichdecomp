# Paper 2 — Complete Ship Package
## 2026-03-16

### Paper
- paper2_v15_draft.md — 8,786 words, final
- paper2_v15_draft.docx — Word version

### Core Code
- v11_nomenclator.py — forward cipher (P_STICKY=0.22, nomenclator lock)
- nomenclator_optimizer.py — greedy optimizer (trains on Ald.211, validates on CI)
- reproduce_termux.py — full reproduction on phone (all numbers match paper)
- reproduce_all.py — full reproduction with all stages + stickiness sweep

### Data
- enriched_records.pkl — VMS: 37,465 tokens, 9 sections, 224 folios (ZLZI→PGCS)
- p70c_full_layer.pkl — P70-C decomposition layer
- ci_corpus_parsed.pkl — Circa Instans: 52,004 tokens (Wellcome MS 624, Transkribus)
- battery_v4.pkl — 84-metric scoring battery
- nomenclator_optimizer_result.pkl — optimized assignments (r=0.96 train, 0.89 CV)
- v11_refactored_10seed.pkl — 10-seed scoring results (62.9/84, σ=2.6)
- cross_section_scores.pkl — 9-section generalisation scores
- ha_enrichment_all_folios.pkl — CV enrichment for all 48 HA folios
- sticky_sweep_results.pkl — P_STICKY calibration sweep

### Source Texts
- ms_ald_211_htr.md — Ald.211 training corpus (2,006 words, 12 folios)
- ms_ald_211_htr_COMPLETE.md — full reading (22 folios, for future retraining)

### Documentation
- AUDIT_REPORT_v15_FINAL.md — every number in paper verified against code
- SESSION_HANDOFF_2026-03-15.json — machine-readable state
- PAPER2_RUNNING_RESULTS_gap_analysis.md — session log
- HA_folio_enrichment_analysis.md — folio-level CV enrichment results
- sample_output_250.md — 250-token cipher output sample

### Support
- test_sticky_sweep.py — P_STICKY calibration code
- v11_nomenclator_v14_backup.py — pre-fix v11 for comparison
- v11_refactor_diff.txt — diff showing nomenclator lock fix
- cipher_trace_voynich.html — three-layer Voynich glyph trace (EVA2 font)

### Key Numbers (all verified on Termux)
- n/84: 62.9 (σ=2.6), C15=12.6, BG42=33.4
- P_STICKY=0.22, COPY_ALPHA=1.3
- Nomenclator: train r=0.96, CV r=0.89, null p<0.0001
- Random nomenclator baseline: mean 62.0 (range 58-66)
- Cross-section: 8/9 above 50/84
- Boundary innovation: 2.0×
- f2r 'mi' enrichment: p<10⁻⁶ (rank 1/48)
