# Eight-manuscript chronology graphic: exact dated-witness and family-matched control audit

- key: `voynich_eo777_eight_manuscript_chronology_controls_20261011_v01`
- status: completed
- created: 2026-10-11T07:25:05.842924+00:00
- updated: 2026-10-11T07:25:05.842924+00:00

---

# RETRACTED / NON-PROMOTED FINDINGS — TOP
Do not treat the eight-manuscript 0.95 graphical correlation as a qualified Voynich writing chronology. No original 48-block model was refitted: author's underlying block data, selection rules and code are unpublished in the public voynich-clusters tree checked 2026-10-11. Earlier Providence order-conditioned chronology remains retracted.
# AUDIT 2026-10-11
From user's word-transition graphic, manuscript-median ranks were approximately reconstructed for eight manuscript witnesses (48 blocks are not 48 independent dated units). Inferred Spearman rho=.952380952, consistent with author's 0.95. Exact two-sided unrestricted rank-label permutation n=8! gives p=.001140873, null mean=0, null SD=.37796447, effect=.95238 (2.52 null SD). This is conditional on visually reconstructed ranks, not primary transition-data replication.
Primary catalog copy-date intervals: Hamburg Stadtrecht 1301; Bremen Saxon World Chronicle *manuscript* 1301–1350 (graphic lists wider 1276–1350); Oldenburg Sachsenspiegel 1336; Bremen Sachsenspiegel 1342; Bremen Stadtrecht Codex III 1428; Mandeville Lüneburg witness 1451–1475; Buxtehuder Evangeliar 1451–1500; illustrated Hamburg Stadtrecht 1497. Only nine admissible chronological orderings; inferred rho bounded [.9047619,1.0], worst-case unrestricted exact two-sided p=.00456349.
Matched null, broad genres law(5), narrative(2), religion(1): 240 exact permutations; null mean rho=.0952381, SD=.387591, effect=.857143 (2.21 SD), p=.00416667. Narrower text families town law(3), Sachsenspiegel(2), prose narrative(2), Gospels(1): 24 exact permutations, mean=.150794, SD=.457504, effect=.801587 (1.75 SD), p=.0416667. THE METRIC DOES NOT RESOLVE historical time independent of text family at 0.01 gate. Only 24 configurations => low power, not absence proof.
Legal manuscript subset 5 has rho=1.0, null SD=.5, effect=1.0 (2 SD), exact two-sided p=.0166667, fails p<.01.
Other figure-reported rho, with n=8 exact p approximate due rounded Spearman values: Word+line .83 -> p=.015377; TF-IDF .67 -> p=.083085; LSA .90 -> p=.004563. Four-arm Bonferroni only word transition clears .01, not compensating for model-selection multiverse.
Sources: https://www.slm.uni-hamburg.de/en/ren/korpus/texte.html ; https://www.fdr.uni-hamburg.de/record/9195 ; https://digital.lb-oldenburg.de/ihd/content/titleinfo/192428 ; https://www.chlg.ugent.be/text-information/ ; https://github.com/eo777/voynich-clusters .
ReN v1.1 TEI ZIP 21.8 MB MD5 bc63dbf7a83d108c60c5de92cd7a72e3; attempted download blocked by university redirect, so no raw block-level LOBO attempt here. The reproducible metadata audit script and stage pickle snapshots are in conversation artefact ren_chronology/audit_bundle.zip.
# DECISION
PASS descriptive cross-witness date association conditioned on metadata ranks. FAIL qualified historical writing-time identification under narrow family-match p<.01. No independent heldout fit on eight texts. Do not promote timeline into canon. Next: obtain exact 48 source blocks and code; run genuine leave-one-whole-manuscript-out feature fitting + calibration, compare genre and family matched baselines, orthographic controls, negative controls, direction anchors.
