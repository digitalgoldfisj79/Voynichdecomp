# Voynich BR1 frozen CF-neighbourhood realisation — Stage 1 closeout — 2026-10-06

- key: `voynich_BR1_cf_neighbourhood_realisation_closeout_20261006`
- status: completed
- created: 2026-10-06T05:37:44.710311+00:00
- updated: 2026-10-10T16:27:25.988447+00:00

---

BR1 / BR1C / BR1P / BR1S closeout snapshot, 2026-10-06.

Primary TTLI fixed split (job 6ac483ea404719ba3765c26d):
M1 physical +0.04104 bits/event, sign-flip null SD .02902, z=1.43.
M2 renderer proxy +0.00448, null SD .00907, z=.48.
M3 SELECT analogue +0.01056, null SD .00506, z=2.10, but fold4 validation was negative at every L2.
M4 past context -0.03324, z=-1.76.
M5 bidirectional diagnostic -0.08803, z=-2.68.
M6 full prospective +0.01895, z=.62.
M3 unique contribution inside M6 +0.01070, null SD .00464, z=2.30.

Nested five-fold crossfit (job 6ac484d5fbc85ba6823b106e):
ZLZI M1 physical z=2.47; M2 renderer z=1.30; M3 SELECT z=1.70; M6 z=1.54.
TTLI M1 physical z=1.25; M2 renderer z=-.38; M3 SELECT z=.26; M6 z=-.65.
Thus fixed-split SELECT signal does not survive physical crossfit.

Hand/section audit (jobs 6ac4857afbc85ba6823b10e0, 6ac48618404719ba3765c396):
ZLZI section-only +.05528 bif mean, null SD .02344, z=2.37; hand-only z=1.77.
TTLI section-only +.03134 bif mean, null SD .01630, z=1.91; hand-only z=.12.
Unique section over hand: ZLZI +.02978, null SD .01447, z=2.06; TTLI +.03013, null SD .01215, z=2.48.
Unique hand over section unresolved: ZLZI z=1.24; TTLI z=.11.
Component heterogeneity: ZLZI cheody/opchedy section z=2.79, hand z=2.87, both z=3.25; otam/shy hand z=3.20, both z=2.66. TTLI kar/odaiin both z=2.56; other components individually mostly unresolved.

Corrected M3 with paragraph reinforcement restored (job 6ac4884a404719ba3765c472; commit 0bdd9ac3d66cde992490a5a49344690e5499ac86):
ZLZI bif mean +.00334, null SD .00248, z=1.36.
TTLI bif mean +.00036, null SD .00222, z=.16.
Therefore the metric does not resolve the MRC SELECT analogue as an exact CF-node selector. The prior TTLI fixed-split 2.10-SD result is fold-specific and is downgraded.

Interpretation:
1. No robust evidence yet that the coarse MRC/K12 upstream SELECT state chooses exact surface node within CF neighbourhoods.
2. The renderer arm in BR1 is only a proxy using candidate ED geometry, not the unavailable frozen ED-router probability model; do not treat its negative result as a definitive renderer rejection.
3. The strongest robust BR1 signal is section-conditioned exact-node preference, with section adding unique information beyond Davis hand in both ZLZI and TTLI.
4. Hand effects are heterogeneous/component-specific, not a global explanation.
5. This section signal overlaps prior whole-token selector findings that section-conditioned token frequency is important; BR1 localises that effect inside CF neighbourhoods rather than discovering section dependence de novo.
6. Stage-2 productive-transfer should test whether section-conditioned transformation features generalise to completely held-out CF components.
