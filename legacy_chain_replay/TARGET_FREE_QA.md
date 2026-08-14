# Target-free QA

Completed before workflow launch.

- Protocol SHA-256 frozen: `d464cdc717e55d4233e2e5700be85b14fa2bc62a7691ac024b9e9bf98949533f`.
- Runner compiles under Python 3.
- Primary metric is copied conceptually from the section-residual chain definition: conditional consecutive ED1 transition probability minus unconditional adjacent ED1 probability.
- Exact equality is excluded from ED1.
- Line-order permutation cannot alter any within-line metric and is an explicit QA.
- Within-line shuffle preserves each line's exact token multiset and destroys transition order.
- G′ parameters are copied from the archived July preregistration/result record; no chain target was used in parameter selection.
- Timm commit and ablations are copied from the faithful August 13 audit; no reconstruction is used for this arm.
- Q57b dependency search did not recover `q56_injective_anonymous_realiser.py`; protocol therefore requires fail-closed handling rather than substitution.
- No native-EVA Voynich `ed1_chain_lift` number was computed during design/QA.
