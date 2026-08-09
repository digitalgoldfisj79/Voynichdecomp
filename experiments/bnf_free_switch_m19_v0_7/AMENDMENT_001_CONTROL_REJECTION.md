# Amendment 001 — enforce the frozen 25-form control condition

Date: 2026-08-09

The first v0.7 launch stopped during synthetic generation, before any control score and before any Voynich access. In Latin replicate 0, one of the 25 opaque surface forms did not occur in the 45,000-letter training segment. The protocol already requires all 25 forms to be train-observed; the implementation raised instead of performing deterministic rejection/resampling.

Prospective repair:

- Keep the plaintext span fixed.
- Re-draw the per-letter BnF value choices and opaque surface realization under deterministic seeds `attempt=0,1,...` until all 25 surface forms occur in the 45,000-letter training segment.
- The six duplicated numerical values remain the six most frequent numerical values in that attempt's training numerical stream, exactly as frozen.
- Maximum 1,000 attempts; failure to satisfy the condition makes the control generator fail closed.
- Record the accepted attempt number.

No language score had been generated when this amendment was frozen.