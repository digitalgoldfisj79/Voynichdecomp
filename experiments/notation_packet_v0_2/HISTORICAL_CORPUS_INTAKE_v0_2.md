# Historical notation calibration intake

No historical-corpus score is included in the v0.2 verdict. The following machine-readable sources were located for the next frozen phase:

1. **ECHOES GABCtoMEI** — Aquitanian and square-neume GABC/MEI encodings with explicit neume cuts, word spaces, graphical joining, stems, accidentals and special neume forms.
2. **Organ-Tablature-Ocr / DeepTab** — 2,400 manually annotated staves from Ammerbach 1575 and 1583, with paired duration/special and pitch/rest label sequences. This is later than Voynich but useful for pipeline calibration.
3. **E-LAUTE** — open-access machine-readable German lute tablature programme for 1450–1550; individual MEI releases are appearing, but the corpus remains under active construction.
4. **Buxheim Organ Book** — complete IIIF facsimile, c.1460–1470; a surface-sign transcription corpus must be curated or licensed before model comparison.

Before scoring, freeze one atomic-sign policy per source and preserve both surface-sign and canonical-event streams. Do not mix modern editorial inference into the surface stream.
