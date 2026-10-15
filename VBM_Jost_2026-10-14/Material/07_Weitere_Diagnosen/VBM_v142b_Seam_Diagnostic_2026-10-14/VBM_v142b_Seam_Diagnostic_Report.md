# VBM-v142b | Surface seam localisation and transition-capacity audit

14 October 2026. Scope: **our VBM track only**. Original author-supplied Currier B REVA reference and frozen v142b. No claims are made about the distinct PIPW proposals or decipherment.

## Retracted / bounded interpretations

- This finding does **not** establish a Voynich-like generator or a cipher. Exact reversible output remains an engineering result.
- Existing v142b surface NO-GO remains. Localising a violation does not make it easy to repair.
- The earlier 88–90% invalid-word headline was not a count of invalid code *cells*. It was a count of generated words whose **joined** output contains character trigrams absent from an author-fitted Currier B reference. This audit pinpoints the origin.
- The B reference is in-sample for construction of the cells and is a necessary surface diagnostic, **not** a held-out historical predictive test. It does not generalise to Currier A.
- V143 carry is observationally reproducible by a ciphertext-only autonomous transducer (v144): this result is not reconsidered here.

## Frozen sources and reproducibility

- Frozen manifest: `04_pair_bridge_manifest.json`; SHA-256 `9371877cf5186b9967d1391c8ee94bb952276c9d0257bed8cf35e8f007c7b8af`.
- Native author-supplied B REVA word reference: `VMS_B_REVA_reference.txt`, 21,679 words, 1,132 distinct character trigrams and 258 distinct character bigrams.
- Sources and outputs: exact frozen `train.reference.txt`, `hartlieb.reference.txt`, `train.pair.vbm`, `hartlieb.pair.vbm`.
- Independent reconstruction segments each normalised source token into initial-state consonant cells, genuine nine bridge pairs, and vowel-state consonant cells, and asserts each concatenation equals the pre-generated v142b word byte for byte. All **81,628 of 81,628** words match. This is a deterministic identity, not a null-calibrated significance test.
- `run_seam_localization.py` and `run_transition_capacity.py` are portable: run with Python 3 in this directory. They do not modify input files.

## Word-level trigram admission, relocated precisely

| Corpus | Exact words checked | Words with unseen trigram | Bad words with seam violation | Bad words with apparent in-unit violation |
|---|---:|---:|---:|---:|
| Development | 68,396 | 60,454 (88.39%) | 60,454 | 9 |
| Hartlieb | 13,232 | 11,745 (88.76%) | 11,745 | 2 |

All 60,454 development and 11,745 Hartlieb words with at least one unattested trigram have **at least one cross-segment trigram violation**. There are 137,616 development unseen trigram instances, of which 9 were counted inside a *logical source-unit codeword*. Those codewords are actually concatenations of an escape cell and a second cell; thus even these are composition seams between primitive code cells. The corresponding Hartlieb count is 2 of 24,605. The 325 unique **selected primitive** code cells each have zero unseen internal trigrams, as do the seven primitive bridge strings.

Leading locations among development invalid trigrams: bridge-to-vowel-code-cell boundaries **79,102 instances**, and cell/bridge combinations involving the initial lead cell **50,170**. These counts are deterministic within one particular segmentation and do not imply a historical segmentation.

## Necessary graph capacity under frozen mapping

Each of the five vowel states exposes 222 source units, so a state-to-state transition admits 222×222=49,284 possible adjacent-unit pairs in this **engineering table**, before restricting to sequences occurring in German. Using *both* B-attested character bigrams and trigrams for the cell + bridge + cell join:

| Transition | Legal pairs / 49,284 | Predecessor units with any permitted continuation |
|---|---:|---:|
| e → o | 6,090 (12.36%) | 35/222 |
| e → e | 16,117 (32.70%) | 188/222 |
| e → i | 28,371 (57.57%) | 147/222 |
| i → i | 31,073 (63.05%) | 161/222 |

The e→o transition is particularly constrained: **only 35/222 predecessor-unit codewords have any B-admissible successor under the currently frozen mapping**. This is an exhaustive *local* compatibility calculation for the frozen codebook, not a proof that no other encoding can work. It is not a matched-null experiment and has no SD or z-score.

## Interpretation / next VBM-specific decision

The origin of v142b's failure is now identified: individually admissible primitive cells are **not compositionally closed** under the required consonant-unit and vowel-bridge joins. They cannot be treated as a freely concatenable alphabet.

A legitimate v145 feasibility test would solve an explicit *joint* allocation problem: assign reversible, uniquely decodable codewords while enforcing legal intercell and bridge seams across source-unit transitions, **without** refitting forbidden native surface statistics or sacrificing independent decoding. Begin with a capacity bound on fixed 443 cells and nine bridges against both A and B repertoires separately. Only if at least one feasible mapping exists should an encoder be built. Compare feasibility with capacity-matched autonomous transduction; the question remains engineering feasibility, not recovery of hidden plaintext.

The current data cannot distinguish source-driven ciphering from autonomous production. No historical-mechanism admission.
