# Amadi Residuals v1 — Historical Provenance Ledger

Date: 2026-08-11
Protocol: `AMADI_RESIDUALS_PROTOCOL_V1.md`
Status: **FROZEN BEFORE AMADI-RESIDUAL VOYNICH SCORING**

## Method

This audit separates the date of the exact Amadi operation from earlier analogues. A conceptual antecedent does not upgrade an exact mechanism. Grades follow the protocol:

- H0 secure primary/manuscript attestation <=1450
- H1 secure near-primary / securely dated <=1450 witness preserving the operation
- H2 later source explicitly attributes the operation to an earlier tradition/person, but no <=1450 operational witness verified
- H3 securely attested only after 1450 in present evidence
- HX chronology unresolved

Voynich-specific literature was excluded from provenance decisions.

## External anchors checked

1. Treccani, *Storia della Scienza*, "La civiltà islamica: ... criptologia e criptoanalisi". Its synthesis of Ibn al-Durayhim (1312–1361), *Miftāḥ al-kunūz fī īḍāḥ al-marmūz*, explicitly records bounded within-word transpositions, simple substitution, and augmentation/reduction of the number of letters. URL: `https://www.treccani.it/enciclopedia/la-civilta-islamica-condizioni-materiali-e-intellettuali-criptologia-e-criptoanalisi_%28Storia-della-Scienza%29/`.
2. Kathryn A. Schwartz, "From Text to Technological Context: Medieval Arabic Cryptology's Relation to Paper, Numbers, and the Post", *Cryptologia* 38.2 (2014), DOI `10.1080/01611194.2014.885801`. This independently places Ibn al-Durayhim's surviving cryptological work in the 14th-century Mamluk/Ayyubid technical tradition and notes his dependence on ʿAlī ibn ʿAdlān (1187–1268).
3. Mohamad Mrayati, Yahya Meer Alam and Hassan al-Tayyan, *Arabic Origins of Cryptology*, Vol. 3, analytical study/edition of Ibn al-Durayhim. The analytical taxonomy includes transposition, substitution, and "augmentation or reduction of the number of letters", and records a family of regulated substitution alphabets. This is used as a specialist edition/synthesis; where web mirrors were used for discovery, the underlying edition—not the mirror—is the authority.
4. Benedek Láng, "Theory and practice of cryptography in early modern Europe", in *Real Life Cryptology* (2018). Láng describes late-medieval Latin/European practice as remaining largely monoalphabetic before the Renaissance transition.
5. Paolo Bonavoglia, "Trithemius, Bellaso, Vigenère – Origins of the Polyalphabetic Ciphers", HistoCrypt 2020, DOI `10.3384/ecp2020171007`, primary-source-oriented history of the later polyalphabetic line.
6. Paolo Bonavoglia, "Bellaso’s 1552 cipher recovered in Venice", *Cryptologia* 43.6 (2019), DOI `10.1080/01611194.2019.1596181`.
7. Augusto Buonafalce, "Bellaso's Reciprocal Ciphers", *Cryptologia* 30.1 (2006), DOI `10.1080/01611190500383581`.
8. The standard Alberti chronology was checked against specialist and catalogue material: Alberti's changing-alphabet/disk work belongs to the 1460s, outside the <=1450 primary horizon.
9. Pre-1450 homophony is independently documented in historical literature: systematic Mantuan diplomatic homophony is attested from 1401, with still earlier 14th-century European examples reported. This is relevant to the *principle* of multiple signs per plaintext value, not to Amadi's exact twelve-letter system.

## Ledger

| ID | exact / computational mechanism | earliest secure exact or near-exact witness found | <=1450 antecedent actually verified | grade for exact Amadi residual | decision |
|---|---|---|---|---|---|
| A01 | 12-letter phonetic reduction + homophonic surface (`R12H`) | Amadi late 16th c.; exact Italian 12-letter reductions in sections 024/454 | Ibn al-Durayhim securely supplies augmentation/reduction of letter count; pre-1450 homophony independently attested. No <=1450 witness found combining Amadi-like phonetic 12-letter reduction with this homophonic surface. | **H3 exact; H1 component antecedents** | Computational stress-test permitted; does **not** reopen 1400–1450 claim unless exact earlier witness appears. |
| A02 | vowels removed from internal positions and appended per word (`VC_END`) | Amadi section 013 | Medieval Arabic transposition is rich and word-local, but no verified <=1450 source in this audit performs the specific vowel/consonant class partition + vowel-to-end transform. | **H3** | Later-Renaissance stress-test. |
| A03 | word-reset positional multi-alphabet (`PWA_K`) | Amadi 445/461; related later polyalphabetic tradition | No <=1450 operational witness found. Alberti's changing-alphabet work is in the 1460s; Trithemius/Bellaso later. Ibn al-Durayhim has many substitution alphabets but the consulted sources explicitly stop short of a demonstrated polyalphabetic state-changing operation. | **H3** | Later-Renaissance stress-test. |
| A04 | multiple substitution houses / visible selector (`GHOUSE5` is target hypothesis) | Amadi 486/489–490 | Pre-1450 homophones/nomenclator-like systems do not establish an observable symbol selecting one of several document-global substitution states. | **H3 exact; target selector interpretation is not historical evidence** | Stress-test only. |
| A05 | plaintext-driven alphabet autokey | Amadi 490; closely related mature systems are 16th c. | No <=1450 operational plaintext-autokey witness verified. | **H3** | Conditional; source reconstruction required. |
| A06 | NTR/NTRC/DBAC tiny-alphabet coordinate code | Amadi 477–484 | Coordinate/device ancestry is much older (including medieval Arabic chessboard/device methods), but no <=1450 witness of Amadi's exact 3/4-letter emitted code was verified. | **H3 exact; H1 coordinate antecedent** | Exact direct mechanism faces structural surface gate. |
| A07 | walking/two-stream extraction | Amadi 369/373–376 | General transposition is ancient/medieval, but no <=1450 operational witness of the exact two-key walking/interlocked-stream algorithm was verified. | **H3** | Conditional; exact reconstruction required. |
| A08 | Glorioso five-layer composite | Amadi 374–376 | Individual primitives have older antecedents; no <=1450 witness of this composition. | **H3** | Not a v1 target family. |
| A09 | one-letter syllable mutation | Amadi 397 | No verified <=1450 executable cipher rule matching the source operation. The source does not establish reuse-conditioned mutation. | **H3 / not independently target-admitted** | No v1 target arm. |
| A10 | three-position modulo-105 numeric cipher | Amadi 498–500 | Medieval Arabic numerical substitution is securely older, but no <=1450 modulo-105/3×5×7 construction verified. | **H3 exact; H1 numerical-cipher antecedent** | Exact direct mechanism faces structural gate. |
| A11 | dual-meaning / two-text cipher | Amadi 491–492 | Older steganographic ideas exist, but no <=1450 operational witness of the exact two-reading construction was verified. | **H3** | Non-identifiable for direct Voynich testing without external constraints. |

## Important historical result

**No exact Amadi-residual mechanism admitted for target testing has been upgraded to H0/H1.**

The audit does reveal that A01 is not historically alien in its components: both reduction/augmentation of the letter stream and homophonic substitution have secure pre-1450 antecedents. That does not establish the particular twelve-letter Italian reduction, the 19→12 surface model used by `R12H`, or their composition before 1450.

Likewise A06/A10 inherit older coordinate/numerical ideas, but their exact Amadi forms remain later-only under current evidence.

Therefore the previous circa-1400–1450 closeout remains intact regardless of the computational outcome of this v1 stress-test. A positive result here would create a chronology/transmission problem unless new historical evidence independently changes the grade.

## Reopening condition

Any future upgrade must cite a concrete <=1450 manuscript, cryptogram/key, edition passage or securely dated witness preserving the *operation*, not just a similar idea. Such an upgrade requires a new provenance commit before any historical interpretation of target results.