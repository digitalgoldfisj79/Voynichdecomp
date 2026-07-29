# Deniable-encryption revision

Date: 2026-07-30

Canonical source SHA-256: `07f8aaba482d00b676874e80d84081d1b9ac70eab0fbb3bdb4aebceeb791e2d9`

## Primary source

Ran Canetti, Cynthia Dwork, Moni Naor, and Rafail Ostrovsky, “Deniable Encryption,” *Advances in Cryptology—CRYPTO '97*, LNCS 1294, pp. 90–104. DOI `10.1007/BFb0052229`.

## Manuscript changes

1. Section 2.4 now uses deniable encryption as an extreme formal illustration that a convincing opening need not identify the historically intended plaintext.
2. The text states explicitly that deniable encryption is not proposed as a likely mechanism for the Voynich Manuscript.
3. New Section 10.6, “Candidate openings must be binding,” requires proposed plaintext-key pairs to be evaluated against materially different keys, representations and plaintexts.
4. Failure to privilege an opening over realistic alternatives yields `NON_IDENTIFIABLE`, even when the opening is internally consistent.
5. The former abstention subsection is renumbered 10.7.

## Technical wording boundary

The manuscript does not claim that a receiver obtains several ordinary decryptions. It follows Canetti et al. in describing alternative openings through fake randomness or key material that make the existing ciphertext or transcript appear consistent with another cleartext.

## Verification

- 71/71 scientific reconstruction checks passed.
- 24/24 release checks passed.
- 28/28 references are cited.
- PDF visual QA passed on 33 pages.
- DOCX visual QA passed on 34 rendered pages.

No empirical result, numerical value, confidence interval, frozen threshold, gate decision or Voynich classification changed.