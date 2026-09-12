# Bovo 1541 OCR development v01 — RESULT

Status: **OCR_ROUTE_REJECTED__STOP_AFTER_TWO_REGISTERED_CANDIDATES**

Frozen protocol: `bovo1541_ocr_dev_v01_spec.md`, committed before OCR outcomes as `cd104668e5a292749ee72356e1d036dcb0808c4d`.
Frozen leaves: n51, n54, n73, n82, n86, n120. Acquisition run 34699457237; artifact 10300155779; artifact digest `sha256:db86b4b82227a1ba38e3d01dd03bbe2cf4fd256ab38af1a18e91c5c01784595c`.

## Candidate B — Florence-2 Large OCR

Model: `fal-ai/florence-2-large/ocr`.
All six registered pages fail the zero-full-line-omission gate. Outputs were essentially only modern stanza/page numerals, with one short Latin-looking hallucination on n73:
- n51: `29. / 30. / 31.`
- n54: `39. / 40`
- n73: `98. / Sik . pllay pik ... / 100.`
- n82: `126. / .`
- n86: `1/40.`
- n120: `24.1. / 243.`
Request IDs are retained in the Fal job history: 01a09608-806b-7e22-ab1d-76d00e7fd586, 01a09608-b4d5-73f0-8031-5d35b5c3b579, 01a09608-d4ae-7183-bb86-df1444d72b43, 01a09609-0047-7d32-a018-1ec7b670c705, 01a09609-215d-7ae2-b8db-f805c570c7f2, 01a09609-492e-75b2-ad35-02fa5a491efa.

## Candidate A — GOT OCR 2.0

Model: `fal-ai/got-ocr/v2`; six registered URLs submitted in the frozen order; request `01a09607-a455-7c31-8ce0-eac03363e145`.
The returned six outputs likewise fail the zero-full-line-omission gate catastrophically. Instead of diplomatic Hebrew-script text they consist predominantly of repeated Arabic numerals/stanza numbers (e.g. hundreds of repetitions of `31.`, `4`, `135`, `141`, or `244`) plus one malformed repeated Hebrew fragment. This is not near-threshold OCR and no atom/boundary accuracy calculation can rescue the registered gate.

## Decision

The v01 protocol stated: if neither individual engine nor the mechanical consensus clears its gate, the OCR route is rejected; no third OCR model is introduced on this source. Both registered engines omit virtually all printed text. A consensus between them is therefore impossible under the registered >=95% resolved-atom coverage requirement.

**Decision: reject this two-engine OCR pipeline. Do not add a third OCR model or tune prompts on Bovo under v01.**

This is a tooling result only. It says nothing about Yiddish, M0, or Voynich. The fresh 1599 Basel witness was not processed or scored and remains unconsumed as potential confirmation evidence.