# Official OCR acquisition freeze before OCR inspection

Retain the exact 77-page set and six fidelity-audit pages from the prior frozen Wagenseil protocol. New source discovery: the official IIIF manifest exposes a per-canvas `seeAlso` hOCR URL. No OCR content has been inspected before this freeze.

Use BSB's existing hOCR as one fixed OCR pipeline. Download and hash raw hOCR for every frozen page; do not substitute other witnesses or OCR engines based on discrepancies. Parse all `ocrx_word` elements within line elements in DOM order, preserving Unicode, line boundaries, page boundaries, and raw XML/HTML. Keep running headers, catchwords, marginal material identifiable separately; no dictionary or spelling correction. Before any a-z reduction or L score, independently audit the frozen six pages against full scan images, including long-s/ligatures, omissions, spurious words, segmentation, and hyphenation. Systematic atom-relevant OCR defects block the uncorrected source until explicitly bounded. No page replacement or target access.

Manifest: https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb10903876/manifest
Image suffixes: 261,263,...,393 plus 395,...,404. Audit suffixes: 273,289,337,381,389,391.
