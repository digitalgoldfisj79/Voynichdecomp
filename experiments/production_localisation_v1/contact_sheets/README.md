# Contact-sheet protocol

Visual evidence is part of the audit trail, not decoration.

## Required manifest fields

Every generated sheet must have a CSV or JSON manifest with:

- sheet_id
- manuscript_id
- institution
- shelfmark
- folio/page
- source URL or IIIF canvas
- source image asset ID
- crop box `(x, y, width, height)` in source pixels
- rotation/flip/scale operations
- label text
- inclusion reason
- expected feature(s) being inspected
- SHA-256 of any locally materialised source image

## Selection rule

The set of manuscripts/pages on a sheet must be defined before viewing the assembled sheet. Negative, ambiguous and failed examples remain visible. A sheet may not be rebuilt by silently dropping inconvenient comparanda.

## Output naming

`contact_sheets/<stage>/<sheet_id>.html`

and, where useful,

`contact_sheets/<stage>/<sheet_id>.png`

The HTML version is canonical because it can preserve clickable source links and metadata.

## Copyright / restricted assets

Public-domain/open images may be redistributed when licence permits. User-supplied, paywalled or otherwise restricted images are not committed to this public repository without explicit permission. Such assets are represented in `provenance/assets.csv` by checksum, dimensions and source description so the exact input remains auditable without republishing it.

## Interpretation

Observations derived from a sheet must be recorded separately from the sheet generation step. The generator never writes similarity labels or conclusions into the image-selection manifest.
