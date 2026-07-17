# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## What this repository is

This is a **computational research repository** analyzing the Voynich Manuscript
(Beinecke MS 408). It is not a software product — there is no application to run,
no test suite, and no build step. It is a collection of **data files, analysis
scripts, and academic paper supplements** supporting two related research claims.

The author is Edward Bozzard ([ORCID 0009-0002-4052-0994](https://orcid.org/0009-0002-4052-0994),
GitHub [@digitalgoldfisj79](https://github.com/digitalgoldfisj79)). Computational
analysis was assisted by Claude (Anthropic). All published results are meant to be
deterministically reproducible from the committed data and code.

There are two distinct research artifacts in this repo:

1. **Paper 1 — The PGCS formal grammar** (root `README.md`, `Paper/`). A lossless
   4-slot morphological decomposition of every VMS token, plus a hierarchy of
   text generators scored against the manuscript.
2. **Paper 2 — The two-table cipher architecture** (`Paper/Cipher_paper/`). A
   testable hypothesis that the manuscript is a nomenclator + syllabic-grid cipher
   over pharmaceutical Latin.

Both build on the same ground-truth data file: `enriched_records.pkl`.

## The central concept: PGCS decomposition

Every VMS word (in the EVA / Extended Voynich Alphabet transcription) is factored
into four slots:

```
word = prefix · gallows · core · suffix
```

- **Prefix** — 8 values: `∅, o, y, d, s, ch, sh, qo`
- **Gallows** — 9 values: `∅, k, t, p, f, ckh, cth, cph, cfh` (a closed functional class)
- **Core** — open set (~2,001 types); carries the most information; 52.7% of tokens have an **empty core**
- **Suffix** — 33 types grouped into 7 families: `Y, N, L, R, BARE, M, OTHER`

The decomposition is **lossless**: concatenating the four slots reconstructs the
original token exactly, for all 37,465 tokens. Read the root `README.md` and
`Paper/PGCS_OPERATIONAL_GUIDE.md` before doing any analysis — they define the
concepts precisely and pre-empt the common misreadings (e.g. why the entropy
residual being zero is a *mathematical identity*, not a finding).

## Repository layout

```
/
├── README.md                       Paper 1 — the PGCS grammar (start here)
├── README-old.md                   Superseded earlier version; do not update
├── enriched_records.pkl / .json    GROUND TRUTH: 37,465 decomposed tokens
├── p70_rules_canonical.json        210 character-adjacency segmentation rules
├── voynich_section_map.json        Folio → section mapping (9 sections)
├── voynich_transcriptions_slim.json  Multi-transcriber corpus (ZLZI, TTLI, ...)
├── voynich_slim_loader.py          Loader helpers for the slim corpus
├── p70_grammar_validation.py       Paper 1 validation: reproduces all metrics + 19 alternatives
├── p70_completion.py               Historical rule-derivation script (P69→P70)
├── p70_rules_canonical.json        Character grammar rules
├── build_daiin_pkl.py + daiin_*    daiin.net (.vml) transliteration builder + output
├── pgcs_architecture.jsx           React visualization of the PGCS slot/MI graph
├── VMS_formal_grammar.pdf          2-page formal spec
│
├── Paper/                          Paper 1 supplements (S1–S9), figures, generators
│   ├── PGCS_OPERATIONAL_GUIDE.md   The layer hierarchy (data → rules → constraints)
│   ├── reproduce_all.py            Regenerates the generator-hierarchy results
│   ├── p70c_full_layer.pkl         Slot+position constraint layer (needs p70c_full.py to unpickle)
│   ├── Generators/                 23 text generators (Gen-SP, Gen-Avoid, ...); see GENERATOR_MAP.md
│   └── S*.md                       Supplement documents
│
├── Paper/Cipher_paper/             Paper 2 — the cipher hypothesis
│   ├── README.md                   Paper 2 overview + reproduction snippets
│   ├── S1..S13                     Supplements, scoring batteries, language tests
│   ├── score_85_metrics.py         84-metric distributional scoring battery
│   ├── row_r_test.py               Source-language discriminator
│   └── *.pkl                       Parsed corpora (Circa Instans Latin, Greek, MHG, ...)
│
├── experiments/                    Exploratory work (e.g. SAGHOG palaeography preflight)
└── review_tmp/                     TEMPORARY: chunked external-review input; not part of the science
```

## Ground-truth data and conventions

`enriched_records.pkl` (and the equivalent self-documenting `enriched_records.json`)
is **the authority**. Every other layer derives from it. If anything contradicts the
pkl, the pkl wins. Each record is a dict:

```python
{
  'token': 'qokeedy', 'prefix': 'qo', 'gallows': 'k', 'core': 'eed', 'suffix': 'y',
  'sfx_fam': 'Y', 'm_core': 'ee', 'empty_core': False,
  'section': 'Herbal-A', 'folio': 'f1r', 'line_no': 1, 'pos': 3, ...
}
```

Load it directly:

```python
import pickle
with open('enriched_records.pkl', 'rb') as f:
    records = pickle.load(f)
```

**Conventions that must be followed** (they are load-bearing across the papers):

- Use `∅` (U+2205 EMPTY SET), **never** `None`, `NULL`, or `""`, to denote an empty
  PGCS slot in analysis output and prose.
- The **9 canonical sections** always use full names: Herbal-A, Herbal-B, Astronomical,
  Cosmological, Zodiac, Rosettes, Balneological, Pharmaceutical, Stars. (Note: some
  Paper 2 text uses "Biological" for what Paper 1 calls "Balneological" — prefer the
  section list in `voynich_section_map.json`, which includes an `old_to_new` mapping.)
- **EC/FC** = empty-core / full-core. The `empty_core` boolean classifies each token.
  Corpus split is ~52.7% EC / ~47.3% FC.
- For cipher-grid analysis use the `m_core` (minimal/suffix-stripped core) field; use
  `core` for raw positional analysis.
- The transliteration source is **ZLZI** (Zandbergen–Landini, ZL file v2b, IVTFF 1.7),
  SHA256 `c7ffff9e1f3ecbec174e234c04f056b2bec14f8d722726c456f108e2c7060db5`, from
  voynich.nu. Do not silently substitute a different transcription.

## Running the code — IMPORTANT caveats

**Scripts contain hardcoded absolute paths from the original authoring environment**
(`/home/claude/...`, `/mnt/user-data/...`) and some reference `enriched_records.pkl.txt`
rather than `enriched_records.pkl`. Examples:

- `p70_grammar_validation.py:31` → `/home/claude/Voynichdecomp/enriched_records.pkl.txt`
- `p70_completion.py` → several `/home/claude/` and `/mnt/user-data/` paths
- `Paper/corpus_forensics_final.py` → `/home/claude/...` and `/mnt/user-data/outputs/...`

Before running any script here, **check and fix its file paths** to point at the
actual repo location (e.g. `enriched_records.pkl` in the repo root). Do not assume a
script runs as-is. `p70_completion.py` in particular depends on upload files that are
not in the repo and is preserved as a **historical derivation record**, not a
re-runnable script.

Dependencies are minimal — Python 3 with `numpy` and `scipy` (Paper 1 validation);
`Paper/requirements.txt` lists only `numpy`. There is no virtualenv, lockfile, or
package manifest. Install ad hoc: `pip install numpy scipy`.

Typical reproduction commands (after fixing paths):

```bash
pip install numpy scipy
python p70_grammar_validation.py          # Paper 1: metrics + 19 alternative decompositions (<60s)
cd Paper && python reproduce_all.py       # Generator hierarchy (auto-downloads data from GitHub)
cd Paper/Generators && python reproduce_s3.py --quick   # Gen-SP + Gen-Avoid only
```

`Paper/p70c_full_layer.pkl` is a pickled `P70C_Full` instance and requires the class
definition (`p70c_full.py` / `Paper/p70c_full.py`) to be importable when unpickling.

## Available MCP tools for VMS work

This environment exposes purpose-built MCP servers — prefer them over ad-hoc parsing
when they fit:

- **`VoynichStats`** — read-only corpus statistics over multiple transliterations
  (folio text, KWIC, token frequency, entropy hierarchy, Zipf fit, word grammar / PGCS
  decomposition, n-gram progression). Start with `list_transliterations` and
  `list_sections`.
- **`Voynich`** — a research knowledge base (sources, claims, citations, convergence
  queries) for the scholarly literature.

There is also a **`vms-research-workbench`** skill for multi-agent VMS analysis
(deep analysis, hypothesis testing, research scouting) and a **`failure-mode`** skill
for adversarial self-review — the latter is highly relevant here (see below).

## Working style and epistemic guardrails

This is a decipherment-adjacent research repo, a field with a long history of
overclaiming. The papers themselves are careful to separate what is demonstrated from
what is suggested. Match that standard:

- **Distinguish "consistent with" from "demonstrated by."** Do not upgrade correlational
  or distributional evidence into proof. The README's own framing ("What this grammar
  does not do", "Known limitations") is the model to follow.
- **No fabricated numbers or citations.** Every statistic should trace to the data files
  or an existing supplement. If you compute something new, show the code and the file it
  read. The `failure-mode` skill exists specifically to catch false precision and
  AI-generated detail presented as fact — invoke it when reviewing claims.
- **The grammar does not decipher the manuscript.** The core slot remains opaque. Do not
  imply readings, translations, or a recovered key unless a supplement actually
  establishes one (none does).
- When updating prose in the READMEs or supplements, preserve the existing hedged,
  reviewer-anticipating tone.

## Git workflow

- Active development branch for this work: **`claude/claude-md-docs-x689yh`**. Develop,
  commit, and push there. Do not push to `main` without explicit permission.
- Push with `git push -u origin claude/claude-md-docs-x689yh`; retry network failures
  with exponential backoff.
- Do **not** open a pull request unless explicitly asked.
- Commit messages should be clear and descriptive. Do not include model identifiers in
  commits, PR text, or any committed artifact.
- `review_tmp/` holds chunked temporary review input (`glm_review_prompt.raw.part*`) and
  `README-old.md` is superseded — leave both alone unless asked; they are not part of the
  living codebase.

## Quick orientation checklist for a new task

1. Read root `README.md` (Paper 1) and/or `Paper/Cipher_paper/README.md` (Paper 2) for
   the relevant claim.
2. Read `Paper/PGCS_OPERATIONAL_GUIDE.md` to understand the data → rules → constraint
   layer hierarchy.
3. Load `enriched_records.pkl` as ground truth; respect the `∅` / section-name / EC-FC
   conventions.
4. If running a committed script, fix its hardcoded paths first.
5. Keep claims proportionate to evidence; use `failure-mode` for self-review of any new
   result.
