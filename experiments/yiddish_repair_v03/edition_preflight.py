#!/usr/bin/env python3
"""Source-only preflight for the supplied Trier editions; never loads a model.

The complete PDF glyph traces are retained privately. Candidate text is the
edited verse, not a reconstruction of the underlying Hebrew printed witness.
No claim of Penn-equivalent spelling follows from successful PDF extraction.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import fitz

SOURCES = {
    "jona": {"sha": "0d38b1cafa59fcdc1f5328cc954b679c87429852bf56df61ff16dfcfe4a7f607",
             "pages": (10, 35), "size": 9.57, "x0": 51, "y0": 85, "stanzas": 102},
    "hiob": {"sha": "fb5eaee50a7e53795856019e25b9dcdc38282b96bb0a7ce60c09fd78cbaeeb79",
             "pages": (10, 26), "size": 13.61, "x0": 65, "y0": 110, "stanzas": 90},
}


def atomic_pickle(value, path):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        stream.write(gzip.compress(pickle.dumps(value, protocol=5), mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def ascii_word(text):
    return re.sub('[^a-z]', '', text.lower())


def glyph_pages(pdf, cfg):
    pages = []
    for pageno in range(cfg['pages'][0], cfg['pages'][1] + 1):
        p = pdf[pageno - 1]
        traces = p.get_texttrace()
        glyphs = []
        for span in traces:
            for char in span['chars']:
                glyphs.append({'char': chr(char[0]), 'glyph': char[1],
                               'origin': char[2], 'bbox': char[3],
                               'size': span['size'], 'font': span['font'],
                               'seqno': span['seqno']})
        pages.append({'pdf_page': pageno, 'glyphs': glyphs,
                      'drawings': p.get_drawings()})
    return pages


def assemble(pages, cfg, gap_em):
    """Use baselines and actual glyph boxes, not PDF-inferred space characters.

    Combining/overprinted accents do not change the candidate ASCII stream.
    They and every excluded glyph survive in the private glyph checkpoint.
    """
    out = []
    for page in pages:
        bases = [g for g in page['glyphs'] if abs(g['size'] - cfg['size']) < .08
                 and g['origin'][0] >= cfg['x0'] and g['origin'][1] > cfg['y0']
                 and not g['font'].startswith('HEB')
                 and g['char'] not in '˙¯ˇ´¨']
        rows = []
        for g in sorted(bases, key=lambda g: (g['origin'][1], g['origin'][0], g['seqno'])):
            if not rows or abs(rows[-1]['baseline'] - g['origin'][1]) > .3:
                rows.append({'baseline': g['origin'][1], 'glyphs': []})
            rows[-1]['glyphs'].append(g)
        # Jona marks conjectural additions in small subscript type. These are
        # part of the edited reading, so retain and flag them. Superscript
        # footnote numbers and smaller witness-label letters are not words.
        if cfg['size'] < 10:
            for g in page['glyphs']:
                if abs(g['size'] - 5.85) < .08 and re.search('[A-Za-z]', g['char']) and g['origin'][0] >= cfg['x0'] and not g['font'].startswith('HEB'):
                    near = [r for r in rows if 2.3 < g['origin'][1] - r['baseline'] < 2.9]
                    if len(near) == 1:
                        near[0]['glyphs'].append({**g, 'editorial_subscript': True})
        for row in rows:
            gs = sorted(row['glyphs'], key=lambda g: (g['origin'][0], g['seqno']))
            text, gaps, previous = '', [], None
            for g in gs:
                if previous:
                    gap = g['bbox'][0] - previous['bbox'][2]
                    if gap > .1:
                        gaps.append({'after': len(text), 'gap_em': gap / cfg['size'],
                                     'left': previous['char'], 'right': g['char']})
                    if gap > cfg['size'] * gap_em:
                        text += ' '
                text += g['char']
                previous = g
            text = re.sub(r'\[\s*\d+\s*\]', '', text).strip()
            out.append({'pdf_page': page['pdf_page'], 'baseline': row['baseline'],
                        'text': text, 'glyphs': gs, 'gaps': gaps,
                        'editorial_italic': any('Italic' in g['font'] for g in gs),
                        'editorial_subscript': any(g.get('editorial_subscript') for g in gs)})
    return out


def verse_rows(name, rows):
    stanza = 0
    out, markers = [], []
    for row in rows:
        text = row['text']
        marker = re.fullmatch(r'\{\s*(\d+)\s*\}' if name == 'jona' else r'(\d+)', text)
        if marker:
            stanza = int(marker.group(1))
            markers.append(stanza)
            continue
        if not stanza:
            continue
        # Printed prose label and closing colophon are paratext, not verse.
        if name == 'jona' and text.replace('`', '').startswith('selik sefer'):
            break
        if name == 'hiob' and ascii_word(text) == 'omenselo':
            break
        if name == 'jona' and text.startswith('(dos is'):
            continue
        # Hiob uses right-aligned bracketed runover on the following physical
        # line; its position in the *previous* verse is explicit in the layout.
        runover = re.search(r'\s+\[([^\]]+)$', text) if name == 'hiob' else None
        if runover:
            assert out and out[-1]['stanza'] == stanza
            out[-1]['text'] += ' ' + runover.group(1)
            out[-1]['runover_provenance'] = {'pdf_page': row['pdf_page'],
                                            'baseline': row['baseline'],
                                            'text': runover.group(1)}
            row = {**row, 'text': text[:runover.start()]}
        out.append({**row, 'stanza': stanza})
    return out, markers


def run(root):
    state = {'status': 'SOURCE_PREFLIGHT_ONLY_UNQUALIFIED', 'target_loaded': False,
             'model_loaded': False, 'sources': {}}
    summary = {}
    for name, cfg in SOURCES.items():
        pdfpath = root / (name + '.pdf')
        assert hashlib.sha256(pdfpath.read_bytes()).hexdigest() == cfg['sha']
        pdf = fitz.open(pdfpath)
        pages = glyph_pages(pdf, cfg)
        variants = {}
        for threshold in [.30, .31, .32]:
            rows, markers = verse_rows(name, assemble(pages, cfg, threshold))
            assert markers == list(range(1, cfg['stanzas'] + 1)), (name, markers)
            words = [w for row in rows for part in row['text'].split() if (w := ascii_word(part))]
            variants[str(threshold)] = {'rows': rows, 'words': words}
        selected = variants['0.31']
        extra_letters = []
        for page in pages:
            main = [r for r in selected['rows'] if r['pdf_page'] == page['pdf_page']]
            if not main:
                continue
            for g in page['glyphs']:
                if g['font'].startswith('HEB') or abs(g['size'] - cfg['size']) < .08:
                    continue
                if g['origin'][0] >= cfg['x0'] and any(abs(g['origin'][1] - r['baseline']) < 4 for r in main) and re.search('[A-Za-z]', g['char']):
                    extra_letters.append({'pdf_page': page['pdf_page'], **g})
        source = {'sha256': cfg['sha'], 'glyph_pages': pages, 'variants': variants,
                  'small_letters_near_body': extra_letters, 'stanzas': markers}
        state['sources'][name] = source
        atomic_pickle(source, root / (name + '_extraction.pkl.gz'))
        (root / (name + '_verse_candidate.txt')).write_text('\n'.join(
            f"[{r['pdf_page']}:{r['stanza']}] {r['text']}" for r in selected['rows']) + '\n')
        summary[name] = {'sha256': cfg['sha'], 'verse_pdf_pages': cfg['pages'],
                         'stanza_markers_complete': True,
                         'word_counts_by_gap_em': {k: len(v['words']) for k, v in variants.items()},
                         'gap_rule_word_stream_stable': len({tuple(v['words']) for v in variants.values()}) == 1,
                         'small_letters_near_body': extra_letters,
                         'status': 'EXTRACTION_CANDIDATE_AWAITING_FIDELITY_AUDIT'}
    (root / 'edition_preflight_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    atomic_pickle(state, root / 'edition_preflight.pkl.gz')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    run(ap.parse_args().root)
