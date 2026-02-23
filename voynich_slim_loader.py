"""
voynich_slim_loader.py — Minimal loader for voynich_transcriptions_slim.json

Usage:
    from voynich_slim_loader import load_slim, get_text, get_tokens, list_transcribers

    data = load_slim('voynich_transcriptions_slim.json')
    
    # Get all ZLZI tokens
    tokens = get_tokens(data, 'ZLZI')
    print(f"ZLZI: {len(tokens)} tokens, {len(set(tokens))} types")
    
    # Get one folio's text for a transcriber
    text = get_text(data, 'ZLZI', folio='f1r')
    print(text)
    
    # Get all transcribers
    print(list_transcribers(data))
    
    # Iterate folio by folio
    for folio, lines in sorted(data['pages'].items()):
        for line_num, line in sorted(lines.items(), key=lambda x: int(x[0])):
            unit = line.get('u', '')
            text = line['t'].get('ZLZI', '')
            if text:
                print(f"{folio}.{line_num} [{unit}] {text}")
"""

import json

def load_slim(path='voynich_transcriptions_slim.json'):
    with open(path, 'r') as f:
        return json.load(f)

def list_transcribers(data):
    """Return list of (id, name) for all transcribers."""
    return [(t['id'], t['name']) for t in data['transcribers']]

def list_sources(data):
    """Return list of (id, file, format, transcribers) for all sources."""
    return [(s['id'], s['file'], s['format'], s['transcribers']) for s in data['sources']]

def get_text(data, transcriber_id, folio=None):
    """Get space-joined text for a transcriber, optionally filtered to one folio."""
    tokens = []
    for fid, lines in data['pages'].items():
        if folio and fid != folio:
            continue
        for lnum in sorted(lines.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            txt = lines[lnum]['t'].get(transcriber_id, '')
            if txt:
                tokens.append(txt)
    return ' '.join(tokens)

def get_tokens(data, transcriber_id, folio=None):
    """Get flat list of tokens for a transcriber."""
    return get_text(data, transcriber_id, folio=folio).split()

def get_lines(data, transcriber_id, folio=None):
    """Get list of (folio, line_num, unit, text) tuples."""
    results = []
    for fid, lines in sorted(data['pages'].items()):
        if folio and fid != folio:
            continue
        for lnum in sorted(lines.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            line = lines[lnum]
            txt = line['t'].get(transcriber_id, '')
            if txt:
                results.append((fid, lnum, line.get('u', ''), txt))
    return results

def get_splat(data, transcriber_id, folio=None):
    """Get splat text (with uncertainty markers) where it differs from normalized."""
    results = []
    for fid, lines in sorted(data['pages'].items()):
        if folio and fid != folio:
            continue
        for lnum in sorted(lines.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            line = lines[lnum]
            splat = line.get('s', {}).get(transcriber_id)
            if splat:
                results.append((fid, lnum, splat))
    return results

def compare_transcribers(data, tid1, tid2, folio=None):
    """Compare two transcribers line by line, showing only differences."""
    diffs = []
    for fid, lines in sorted(data['pages'].items()):
        if folio and fid != folio:
            continue
        for lnum in sorted(lines.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            t = lines[lnum]['t']
            txt1 = t.get(tid1, '')
            txt2 = t.get(tid2, '')
            if txt1 and txt2 and txt1 != txt2:
                diffs.append((fid, lnum, txt1, txt2))
    return diffs


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'voynich_transcriptions_slim.json'
    data = load_slim(path)
    
    print("=== Voynich Transcriptions (Slim) ===")
    print(f"Sources: {len(data['sources'])}")
    print(f"Transcribers: {len(data['transcribers'])}")
    print(f"Folios: {len(data['pages'])}")
    
    print("\nTranscribers:")
    for tid, name in list_transcribers(data):
        tokens = get_tokens(data, tid)
        print(f"  {tid:10s} ({name:40s}): {len(tokens):>6,} tokens, {len(set(tokens)):>5,} types")
    
    print("\n=== Key transcriptions: token counts ===")
    for tid in ['ZLZI', 'TTLI', 'PCCA', 'FFSG', 'GCGA', 'TTIA', 'RGVN']:
        tokens = get_tokens(data, tid)
        if tokens:
            print(f"  {tid:10s}: {len(tokens):>6,} tokens, {len(set(tokens)):>5,} types")
