#!/usr/bin/env python3
"""
Build Supplement S12: Sforza Cancelleria Cipher Catalogue
Unifies three batches from ÖNB Cod. 2398 into a single catalogue.
"""

import pandas as pd
import numpy as np

# ── Load batch 1 (83 keys, folios 1r-79v, 1450-1468) ──
b1 = pd.read_csv('/mnt/user-data/uploads/sforza_ciphers.csv')
b1['batch'] = 1
# Normalize columns
b1 = b1.rename(columns={
    'alpha_rows': 'n_alpha_rows',
    'null_count': 'n_nulls',
    'nom_count': 'n_nomenclator',
    'has_fw_encoding': 'has_fw',
    'has_gemine': 'has_geminate',
})
# Add missing columns
if 'id' not in b1.columns:
    b1['id'] = range(1, len(b1) + 1)
if 'n_fw' not in b1.columns:
    b1['n_fw'] = np.nan
if 'has_latin_fw' not in b1.columns:
    b1['has_latin_fw'] = b1['fw_language'].apply(
        lambda x: True if pd.notna(x) and 'latin' in str(x).lower() else False
    )
if 'year' not in b1.columns and 'date' in b1.columns:
    b1['year'] = pd.to_numeric(b1['date'], errors='coerce')

# ── Load batch 2 (101 keys, folios 42v-116v, 1465-1485) ──
b2 = pd.read_csv('/mnt/user-data/uploads/sforza_ciphers_batch2.csv')
b2['batch'] = 2
if 'date_year' in b2.columns:
    b2['year'] = b2['date_year']

# ── Load batch 3 (41 keys, folios 117r-142r, 1483-1494) ──
b3 = pd.read_csv('/mnt/user-data/uploads/cod2398_batch3csv')
b3['batch'] = 3
# Extract year from date
import re
def extract_year(d):
    if pd.isna(d): return np.nan
    m = re.search(r'(\d{4})', str(d))
    return int(m.group(1)) if m else np.nan

if 'year' not in b3.columns:
    b3['year'] = b3['date'].apply(extract_year)

# Also fix batch 1 years
b1['year'] = b1['date'].apply(extract_year) if 'year' not in b1.columns or b1['year'].isna().all() else b1['year']
b2['year'] = b2['date'].apply(extract_year) if 'date_year' not in b2.columns else b2['year']

# ── Normalize all to common columns ──
COLS = ['id', 'batch', 'folio', 'date', 'year', 'correspondent',
        'n_alpha_rows', 'has_syllabary', 'has_geminate', 'has_nulls', 'n_nulls',
        'has_fw', 'n_fw', 'has_nomenclator', 'n_nomenclator', 'has_latin_fw',
        'notes']

for df in [b1, b2, b3]:
    for col in COLS:
        if col not in df.columns:
            df[col] = np.nan

b1s = b1[COLS].copy()
b2s = b2[COLS].copy()
b3s = b3[COLS].copy()

# Renumber IDs sequentially
b1s['id'] = range(1, len(b1s) + 1)
b2s['id'] = range(len(b1s) + 1, len(b1s) + len(b2s) + 1)
b3s['id'] = range(len(b1s) + len(b2s) + 1, len(b1s) + len(b2s) + len(b3s) + 1)

full = pd.concat([b1s, b2s, b3s], ignore_index=True)

# Fix boolean columns
for col in ['has_syllabary', 'has_geminate', 'has_nulls', 'has_fw', 
            'has_nomenclator', 'has_latin_fw']:
    full[col] = full[col].apply(lambda x: 
        True if str(x).strip().lower() in ('true', '1', 'yes', '✓') else False
    )

print(f"Total keys: {len(full)}")
print(f"Date range: {full['year'].min():.0f}–{full['year'].max():.0f}")
print(f"\n── Feature prevalence ──")

n = len(full)
stats = {
    'Two-table (alphabet + nomenclator)': full['has_nomenclator'].sum(),
    'Homophonic (≥2 alphabet rows)': (full['n_alpha_rows'] >= 2).sum(),
    '3+ alphabet rows': (full['n_alpha_rows'] >= 3).sum(),
    '4+ alphabet rows': (full['n_alpha_rows'] >= 4).sum(),
    '5+ alphabet rows': (full['n_alpha_rows'] >= 5).sum(),
    'Consonant-vowel syllabary': full['has_syllabary'].sum(),
    'Geminate table': full['has_geminate'].sum(),
    'Null symbols': full['has_nulls'].sum(),
    'Function word encoding': full['has_fw'].sum(),
    'Latin function words': full['has_latin_fw'].sum(),
}

for feat, count in stats.items():
    pct = count / n * 100
    print(f"  {feat:<45} {count:>4} / {n}  ({pct:>5.1f}%)")

# Nomenclator size stats
nom_sizes = full['n_nomenclator'].dropna()
print(f"\nNomenclator size: median={nom_sizes.median():.0f}, "
      f"mean={nom_sizes.mean():.1f}, range={nom_sizes.min():.0f}–{nom_sizes.max():.0f}")

# Alpha rows distribution
print(f"\nAlphabet row distribution:")
for nrows in sorted(full['n_alpha_rows'].dropna().unique()):
    count = (full['n_alpha_rows'] == nrows).sum()
    print(f"  {nrows:.0f} rows: {count} ({count/n*100:.1f}%)")

# Save unified CSV
full.to_csv('/home/claude/s12_cod2398_catalogue.csv', index=False)
print(f"\nSaved unified catalogue: {len(full)} keys")

# ── VMS comparison row ──
print(f"\n── VMS architecture vs Cod. 2398 ──")
print(f"{'Feature':<45} {'Cod.2398':>10} {'VMS':>10}")
print("-" * 67)
comparisons = [
    ('Two-table structure', f"{stats['Two-table (alphabet + nomenclator)']/n*100:.0f}%", 'Yes'),
    ('Homophonic substitution', f"{stats['Homophonic (≥2 alphabet rows)']/n*100:.0f}%", 'Yes (~4 houses)'),
    ('4+ alphabet rows/houses', f"{stats['4+ alphabet rows']/n*100:.0f}%", 'Yes (4)'),
    ('CV syllabary', f"{stats['Consonant-vowel syllabary']/n*100:.0f}%", 'Yes'),
    ('Function word encoding', f"{stats['Function word encoding']/n*100:.0f}%", 'Yes (~11 FWs)'),
    ('Latin function words', f"{stats['Latin function words']/n*100:.0f}%", 'Yes'),
    ('Nomenclator', f"{stats['Two-table (alphabet + nomenclator)']/n*100:.0f}%", f'Yes (~11 entries)'),
    ('Nomenclator size (median)', f"{nom_sizes.median():.0f}", '~11'),
    ('Null symbols', f"{stats['Null symbols']/n*100:.0f}%", 'No'),
]
for feat, cod, vms in comparisons:
    print(f"  {feat:<45} {cod:>10} {vms:>10}")
