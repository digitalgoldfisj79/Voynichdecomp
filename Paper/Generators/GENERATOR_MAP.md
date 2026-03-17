# Generator Map

Maps paper names to source files. All generators are run and scored by `reproduce_s3.py`.

## Paper 1 Names → Files

| Paper name | Score | Family | File | Key in reproduce_s3.py |
|---|---|---|---|---|
| **Gen-SP** | 59/84 | Attested inventory | `gen_scribal_p70c.py` | Gen-SP |
| **Gen-Avoid** | 67–76/84 | Transcription | `gen_transcription_avoid.py` | Gen-Avoid |
| Gen-TS | 46/84 | Two-stream | `gen_ts_v8b.py` | Gen-TS |
| Gen-00 (f57v basic) | 14/84 | Zero-corpus | `gen_f57v.py` | Gen-00 |
| Gen-0M (f57v manual) | 29/84 | Zero-corpus | `gen_scribal_manual.py` | Gen-0M |
| Gen-0W (f57v workshop) | 31/84 | Zero-corpus | `gen_scribal_workshop.py` | Gen-0W |
| Gen-SD (ductus) | 26/84 | Attested inventory | `gen_template_ductus.py` | Gen-SD |
| Gen-02 through Gen-10 | 41–55/84 | Attested inventory | `gen_template_v2.py` – `v10.py` | Gen-02 – Gen-10 |
| Gen-04T | 44/84 | Attested inventory | `gen_template_v4_tuned.py` | Gen-04T |

## Corpus-wide generators (BG22 family)

These six generators are defined as functions inside `reproduce_s3.py`, not as separate files.

| Paper name | Score | Function in reproduce_s3.py |
|---|---|---|
| Char Bigram | 40/84 | `gen_char_bigram` |
| Scribal | 33/84 | `gen_scribal` |
| P70C Slot | 37/84 | `gen_p70c` |
| Currier A/B | 36/84 | `gen_currier` |
| Section-Profiled (BG22) | 42/84 | `gen_p70c_section_profiled` |
| Combined | 43/84 | `gen_combined` |

## Key generators for Paper 2

| Paper name | File | Notes |
|---|---|---|
| Forward cipher v11 | `forward_cipher_v11_CLEAN.py` | In Paper 2 supplements, not this folder |
| Forward cipher v11 + nomenclator | `v11_nomenclator.py` | In Paper 2 supplements |

## How to run

```bash
python reproduce_s3.py          # runs all 23 generators + cross-section validation
python reproduce_s3.py --quick  # runs Gen-SP + Gen-Avoid only
```

Requires `enriched_records.pkl` and `p70c_full_layer.pkl` in parent directory.
