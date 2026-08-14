import json, tempfile
from pathlib import Path
from src.transliteration_uncertainty import build_lattice

def test_u1_one_vote_per_family():
    slim = {
        "transcribers": [{"id": x, "name": x} for x in ["ZLZI", "TTII", "TTVE", "VDRB-1", "GCGI"]],
        "pages": {
            "f1r": {
                "1": {
                    "u": "+P0",
                    "t": {
                        "ZLZI": "abc def",
                        "TTII": "abc def",
                        "TTVE": "abc xef",
                        "VDRB-1": "abc def",
                        "GCGI": "abc def"
                    }
                }
            }
        }
    }
    cfg = {
        "schema": "voynich-frontier-programme-v0.1",
        "u1": {
            "reference": "ZLZI",
            "independent_family_representatives": {"ZL": "ZLZI", "IT": "TTII", "VT": "TTVE", "RF": "VDRB-1", "GC": "GCGI"},
            "min_family_readings_per_line": 3,
            "min_reference_line_coverage": .9
        }
    }
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "slim.json").write_text(json.dumps(slim), encoding="utf-8")
        (td / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
        out = build_lattice(td / "slim.json", td / "cfg.json", td / "out")
        assert out["formal_verdict"] == "PASS"
        row = json.loads((td / "out/LINE_LATTICE.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert len(row["readings"]) == 5
