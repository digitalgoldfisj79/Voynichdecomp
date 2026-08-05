"""Transparent reconstruction of the ephemeral corpus.pkl used by the recovered C2ST scripts.

Source: committed enriched_records.pkl. This is a v0.6.2 reconstruction, not recovered original source.
"""
import pickle
from collections import defaultdict
from itertools import groupby

RECORDS = "/tmp/vms/repo/enriched_records.pkl"
OUTPUT = "/tmp/vms/work/corpus.pkl"
records = pickle.load(open(RECORDS, "rb"))
line_tokens = []
sec_tokens = defaultdict(list)
sec_lines = defaultdict(list)
tokens = []
errors = {"boundary": 0, "line_len": 0, "position": 0}
for _, group in groupby(records, key=lambda r: (r["folio"], r["line_no"])):
    rows = list(group)
    line = [r["token"] for r in rows]
    section = rows[0]["section"]
    if len(rows) != rows[0]["line_len"]:
        errors["line_len"] += 1
    for i, row in enumerate(rows):
        if row["pos"] != i:
            errors["position"] += 1
        if row["is_first_word"] != (i == 0) or row["is_last_word"] != (i == len(rows) - 1):
            errors["boundary"] += 1
        tokens.append(row["token"])
        sec_tokens[row["section"]].append(row["token"])
    line_tokens.append(line)
    sec_lines[section].append(line)
assert len(tokens) == len(records) == 37465
assert not any(errors.values()), errors
pickle.dump({"line_tokens": line_tokens, "tokens": tokens, "sec_tokens": dict(sec_tokens), "sec_lines": dict(sec_lines)}, open(OUTPUT, "wb"))
print({"tokens": len(tokens), "lines": len(line_tokens), "errors": errors, "output": OUTPUT})
