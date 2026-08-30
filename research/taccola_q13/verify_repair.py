#!/usr/bin/env python3
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

HERE = Path(__file__).parent
ORIGINAL_LOADER = HERE / "taccola_calibration.py"
REPAIRED = HERE / "taccola_calibration_v01b.py"
PANEL = HERE / "panel_v01b.json"
OUT = Path("taccola_calibration_output")
OUT.mkdir(exist_ok=True)
EXPECTED_ORIGINAL_SHA = "59d4a2635b5a64de32a1fd69c577c123fab1fe81bb1739a632ec7035dc5b4f5b"
EXPECTED_PANEL_SHA = "8226f0435ee07d8af1e4b0d8cb4a8f09af8f7a82b32bf04441f8dab7ae49c905"
CORE = ["even_sample","hog_vec","page_features","cosine","chamfer","pair_matrix","manuscript_score","z_and_p","boot_stability","build_direction"]

def decode_original():
    tree = ast.parse(ORIGINAL_LOADER.read_text(encoding="utf-8"))
    a = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PAYLOAD" for t in n.targets))
    payload = ast.literal_eval(a.value)
    if isinstance(payload, tuple): payload = payload[0]
    raw = zlib.decompress(base64.b64decode(payload))
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_ORIGINAL_SHA:
        raise RuntimeError(f"original payload checksum mismatch: {sha}")
    return raw.decode("utf-8"), sha

def fhash(src, name):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

def sha_json(obj):
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

orig, orig_sha = decode_original()
rep = REPAIRED.read_text(encoding="utf-8")
panel = json.loads(PANEL.read_text(encoding="utf-8"))
panel_sha = sha_json(panel)
if panel_sha != EXPECTED_PANEL_SHA:
    raise RuntimeError(f"panel checksum mismatch: {panel_sha}")
if panel.get("repair_provenance",{}).get("q13_sealed") is not True:
    raise RuntimeError("Q13 seal missing from repaired panel")
comparisons={}
for name in CORE:
    a=fhash(orig,name); b=fhash(rep,name)
    comparisons[name]={"original":a,"repair":b,"identical":a==b}
if not all(v["identical"] for v in comparisons.values()):
    raise RuntimeError("scientific core differs from frozen v0.1")
audit={
    "original_scientific_payload_sha256":orig_sha,
    "repaired_source_sha256":hashlib.sha256(rep.encode()).hexdigest(),
    "panel_sha256":panel_sha,
    "core_functions":comparisons,
    "all_core_functions_ast_identical":True,
    "q13_sealed":True,
    "allowed_non_scientific_changes":["transport-accessible pre-score control panel","version/provenance metadata","atomic per-manuscript pickle checkpoints"],
}
(OUT/"repair_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps({"event":"repair_audit_passed","panel_sha256":panel_sha,"core_functions":len(CORE)},sort_keys=True))
