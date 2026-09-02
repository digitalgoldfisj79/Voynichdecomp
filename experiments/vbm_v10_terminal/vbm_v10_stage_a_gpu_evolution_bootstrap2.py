# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "triton>=3.0"]
# ///
"""Filesystem bootstrap for the frozen VBM v10 Stage-A GPU evolutionary runner.
The embedded payload remains in bootstrap1; this wrapper materializes the exact
verified production source before import so Triton can inspect its source file.
"""
import ast, base64, hashlib, pathlib, re, runpy, urllib.request, zlib
URL = "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/experiment/vbm-v10-terminal-identifiability-20260901/experiments/vbm_v10_terminal/vbm_v10_stage_a_gpu_evolution_bootstrap.py"
EXPECTED = "d64bb3a63f0c17cfc9326ca45336d12ea462feb15544a7094398fb022445e95c"
txt = urllib.request.urlopen(URL, timeout=120).read().decode("utf-8")
m = re.search(r"^PAYLOAD=(.+)$", txt, re.M)
if not m:
    raise RuntimeError("embedded PAYLOAD not found")
payload = ast.literal_eval(m.group(1))
src = zlib.decompress(base64.b64decode(payload))
sha = hashlib.sha256(src).hexdigest()
if sha != EXPECTED:
    raise RuntimeError(("production source SHA mismatch", sha, EXPECTED))
p = pathlib.Path("/tmp/vbm_v10_stage_a_gpu_evolution.py")
p.write_bytes(src)
runpy.run_path(str(p), run_name="__main__")
