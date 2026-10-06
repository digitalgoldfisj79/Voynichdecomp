#!/usr/bin/env bash
set -euo pipefail
START="$1"
END="$2"
EXPECT=$((END-START+1))
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5f4e0ef14ef78e0fde3039c298f69d1fdc0694f0/research/'
python - "$BASE" <<'PY'
import sys,urllib.request,base64,gzip,pathlib
base=sys.argv[1]
for src,out in [
 ('complete_form_hf_qualification_20261006.py','/tmp/qual_runner.py'),
 ('complete_form_harness_v4_20261006.py.gz.b64','/tmp/harness.b64'),
 ('data/complete_form_hf_meta_20261006.json.gz.b64','/tmp/meta.b64')]:
    pathlib.Path(out).write_bytes(urllib.request.urlopen(base+src,timeout=120).read())
pathlib.Path('/tmp/complete_form_harness_v4.py').write_bytes(
    gzip.decompress(base64.b64decode(pathlib.Path('/tmp/harness.b64').read_text().strip())))
PY
pip install -q 'numpy==2.3.5' 'scikit-learn==1.8.0'
mkdir -p /tmp/qual
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python - <<'PY'
import urllib.request,hashlib,pathlib
u='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json'
p=pathlib.Path('/tmp/voynich_slim.json'); b=urllib.request.urlopen(u,timeout=120).read(); p.write_bytes(b)
assert hashlib.sha256(b).hexdigest()=='26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f'
PY
seq "$START" "$END" | xargs -P8 -I{} bash -lc 's={}; python -u /tmp/qual_runner.py seed --seed "$s" --harness /tmp/complete_form_harness_v4.py --meta /tmp/meta.b64 --corpus /tmp/voynich_slim.json --outdir /tmp/qual >"/tmp/qual/log_$s.txt" 2>&1 && echo "QUAL_DONE=$s" || { echo "QUAL_FAIL=$s"; cat "/tmp/qual/log_$s.txt"; exit 1; }'
python -u /tmp/qual_runner.py summarize --outdir /tmp/qual --expect "$EXPECT"
python - <<'PY'
import pathlib,json,gzip,base64
rows=[json.loads(p.read_text()) for p in sorted(pathlib.Path('/tmp/qual').glob('qual_*.json'))]
raw=json.dumps(rows,separators=(',',':')).encode()
print('QUAL_PAYLOAD_GZ_B64='+base64.b64encode(gzip.compress(raw,9)).decode(),flush=True)
PY
