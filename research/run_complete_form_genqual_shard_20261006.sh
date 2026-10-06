#!/usr/bin/env bash
set -euo pipefail
START="$1"; END="$2"
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/6072424401a5da2157c03e43a14b162128e079f6/research/complete_form_generation_qualification_seed_20261006.py'
python - "$BASE" <<'PY'
import sys,urllib.request,pathlib
pathlib.Path('/tmp/gq.py').write_bytes(urllib.request.urlopen(sys.argv[1],timeout=120).read())
PY
pip install -q 'numpy==2.3.5' 'scikit-learn==1.8.0'
mkdir -p /tmp/gq
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for s in $(seq "$START" "$END"); do
  python -u /tmp/gq.py --seed "$s" >"/tmp/gq/$s.log" 2>&1
  grep '^GENQUAL=' "/tmp/gq/$s.log" | sed 's/^GENQUAL=//' >"/tmp/gq/$s.json"
  echo "GENQUAL_DONE=$s"
done
python - <<'PY'
import pathlib,json,gzip,base64
rows=[json.loads(p.read_text()) for p in sorted(pathlib.Path('/tmp/gq').glob('*.json'))]
raw=json.dumps(rows,separators=(',',':')).encode()
print('GENQUAL_PAYLOAD_GZ_B64='+base64.b64encode(gzip.compress(raw,9)).decode(),flush=True)
PY
