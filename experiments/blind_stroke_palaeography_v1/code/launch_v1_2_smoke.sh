#!/usr/bin/env bash
set -euo pipefail
python -m pip install --quiet --upgrade 'transformers>=4.53,<5' huggingface_hub requests pillow numpy scipy scikit-learn scikit-image opencv-python-headless pandas
mkdir -p /tmp/run
python - <<'PY'
import urllib.request
base='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/d720101e6824855a14ef5015c873a0a53b171439/experiments/blind_stroke_palaeography_v1/code/'
urllib.request.urlretrieve(base+'external_calibration_v1_2_launcher.py','/tmp/run/external_calibration_v1_2_launcher.py')
urllib.request.urlretrieve(base+'run_external_calibration_v1_2_logged.py','/tmp/run/run_external_calibration_v1_2_logged.py')
PY
cd /tmp/run
python run_external_calibration_v1_2_logged.py \
  --corpus historical_wi \
  --work /tmp/blindpal_smoke_v12b \
  --output-repo Digitalgoldfish79/blind-scribal-hands-v1 \
  --max-writers 20 \
  --pages-per-writer 3 \
  --fragments-per-page 1 \
  --max-tiles 2 \
  --workers 32 \
  --batch-size 128 \
  --permutations 3 \
  --panel-seed 20260717
