#!/usr/bin/env bash
set -euo pipefail
# Orchestration only. Scientific runner/scorer remain pinned to SOURCE_COMMIT.
SPLIT=${1:?split}; SOURCE=${2:?source}; START=${3:?start}; STOP=${4:?stop}; STRENGTH=${5:-}; PARALLEL=${6:-10}
SOURCE_COMMIT=765a0de332c2d31b0e97738885c35b1e759e5233
BASE_COMMIT=2ea089393f848329b0673cddf09833789643e779
RUNNER_SHA=d17668f7040444f9fb51ec1aeaba24ea13d4ba86fc4f88e5a4e711a3ce81c525
MANIFEST_CANON_SHA=62655854117793168d46bfb05b548d5993f28dd613301d859fd666504a3b51c0
EVENTS_SHA=74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rm -rf /workspace/source /workspace/base /workspace/t0pkg /workspace/t0b
mkdir -p /workspace/source /workspace/base
git -C /workspace/source init -q; git -C /workspace/source remote add origin https://github.com/digitalgoldfisj79/Voynichdecomp.git; git -C /workspace/source fetch -q --depth 1 origin "$SOURCE_COMMIT"; git -C /workspace/source checkout -q FETCH_HEAD
git -C /workspace/base init -q; git -C /workspace/base remote add origin https://github.com/digitalgoldfisj79/Voynichdecomp.git; git -C /workspace/base fetch -q --depth 1 origin "$BASE_COMMIT"; git -C /workspace/base checkout -q FETCH_HEAD
test "$(git -C /workspace/source rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git -C /workspace/base rev-parse HEAD)" = "$BASE_COMMIT"
test "$(sha256sum /workspace/source/instrumentation/memory_recurrence_runner_v1.py | cut -d' ' -f1)" = "$RUNNER_SHA"
python - <<'PY'
import json,hashlib
p='/workspace/source/instrumentation/manifests/memory_recurrence_source_id_v1.json'; o=json.load(open(p)); h=hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(); assert h=='62655854117793168d46bfb05b548d5993f28dd613301d859fd666504a3b51c0'; print('P1_REMOTE_MANIFEST='+h)
PY
mkdir -p /workspace/t0pkg /workspace/t0b
cat /workspace/base/experiments/t0_known_source_audit_20260907/chunks/chunk*.txt | base64 -d | tar -xz -C /workspace/t0pkg
cp /workspace/base/voynich_transcriptions_slim.json /workspace/t0pkg/
tar -xzf /workspace/base/experiments/t0b_e2e_adequacy_20260907/t0b_codebundle.tgz -C /workspace/t0b
python -m pip -q install numpy scipy scikit-learn
mkdir -p /mnt/data/surface_payload_resolution_v01_stage2 /mnt/data/surface_payload_resolution_v01_run /mnt/data/surface_payload_resolution_v01_repair /mnt/data/voynich_workingset_v01 /mnt/data/voynich_regional_v01
cp /workspace/t0pkg/stage2/* /mnt/data/surface_payload_resolution_v01_stage2/; cp /workspace/t0pkg/repair/* /mnt/data/surface_payload_resolution_v01_repair/; cp /workspace/t0pkg/working/* /mnt/data/voynich_workingset_v01/; cp /workspace/t0pkg/regional/* /mnt/data/voynich_regional_v01/; cp /workspace/t0pkg/base/* /mnt/data/surface_payload_resolution_v01_run/
python /workspace/t0pkg/reconstruct_events.py /workspace/t0pkg >/dev/null
cp /workspace/t0pkg/base/events_raw.pkl /mnt/data/surface_payload_resolution_v01_run/events_raw.pkl
python - <<'PY'
import pickle,json,hashlib
rows=pickle.load(open('/mnt/data/surface_payload_resolution_v01_run/events_raw.pkl','rb')); h=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode()).hexdigest(); assert len(rows)==34087 and h=='74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f'; print('P1_REMOTE_EVENTS='+h)
PY
export SPLIT SOURCE STRENGTH
seq "$START" $((STOP-1)) | xargs -P"$PARALLEL" -I{} bash -lc 't={}; extra=(); if [ -n "$STRENGTH" ]; then extra=(--strength "$STRENGTH"); fi; python /workspace/source/instrumentation/memory_recurrence_runner_v1.py "$SPLIT" "$SOURCE" "$t" "$((t+1))" --manifest /workspace/source/instrumentation/manifests/memory_recurrence_source_id_v1.json "${extra[@]}"'
