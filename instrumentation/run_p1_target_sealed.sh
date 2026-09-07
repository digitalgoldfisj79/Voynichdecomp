#!/usr/bin/env bash
set -euo pipefail
QUAL=${1:?qualification-json}; DEVJSONL=${2:?development-jsonl}; OUT=${3:?output-json}
SOURCE_COMMIT=765a0de332c2d31b0e97738885c35b1e759e5233
BASE_COMMIT=2ea089393f848329b0673cddf09833789643e779
# CRITICAL: target permission is checked before source/base/Voynich data are cloned or reconstructed.
python /workspace/qual/instrumentation/source_id_guard.py check-target /workspace/qual/instrumentation/manifests/memory_recurrence_source_id_v1.json "$QUAL"
echo P1_TARGET_SEAL_RELEASED
mkdir -p /workspace/source /workspace/base
git -C /workspace/source init -q; git -C /workspace/source remote add origin https://github.com/digitalgoldfisj79/Voynichdecomp.git; git -C /workspace/source fetch -q --depth 1 origin "$SOURCE_COMMIT"; git -C /workspace/source checkout -q FETCH_HEAD
git -C /workspace/base init -q; git -C /workspace/base remote add origin https://github.com/digitalgoldfisj79/Voynichdecomp.git; git -C /workspace/base fetch -q --depth 1 origin "$BASE_COMMIT"; git -C /workspace/base checkout -q FETCH_HEAD
test "$(git -C /workspace/source rev-parse HEAD)" = "$SOURCE_COMMIT"; test "$(git -C /workspace/base rev-parse HEAD)" = "$BASE_COMMIT"
mkdir -p /workspace/t0pkg /workspace/t0b
cat /workspace/base/experiments/t0_known_source_audit_20260907/chunks/chunk*.txt | base64 -d | tar -xz -C /workspace/t0pkg
cp /workspace/base/voynich_transcriptions_slim.json /workspace/t0pkg/
tar -xzf /workspace/base/experiments/t0b_e2e_adequacy_20260907/t0b_codebundle.tgz -C /workspace/t0b
python -m pip -q install numpy scipy scikit-learn
mkdir -p /mnt/data/surface_payload_resolution_v01_stage2 /mnt/data/surface_payload_resolution_v01_run /mnt/data/surface_payload_resolution_v01_repair /mnt/data/voynich_workingset_v01 /mnt/data/voynich_regional_v01
cp /workspace/t0pkg/stage2/* /mnt/data/surface_payload_resolution_v01_stage2/; cp /workspace/t0pkg/repair/* /mnt/data/surface_payload_resolution_v01_repair/; cp /workspace/t0pkg/working/* /mnt/data/voynich_workingset_v01/; cp /workspace/t0pkg/regional/* /mnt/data/voynich_regional_v01/; cp /workspace/t0pkg/base/* /mnt/data/surface_payload_resolution_v01_run/
python /workspace/t0pkg/reconstruct_events.py /workspace/t0pkg >/dev/null; cp /workspace/t0pkg/base/events_raw.pkl /mnt/data/surface_payload_resolution_v01_run/events_raw.pkl
python /workspace/qual/instrumentation/memory_recurrence_target_v1.py --development-jsonl "$DEVJSONL" --output "$OUT"
