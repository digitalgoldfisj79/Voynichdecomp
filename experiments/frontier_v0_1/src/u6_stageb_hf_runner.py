from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zlib

import requests

BRANCH = "experiment/voynich-frontier-programme-v0.1-20260814"
BASE = f"https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/"
BRIDGE_URL = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814"
BRIDGE_CODE = "frontier-u6-stageb-20260815"
EXPECTED_ENCODER_SHA256 = "54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0"
EXPECTED_PAIR_SHA256 = "7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(rel: str, dst: Path) -> None:
    r = requests.get(BASE + rel, timeout=600)
    r.raise_for_status()
    dst.write_bytes(r.content)


def bridge(identifier: str, obj: dict, meta: dict | None = None) -> None:
    if not identifier.startswith("u6-stageb-20260815-"):
        raise ValueError(identifier)
    r = requests.post(
        BRIDGE_URL,
        json={
            "secret": BRIDGE_CODE,
            "id": identifier,
            "payload": json.dumps(obj, sort_keys=True),
            "meta": meta or {},
        },
        timeout=120,
    )
    r.raise_for_status()


def main() -> int:
    status = {
        "schema": "frontier-u6-stageb-hf-runner-v0.3",
        "status": "starting",
        "target_opened": False,
        "true_retention_read": False,
        "started_unix": time.time(),
    }
    out = Path("/tmp/stageb_out")
    out.mkdir(parents=True, exist_ok=True)
    try:
        root = Path("/manifest")
        manifest = root / "results/corpus_crop_manifest.jsonl"
        if not manifest.is_file():
            raise RuntimeError("full-corpus manifest missing from authenticated /manifest dataset mount")
        status["manifest_path"] = str(manifest)
        status["manifest_exists"] = True

        stageb = Path("/tmp/u6_stageb.py")
        encoder = Path("/tmp/U6_EXTERNAL_ENCODER.pt")
        pairs_b64 = Path("/tmp/U6_STAGEB_EVENT_SKELETON.zlib.b64")
        pairs = Path("/tmp/U6_STAGEB_EVENT_SKELETON.csv")
        download("experiments/frontier_v0_1/src/u6_stageb.py", stageb)
        download("experiments/frontier_v0_1/_stageb_bridge/U6_EXTERNAL_ENCODER.pt", encoder)
        download("experiments/frontier_v0_1/_stageb_bridge/U6_STAGEB_EVENT_SKELETON.zlib.b64", pairs_b64)

        enc_hash = sha256(encoder)
        if enc_hash != EXPECTED_ENCODER_SHA256:
            raise RuntimeError(f"encoder hash gate failed: {enc_hash}")
        pair_bytes = zlib.decompress(base64.b64decode(pairs_b64.read_text(encoding="utf-8").strip()))
        pairs.write_bytes(pair_bytes)
        pair_hash = sha256(pairs)
        if pair_hash != EXPECTED_PAIR_SHA256:
            raise RuntimeError(f"pair skeleton hash gate failed: {pair_hash}")
        status.update(
            encoder_sha256=enc_hash,
            pair_skeleton_sha256=pair_hash,
            status="running_stageb",
        )
        bridge("u6-stageb-20260815-status", status, {"phase": "pre-calibration"})

        cmd = [
            sys.executable,
            str(stageb),
            "--data", str(root),
            "--encoder", str(encoder),
            "--pair-skeleton", str(pairs),
            "--out", str(out),
            "--dev-reps", "60",
            "--confirm-reps", "100",
        ]
        p = subprocess.run(cmd, text=True, capture_output=True)
        status.update(
            returncode=p.returncode,
            stdout_tail=p.stdout[-16000:],
            stderr_tail=p.stderr[-16000:],
            status="completed" if p.returncode == 0 else "failed_stageb",
            target_opened=False,
            true_retention_read=False,
            finished_unix=time.time(),
        )
        result_path = out / "U6_STAGEB_RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status["formal_verdict"] = result.get("formal_verdict")
            status["interpretation"] = result.get("interpretation")
            status["components"] = {
                k: {
                    "null_fpr": v.get("null_fpr"),
                    "physical_power_beta_0_50": v.get("physical_power_beta_0_50"),
                    "pass": v.get("pass"),
                }
                for k, v in result.get("components", {}).items()
            }
            bridge("u6-stageb-20260815-result", result, {"phase": "sealed-calibration-result"})
        bridge("u6-stageb-20260815-status", status, {"phase": "terminal"})
        print(json.dumps(status, indent=2, sort_keys=True))
        return p.returncode
    except Exception as exc:
        status.update(
            status="wrapper_failed",
            error=repr(exc),
            traceback=traceback.format_exc()[-16000:],
            target_opened=False,
            true_retention_read=False,
            finished_unix=time.time(),
        )
        try:
            bridge("u6-stageb-20260815-status", status, {"phase": "wrapper-failure"})
        finally:
            print(json.dumps(status, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
