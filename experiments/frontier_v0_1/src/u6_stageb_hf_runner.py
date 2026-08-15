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
BRIDGE_PREFIX = "u6-stageb-20260815-pathfix"
EXPECTED_ENCODER_SHA256 = "54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0"
EXPECTED_PAIR_SHA256 = "7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4"

ROW_OLD = "rows.append({'id':str(r['id']),'folio':str(r['folio']),'word_index':int(r['word_index'])})"
ROW_NEW = "rows.append({'id':str(r['id']),'folio':str(r['folio']),'word_index':int(r['word_index']),'path':str(r['path'])})"
DISCOVER_OLD = "required=set(W.id.astype(str)); id2path,scanned=discover_word_paths(a.data,required); audit['norm_png_scanned']=scanned; audit['required_crop_paths']=len(id2path)"
DISCOVER_NEW = """if W.id.astype(str).nunique()!=len(W): raise RuntimeError(f'manifest id uniqueness gate failed: rows={len(W)} unique_ids={W.id.astype(str).nunique()}')
    id2path={}; missing_paths=[]; unsafe_paths=[]
    for r in W.itertuples():
        rel=Path(str(r.path))
        if rel.is_absolute() or '..' in rel.parts:
            unsafe_paths.append(str(r.path)); continue
        p=a.data/rel
        if not p.is_file(): missing_paths.append(str(p)); continue
        id2path[str(r.id)]=p
    if unsafe_paths or missing_paths: raise RuntimeError(f'manifest path gate failed: unsafe={len(unsafe_paths)} missing={len(missing_paths)} first_unsafe={unsafe_paths[:5]} first_missing={missing_paths[:5]}')
    if len(id2path)!=len(W): raise RuntimeError(f'manifest path cardinality gate failed: rows={len(W)} resolved={len(id2path)}')
    audit['path_resolution']='manifest_exact'; audit['norm_png_scanned']=0; audit['required_crop_paths']=len(id2path)"""


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
        json={"secret": BRIDGE_CODE, "id": identifier, "payload": json.dumps(obj, sort_keys=True), "meta": meta or {}},
        timeout=120,
    )
    r.raise_for_status()


def apply_pathfix(stageb: Path) -> tuple[str, str]:
    original_sha = sha256(stageb)
    text = stageb.read_text(encoding="utf-8")
    if text.count(ROW_OLD) != 1:
        raise RuntimeError(f"row patch anchor count != 1: {text.count(ROW_OLD)}")
    if text.count(DISCOVER_OLD) != 1:
        raise RuntimeError(f"discover patch anchor count != 1: {text.count(DISCOVER_OLD)}")
    text = text.replace(ROW_OLD, ROW_NEW, 1).replace(DISCOVER_OLD, DISCOVER_NEW, 1)
    stageb.write_text(text, encoding="utf-8")
    return original_sha, sha256(stageb)


def main() -> int:
    status = {
        "schema": "frontier-u6-stageb-hf-runner-v0.4",
        "pathfix_freeze": "U6_STAGEB_ASSET_PATHFIX_FREEZE_v0_2.md",
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

        original_stageb_sha, patched_stageb_sha = apply_pathfix(stageb)
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
            stageb_original_sha256=original_stageb_sha,
            stageb_pathfixed_sha256=patched_stageb_sha,
            status="running_stageb",
        )
        bridge(f"{BRIDGE_PREFIX}-status", status, {"phase": "pre-calibration", "pathfix": "manifest_exact"})

        cmd = [sys.executable, str(stageb), "--data", str(root), "--encoder", str(encoder), "--pair-skeleton", str(pairs), "--out", str(out), "--dev-reps", "60", "--confirm-reps", "100"]
        p = subprocess.run(cmd, text=True, capture_output=True)
        status.update(
            returncode=p.returncode,
            stdout_tail=p.stdout[-20000:],
            stderr_tail=p.stderr[-20000:],
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
                k: {"null_fpr": v.get("null_fpr"), "physical_power_beta_0_50": v.get("physical_power_beta_0_50"), "pass": v.get("pass")}
                for k, v in result.get("components", {}).items()
            }
            bridge(f"{BRIDGE_PREFIX}-result", result, {"phase": "sealed-calibration-result", "pathfix": "manifest_exact", "patched_stageb_sha256": patched_stageb_sha})
        bridge(f"{BRIDGE_PREFIX}-status", status, {"phase": "terminal", "pathfix": "manifest_exact"})
        print(json.dumps(status, indent=2, sort_keys=True))
        return p.returncode
    except Exception as exc:
        status.update(status="wrapper_failed", error=repr(exc), traceback=traceback.format_exc()[-20000:], target_opened=False, true_retention_read=False, finished_unix=time.time())
        try:
            bridge(f"{BRIDGE_PREFIX}-status", status, {"phase": "wrapper-failure", "pathfix": "manifest_exact"})
        finally:
            print(json.dumps(status, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
