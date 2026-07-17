#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any, BinaryIO
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

SOURCE_LAUNCHER = (
    "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
    "f376ee2a560dbbd1a0d2a3f06402cc70ec48b556/"
    "experiments/blind_stroke_palaeography_v1/code/run_saghog_v1_5_1.py"
)
SOURCE_LAUNCHER_SHA256 = "3aaad631b72c4d6154f2058de24dfa1058db40af"
# The value above is the Git blob SHA, not a byte SHA. The byte-level integrity of
# the assembled v1.5.1 source is enforced by the immutable launcher itself.
ASSEMBLED_SOURCE_SHA256 = "fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8"
ORIGINAL_JOB_ID = "6a5a1540d216bd6f3a1fb177"
OUT = Path("/tmp/saghog_v15_full/output")
ASSEMBLED_SOURCE = Path("/tmp/saghog_v1_5_1_full.py")
WORK = Path("/tmp/saghog_v16_reproduction")
RUN_LOG = WORK / "reproduction_stdout.log"
CHUNK_BYTES = 4 * 1024 * 1024

EXPECTED = {
    "selected_checkpoint_step": 500,
    "selected_by_validation": "resid_combined",
    "validation_map": {
        "raw": 0.29751631991776206,
        "resid_acquisition": 0.3754008770885403,
        "resid_ink": 0.3192494637580343,
        "resid_combined": 0.38410194015646143,
    },
    "test_map": {
        "raw": 0.32235582526308526,
        "resid_acquisition": 0.3973870413929802,
        "resid_ink": 0.32846542393624034,
        "resid_combined": 0.4107341042252194,
    },
    "nuisance_map": {
        "acquisition": 0.3053187055502651,
        "ink": 0.21899506475652689,
        "combined": 0.2954052001550606,
    },
    "criteria": {
        "absolute_over_acquisition": True,
        "ratio_over_acquisition": False,
        "permutation": True,
        "perturbation": True,
        "exact_k": False,
        "within_one_k": False,
        "all_pass": False,
    },
    "writer_split_sha256": "aa111caf6db8f1c3738ccbbc8c20b518c671e70520822379d75f41be4180d296",
}


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class StorageClient:
    def __init__(self, base_url: str, anon_key: str, bucket: str, root: str):
        self.base_url = base_url.rstrip("/")
        self.anon_key = anon_key
        self.bucket = bucket
        self.root = root.strip("/")
        if not self.root:
            raise ValueError("empty storage root")

    def _url(self, relative: str) -> str:
        object_path = f"{self.root}/{relative.lstrip('/')}"
        encoded = urllib.parse.quote(object_path, safe="/")
        return f"{self.base_url}/storage/v1/object/{self.bucket}/{encoded}"

    def _request(self, method: str, relative: str, data: bytes | None = None) -> bytes:
        headers = {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {self.anon_key}",
        }
        if data is not None:
            headers.update({
                "Content-Type": "application/octet-stream",
                "x-upsert": "false",
            })
        request = urllib.request.Request(
            self._url(relative), data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"storage {method} failed for {relative}: HTTP {exc.code}: {body}"
            ) from exc

    def upload_verified_bytes(self, relative: str, raw: bytes) -> dict[str, Any]:
        expected = hashlib.sha256(raw).hexdigest()
        self._request("POST", relative, raw)
        readback = self._request("GET", relative)
        observed = hashlib.sha256(readback).hexdigest()
        if readback != raw:
            raise RuntimeError(
                f"storage round-trip mismatch for {relative}: {expected} != {observed}"
            )
        return {
            "object": f"{self.root}/{relative}",
            "bytes": len(raw),
            "sha256": expected,
            "roundtrip_verified": True,
        }

    def upload_verified_file(self, relative: str, path: Path) -> dict[str, Any]:
        size = path.stat().st_size
        full_expected = sha256_path(path)
        if size <= CHUNK_BYTES:
            meta = self.upload_verified_bytes(relative, path.read_bytes())
            meta.update({"source_file": path.name, "chunked": False})
            return meta

        parts: list[dict[str, Any]] = []
        full_readback = hashlib.sha256()
        with path.open("rb") as handle:
            index = 0
            while True:
                raw = handle.read(CHUNK_BYTES)
                if not raw:
                    break
                part_name = f"{relative}.part{index:05d}"
                part_meta = self.upload_verified_bytes(part_name, raw)
                full_readback.update(raw)
                parts.append(part_meta)
                index += 1
        if full_readback.hexdigest() != full_expected:
            raise RuntimeError(f"internal chunk reconstruction mismatch for {path}")
        return {
            "source_file": path.name,
            "chunked": True,
            "bytes": size,
            "sha256": full_expected,
            "chunk_bytes": CHUNK_BYTES,
            "chunks": parts,
            "roundtrip_verified": True,
        }


def package_versions() -> dict[str, str]:
    packages = [
        "torch", "torchvision", "timm", "numpy", "opencv-python-headless",
        "pillow", "einops", "scipy", "scikit-learn", "scikit-image",
        "pandas", "pytorch-metric-learning",
    ]
    result = {"python": sys.version.split()[0]}
    for package in packages:
        result[package] = importlib.metadata.version(package)
    return result


def compare_result(result: dict[str, Any]) -> dict[str, Any]:
    tolerance = 0.01
    checks: dict[str, Any] = {}
    checks["checkpoint_step"] = {
        "observed": result["selected_checkpoint_step"],
        "expected": EXPECTED["selected_checkpoint_step"],
        "pass": result["selected_checkpoint_step"] == EXPECTED["selected_checkpoint_step"],
    }
    checks["representation"] = {
        "observed": result["selected_by_validation"],
        "expected": EXPECTED["selected_by_validation"],
        "pass": result["selected_by_validation"] == EXPECTED["selected_by_validation"],
    }
    for group, result_key, expected_key in [
        ("validation", "validation_metrics", "validation_map"),
        ("test", "test_metrics", "test_map"),
        ("nuisance", "nuisance_metrics", "nuisance_map"),
    ]:
        entries: dict[str, Any] = {}
        for name, expected in EXPECTED[expected_key].items():
            observed = float(result[result_key][name]["map"])
            delta = observed - float(expected)
            entries[name] = {
                "observed": observed,
                "expected": expected,
                "absolute_delta": abs(delta),
                "pass": abs(delta) <= tolerance,
            }
        checks[group] = entries
    checks["criteria"] = {
        "observed": result["criteria"],
        "expected": EXPECTED["criteria"],
        "pass": result["criteria"] == EXPECTED["criteria"],
    }
    split_hash = sha256_path(OUT / "writer_split.json")
    checks["writer_split"] = {
        "observed_sha256": split_hash,
        "expected_sha256": EXPECTED["writer_split_sha256"],
        "pass": split_hash == EXPECTED["writer_split_sha256"],
    }
    with np.load(OUT / "exact_features.npz", allow_pickle=False) as archive:
        finite = {
            name: bool(np.isfinite(archive[name]).all())
            for name in archive.files
            if np.issubdtype(archive[name].dtype, np.number)
        }
        shapes = {name: list(archive[name].shape) for name in archive.files}
    checks["features"] = {
        "finite": finite,
        "shapes": shapes,
        "pass": all(finite.values()),
    }

    def all_pass(value: Any) -> bool:
        if isinstance(value, dict):
            if "pass" in value and isinstance(value["pass"], bool):
                direct = value["pass"]
            else:
                direct = True
            return direct and all(all_pass(v) for k, v in value.items() if k != "pass")
        return True

    return {
        "absolute_map_tolerance": tolerance,
        "checks": checks,
        "accepted_reproduction": all_pass(checks),
    }


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=False)
    versions = package_versions()
    print("V16_REPRO_ENV " + json.dumps(versions, sort_keys=True), flush=True)

    launcher_path = WORK / "run_saghog_v1_5_1.py"
    launcher_bytes = urllib.request.urlopen(SOURCE_LAUNCHER, timeout=120).read()
    launcher_path.write_bytes(launcher_bytes)
    print(
        "V16_REPRO_SOURCE "
        + json.dumps(
            {
                "source_launcher": SOURCE_LAUNCHER,
                "launcher_bytes": len(launcher_bytes),
                "launcher_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
                "assembled_source_expected_sha256": ASSEMBLED_SOURCE_SHA256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with RUN_LOG.open("w", encoding="utf-8") as log_handle:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_handle)
        try:
            try:
                runpy.run_path(str(launcher_path), run_name="__main__")
            except SystemExit as exc:
                code = 0 if exc.code is None else int(exc.code)
                if code != 0:
                    raise RuntimeError(f"v1.5.1 reproduction exited with code {code}") from exc
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout

    if not ASSEMBLED_SOURCE.exists():
        raise RuntimeError("assembled v1.5.1 source was not preserved in /tmp")
    assembled_sha = sha256_path(ASSEMBLED_SOURCE)
    if assembled_sha != ASSEMBLED_SOURCE_SHA256:
        raise RuntimeError(
            f"assembled source mismatch: {assembled_sha} != {ASSEMBLED_SOURCE_SHA256}"
        )

    required = [
        OUT / "result.json",
        OUT / "writer_split.json",
        OUT / "exact_features.npz",
        OUT / "saghog_v15_best.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing reproduction outputs: {missing}")

    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    comparison = compare_result(result)
    comparison_path = WORK / "reproduction_comparison.json"
    comparison_path.write_bytes(stable_json(comparison))
    versions_path = WORK / "environment_versions.json"
    versions_path.write_bytes(stable_json(versions))

    base_url = os.environ["SUPABASE_URL"]
    anon_key = os.environ["SUPABASE_ANON_KEY"]
    storage_root = os.environ["SUPABASE_STORAGE_ROOT"]
    run_id = os.environ["V16_REPRO_RUN_ID"]
    storage = StorageClient(
        base_url=base_url,
        anon_key=anon_key,
        bucket="voynich-compute",
        root=f"{storage_root}/{run_id}",
    )

    to_persist = [
        launcher_path,
        ASSEMBLED_SOURCE,
        OUT / "result.json",
        OUT / "writer_split.json",
        OUT / "exact_features.npz",
        OUT / "saghog_v15_best.pt",
        RUN_LOG,
        comparison_path,
        versions_path,
    ]
    persisted: dict[str, Any] = {}
    for path in to_persist:
        relative = f"artifacts/{path.name}"
        print(
            "V16_REPRO_UPLOAD_START "
            + json.dumps({"file": str(path), "relative": relative}, sort_keys=True),
            flush=True,
        )
        persisted[path.name] = storage.upload_verified_file(relative, path)
        print(
            "V16_REPRO_UPLOAD_DONE "
            + json.dumps(persisted[path.name], sort_keys=True),
            flush=True,
        )

    manifest = {
        "schema": "blind-pal-saghog-v1.6-reproduction-persistence",
        "created_unix": time.time(),
        "original_job_id": ORIGINAL_JOB_ID,
        "source_launcher": SOURCE_LAUNCHER,
        "assembled_source_sha256": assembled_sha,
        "environment": versions,
        "storage_bucket": "voynich-compute",
        "storage_root": f"{storage_root}/{run_id}",
        "write_once": True,
        "comparison": comparison,
        "persisted": persisted,
        "seal": {
            "voynich_opened": False,
            "davis_labels_loaded": False,
            "f115r_loaded": False,
        },
    }
    manifest_meta = storage.upload_verified_bytes(
        "manifest/reproduction_manifest.json", stable_json(manifest)
    )
    manifest["manifest_object"] = manifest_meta
    print("V16_REPRO_PERSIST_RESULT " + json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
