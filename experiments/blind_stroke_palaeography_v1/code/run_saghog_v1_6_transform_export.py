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
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.decomposition import PCA

WORK = Path("/tmp/saghog_v16_transform_export")
OUT = WORK / "output"
ARCHIVES = WORK / "archives"
CHUNK_BYTES = 4 * 1024 * 1024
EXPECTED_ASSEMBLED_SOURCE_SHA256 = "fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8"
EXPECTED_HELPER_SHA256 = "55a6aac2f6fa831e6624c57b57ade5d49d8994ebc8f420b4f267a56c68dabeeb"
EXPECTED_UPSTREAM_COMMIT = "123cf0f306f105a46edbe8def06f49b54e64832e"
EXPECTED_ARCHIVE_MD5 = "e5ba2c7049bfb1453946233f681e4d53"
MAP_TOLERANCE = 0.001
ARRAY_TOLERANCE = 1e-5


def stable_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5_path(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    names = [
        "torch", "torchvision", "timm", "numpy", "opencv-python-headless",
        "pillow", "einops", "scipy", "scikit-learn", "scikit-image",
        "pandas", "pytorch-metric-learning",
    ]
    result = {"python": sys.version.split()[0]}
    for name in names:
        result[name] = importlib.metadata.version(name)
    return result


class Storage:
    def __init__(self, base_url: str, anon_key: str, bucket: str):
        self.base_url = base_url.rstrip("/")
        self.anon_key = anon_key
        self.bucket = bucket

    def _url(self, object_path: str) -> str:
        encoded = urllib.parse.quote(object_path.lstrip("/"), safe="/")
        return f"{self.base_url}/storage/v1/object/{self.bucket}/{encoded}"

    def request(self, method: str, object_path: str, data: bytes | None = None) -> bytes:
        headers = {"apikey": self.anon_key, "Authorization": f"Bearer {self.anon_key}"}
        if data is not None:
            headers.update({"Content-Type": "application/octet-stream", "x-upsert": "false"})
        req = urllib.request.Request(self._url(object_path), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"storage {method} {object_path}: HTTP {exc.code}: {body}") from exc

    def get(self, object_path: str, expected_sha256: str | None = None) -> bytes:
        raw = self.request("GET", object_path)
        if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
            raise RuntimeError(f"download SHA mismatch for {object_path}")
        return raw

    def put_verified(self, object_path: str, raw: bytes) -> dict[str, Any]:
        expected = sha256_bytes(raw)
        self.request("POST", object_path, raw)
        readback = self.get(object_path)
        if readback != raw:
            raise RuntimeError(f"round-trip mismatch for {object_path}")
        return {"object": object_path, "bytes": len(raw), "sha256": expected, "roundtrip_verified": True}

    def put_file_verified(self, object_path: str, path: Path) -> dict[str, Any]:
        size = path.stat().st_size
        expected = sha256_path(path)
        if size <= CHUNK_BYTES:
            meta = self.put_verified(object_path, path.read_bytes())
            return {**meta, "source_file": path.name, "chunked": False}
        parts = []
        reconstructed = hashlib.sha256()
        with path.open("rb") as handle:
            index = 0
            while True:
                raw = handle.read(CHUNK_BYTES)
                if not raw:
                    break
                part_path = f"{object_path}.part{index:05d}"
                parts.append(self.put_verified(part_path, raw))
                reconstructed.update(raw)
                index += 1
        if reconstructed.hexdigest() != expected:
            raise RuntimeError(f"chunk reconstruction mismatch for {path}")
        return {
            "source_file": path.name, "chunked": True, "bytes": size,
            "sha256": expected, "chunk_bytes": CHUNK_BYTES, "chunks": parts,
            "roundtrip_verified": True,
        }


def materialize_persisted_file(storage: Storage, meta: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    with destination.open("wb") as out:
        if meta["chunked"]:
            entries = meta["chunks"]
        else:
            entries = [meta]
        for entry in entries:
            raw = storage.get(entry["object"], entry["sha256"])
            out.write(raw)
            h.update(raw)
            total += len(raw)
    if total != int(meta["bytes"]) or h.hexdigest() != meta["sha256"]:
        raise RuntimeError(f"persisted-file reconstruction failed for {destination.name}")


def manual_pca_transform(x: np.ndarray, pca: PCA) -> np.ndarray:
    projected = (np.asarray(x, dtype=np.float64) - pca.mean_) @ pca.components_.T
    if pca.whiten:
        projected = projected / np.sqrt(pca.explained_variance_)
    return projected.astype(np.float32)


def residual_to_dict(model: Any, alpha: float = 10.0) -> dict[str, np.ndarray]:
    return {
        "n_mean": np.asarray(model.n_mean),
        "n_std": np.asarray(model.n_std),
        "x_mean": np.asarray(model.x_mean),
        "x_std": np.asarray(model.x_std),
        "beta": np.asarray(model.beta),
        "alpha": np.asarray([alpha], dtype=np.float32),
    }


def manual_residual_apply(x: np.ndarray, n: np.ndarray, values: dict[str, np.ndarray]) -> np.ndarray:
    nn = (n - values["n_mean"]) / values["n_std"]
    xx = (x - values["x_mean"]) / values["x_std"]
    nn = np.column_stack([np.ones(len(nn)), nn])
    return (xx - nn @ values["beta"]).astype(np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True)
    ARCHIVES.mkdir(parents=True)
    events: list[dict[str, Any]] = []

    def event(name: str, **payload: Any) -> None:
        record = {"event": name, "unix": time.time(), **payload}
        events.append(record)
        print("V16_EXPORT_EVENT " + json.dumps(record, sort_keys=True), flush=True)

    storage = Storage(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"], "voynich-compute"
    )
    source_root = f"{os.environ['SUPABASE_STORAGE_ROOT'].strip('/')}/{os.environ['V16_REPRO_RUN_ID'].strip('/')}"
    target_root = f"{os.environ['SUPABASE_STORAGE_ROOT'].strip('/')}/{os.environ['V16_EXPORT_RUN_ID'].strip('/')}"
    reproduction_manifest_raw = storage.get(f"{source_root}/manifest/reproduction_manifest.json")
    reproduction_manifest = json.loads(reproduction_manifest_raw)
    if not reproduction_manifest["comparison"]["accepted_reproduction"]:
        raise RuntimeError("reproduction was not accepted; export prohibited")
    event("reproduction_manifest_accepted", source_root=source_root)

    persisted = reproduction_manifest["persisted"]
    local_files = {
        "saghog_v15_best.pt": WORK / "saghog_v15_best.pt",
        "writer_split.json": WORK / "writer_split.json",
        "result.json": WORK / "reproduction_result.json",
        "saghog_v1_5_1_full.py": WORK / "saghog_v1_5_1_full.py",
    }
    for name, destination in local_files.items():
        materialize_persisted_file(storage, persisted[name], destination)
        event("source_artifact_materialized", file=name, bytes=destination.stat().st_size, sha256=sha256_path(destination))

    if sha256_path(local_files["saghog_v1_5_1_full.py"]) != EXPECTED_ASSEMBLED_SOURCE_SHA256:
        raise RuntimeError("assembled source SHA mismatch")
    module = runpy.run_path(str(local_files["saghog_v1_5_1_full.py"]), run_name="v15_export_module")
    helper = module["load_helper"]()
    if helper.UPSTREAM_COMMIT != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("upstream commit mismatch")

    split_manifest_raw = local_files["writer_split.json"].read_bytes()
    split_manifest = json.loads(split_manifest_raw)
    splits = split_manifest["splits"]
    if split_manifest["helper_sha256"] != EXPECTED_HELPER_SHA256:
        raise RuntimeError("helper SHA mismatch")

    archive = ARCHIVES / "historical_wi_color.zip"
    helper.download(module["ARCHIVE_URL"], archive, EXPECTED_ARCHIVE_MD5)
    if md5_path(archive) != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("Historical-WI archive MD5 mismatch")
    pages = helper.parse_pages(helper.extract_archive(archive))
    event("historical_wi_ready", writers=len(pages), archive_bytes=archive.stat().st_size)

    patch_x: list[np.ndarray] = []
    patch_t: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for part in ["train", "val", "test"]:
        for writer in splits[part]:
            for page_id, path in pages[writer][:3]:
                px, pt, _ = helper.page_patches(path, module["PATCHES_PER_PAGE"])
                start = len(patch_x)
                patch_x.extend(px)
                patch_t.extend(pt)
                rgb = helper.read_rgb(path)
                acquisition, ink = helper.nuisance(rgb)
                records.append({
                    "part": part, "writer": writer, "page": str(page_id),
                    "source_name": path.name, "source_sha256": sha256_path(path),
                    "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                    "start": start, "end": len(patch_x),
                    "acq": acquisition, "ink": ink,
                })
    patch_xa = np.stack(patch_x)
    patch_ta = np.stack(patch_t)
    event("patches_ready", patches=len(patch_xa), pages=len(records), bytes=int(patch_xa.nbytes))

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    MaskedAutoencoderViT, Wrapper = helper.load_upstream()
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=512, hog_pool=4, hog_bins=9,
        depth=8, decoder_depth=1, in_chans=3, global_pool=False,
        norm_pix_loss=False, target_in_chans=1,
    ).to(device)
    wrapper_args = {
        "model_options": {"in_dim": -1},
        "netvlad": {"num_clusters": 100, "random": True},
        "netvlad_pooling": False,
    }
    wrapped = Wrapper(model, wrapper_args).to(device)
    checkpoint = torch.load(local_files["saghog_v15_best.pt"], map_location="cpu")
    wrapped.load_state_dict(checkpoint["model_state_dict"])
    wrapped.to(device).eval()
    checkpoint_sha = sha256_path(local_files["saghog_v15_best.pt"])
    if checkpoint_sha != persisted["saghog_v15_best.pt"]["sha256"]:
        raise RuntimeError("checkpoint SHA differs from accepted reproduction")
    event("checkpoint_loaded", step=int(checkpoint["step"]), checkpoint_sha256=checkpoint_sha)

    masks = {
        part: np.array([record["part"] == part for record in records])
        for part in ["train", "val", "test"]
    }
    raw_vectors: dict[str, np.ndarray] = {}
    writers: dict[str, np.ndarray] = {}
    page_ids: dict[str, np.ndarray] = {}
    source_names: dict[str, np.ndarray] = {}
    for part in ["train", "val", "test"]:
        vector, writer_list = module["page_vectors"](
            wrapped, patch_xa, records, masks[part], device
        )
        raw_vectors[part] = vector.astype(np.float32)
        writers[part] = np.array(writer_list)
        selected_records = [record for record in records if record["part"] == part]
        page_ids[part] = np.array([record["page"] for record in selected_records])
        source_names[part] = np.array([record["source_name"] for record in selected_records])
        if list(writers[part].astype(str)) != [record["writer"] for record in selected_records]:
            raise RuntimeError(f"writer row alignment failed for {part}")
    event("raw_page_vectors_ready", dimensions={part: list(value.shape) for part, value in raw_vectors.items()})

    pca = PCA(n_components=module["PCA_DIM"], whiten=True, svd_solver="randomized", random_state=module["SEED"])
    z = {
        "train": pca.fit_transform(raw_vectors["train"]).astype(np.float32),
        "val": pca.transform(raw_vectors["val"]).astype(np.float32),
        "test": pca.transform(raw_vectors["test"]).astype(np.float32),
    }
    all_acq = np.stack([record["acq"] for record in records])
    all_ink = np.stack([record["ink"] for record in records])
    nuisance: dict[str, dict[str, np.ndarray]] = {"acquisition": {}, "ink": {}, "combined": {}}
    for part in ["train", "val", "test"]:
        nuisance["acquisition"][part] = all_acq[masks[part]]
        nuisance["ink"][part] = all_ink[masks[part]]
        nuisance["combined"][part] = np.concatenate([
            nuisance["acquisition"][part], nuisance["ink"][part]
        ], axis=1)

    residuals = {
        "resid_acquisition": module["fit_residual_model"](z["train"], nuisance["acquisition"]["train"]),
        "resid_ink": module["fit_residual_model"](z["train"], nuisance["ink"]["train"]),
        "resid_combined": module["fit_residual_model"](z["train"], nuisance["combined"]["train"]),
    }
    candidates: dict[str, dict[str, np.ndarray]] = {part: {"raw": z[part]} for part in ["val", "test"]}
    for part in ["val", "test"]:
        candidates[part]["resid_acquisition"] = residuals["resid_acquisition"].apply(z[part], nuisance["acquisition"][part])
        candidates[part]["resid_ink"] = residuals["resid_ink"].apply(z[part], nuisance["ink"][part])
        candidates[part]["resid_combined"] = residuals["resid_combined"].apply(z[part], nuisance["combined"][part])

    reproduction_result = json.loads(local_files["result.json"].read_text())
    selected = reproduction_result["selected_by_validation"]
    if selected != "resid_combined":
        raise RuntimeError(f"frozen selected representation changed: {selected}")
    validation_metrics = {name: helper.retrieval(value, writers["val"].tolist()) for name, value in candidates["val"].items()}
    test_metrics = {name: helper.retrieval(value, writers["test"].tolist()) for name, value in candidates["test"].items()}
    nuisance_metrics = {
        name: helper.retrieval(nuisance[name]["test"], writers["test"].tolist())
        for name in ["acquisition", "ink", "combined"]
    }

    metric_checks: dict[str, Any] = {}
    for group_name, observed_group, expected_group in [
        ("validation", validation_metrics, reproduction_result["validation_metrics"]),
        ("test", test_metrics, reproduction_result["test_metrics"]),
        ("nuisance", nuisance_metrics, reproduction_result["nuisance_metrics"]),
    ]:
        metric_checks[group_name] = {}
        for name, observed in observed_group.items():
            delta = abs(float(observed["map"]) - float(expected_group[name]["map"]))
            metric_checks[group_name][name] = {
                "observed_map": float(observed["map"]),
                "expected_map": float(expected_group[name]["map"]),
                "absolute_delta": delta,
                "pass": delta <= MAP_TOLERANCE,
            }

    pca_manual = {part: manual_pca_transform(raw_vectors[part], pca) for part in ["train", "val", "test"]}
    pca_errors = {part: max_abs(pca_manual[part], z[part]) for part in ["train", "val", "test"]}
    residual_values = {name: residual_to_dict(model) for name, model in residuals.items()}
    residual_errors = {
        "val": {
            "resid_acquisition": max_abs(manual_residual_apply(z["val"], nuisance["acquisition"]["val"], residual_values["resid_acquisition"]), candidates["val"]["resid_acquisition"]),
            "resid_ink": max_abs(manual_residual_apply(z["val"], nuisance["ink"]["val"], residual_values["resid_ink"]), candidates["val"]["resid_ink"]),
            "resid_combined": max_abs(manual_residual_apply(z["val"], nuisance["combined"]["val"], residual_values["resid_combined"]), candidates["val"]["resid_combined"]),
        },
        "test": {
            "resid_acquisition": max_abs(manual_residual_apply(z["test"], nuisance["acquisition"]["test"], residual_values["resid_acquisition"]), candidates["test"]["resid_acquisition"]),
            "resid_ink": max_abs(manual_residual_apply(z["test"], nuisance["ink"]["test"], residual_values["resid_ink"]), candidates["test"]["resid_ink"]),
            "resid_combined": max_abs(manual_residual_apply(z["test"], nuisance["combined"]["test"], residual_values["resid_combined"]), candidates["test"]["resid_combined"]),
        },
    }

    selected_x = candidates["test"][selected]
    observed_map = test_metrics[selected]["map"]
    permutation = module["permutation_p"](
        helper.retrieval, selected_x, writers["test"].tolist(), observed_map, module["PERMUTATIONS"]
    )
    k_calibration = module["k_calibration"](selected_x, writers["test"].tolist())
    perturbation: dict[str, Any] = {}
    test_record_indices = np.flatnonzero(masks["test"])
    for mode in ["contrast", "scale", "erosion", "dilation", "translation"]:
        vectors = []
        with torch.inference_mode():
            for record_index in test_record_indices:
                record = records[record_index]
                pp = module["perturb_patches"](
                    patch_xa[record["start"]:record["end"]],
                    patch_ta[record["start"]:record["end"]], mode,
                )
                total = None
                for start in range(0, len(pp), 128):
                    value = wrapped(module["eval_tensor"](pp[start:start + 128], device)).float().sum(0).cpu().numpy()
                    total = value if total is None else total + value
                total = np.sign(total) * np.abs(total) ** 0.4
                total = total / max(float(np.linalg.norm(total)), 1e-12)
                vectors.append(total.astype(np.float32))
        transformed = pca.transform(np.stack(vectors)).astype(np.float32)
        transformed = residuals[selected].apply(transformed, nuisance["combined"]["test"])
        score = helper.retrieval(transformed, writers["test"].tolist())
        perturbation[mode] = {**score, "retention": score["map"] / max(observed_map, 1e-12)}

    acquisition_map = nuisance_metrics["acquisition"]["map"]
    criteria = {
        "absolute_over_acquisition": observed_map - acquisition_map >= 0.05,
        "ratio_over_acquisition": observed_map / max(acquisition_map, 1e-12) >= 1.5,
        "permutation": permutation["p"] <= 0.01,
        "perturbation": all(value["retention"] >= 0.80 for value in perturbation.values()),
        "exact_k": k_calibration["exact_rate"] >= 0.70,
        "within_one_k": k_calibration["within_one_rate"] >= 0.90,
    }
    criteria["all_pass"] = all(criteria.values())

    transform_path = OUT / "v16_transform_bundle.npz"
    transform_arrays: dict[str, np.ndarray] = {
        "pca_mean": pca.mean_, "pca_components": pca.components_,
        "pca_explained_variance": pca.explained_variance_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_singular_values": pca.singular_values_,
        "pca_n_samples": np.asarray([pca.n_samples_], dtype=np.int64),
        "pca_n_features_in": np.asarray([pca.n_features_in_], dtype=np.int64),
        "pca_n_components": np.asarray([pca.n_components_], dtype=np.int64),
        "pca_whiten": np.asarray([1], dtype=np.int8),
        "selected_representation": np.asarray([selected]),
    }
    for name, values in residual_values.items():
        for key, value in values.items():
            transform_arrays[f"{name}_{key}"] = value
    np.savez_compressed(transform_path, **transform_arrays)

    feature_path = OUT / "v16_exact_features_with_provenance.npz"
    feature_arrays: dict[str, np.ndarray] = {}
    for part in ["train", "val", "test"]:
        feature_arrays[f"raw_page_vector_{part}"] = raw_vectors[part]
        feature_arrays[f"pca_{part}"] = z[part]
        feature_arrays[f"writer_{part}"] = writers[part]
        feature_arrays[f"page_id_{part}"] = page_ids[part]
        feature_arrays[f"source_name_{part}"] = source_names[part]
        feature_arrays[f"acquisition_{part}"] = nuisance["acquisition"][part]
        feature_arrays[f"ink_{part}"] = nuisance["ink"][part]
        feature_arrays[f"combined_{part}"] = nuisance["combined"][part]
    for part in ["val", "test"]:
        for name, value in candidates[part].items():
            feature_arrays[f"{name}_{part}"] = value
    np.savez_compressed(feature_path, **feature_arrays)

    page_manifest_path = OUT / "v16_page_manifest.json"
    page_manifest_path.write_bytes(stable_json({
        "schema": "blind-pal-saghog-v1.6-page-provenance",
        "archive_md5": EXPECTED_ARCHIVE_MD5,
        "writer_split_sha256": sha256_bytes(split_manifest_raw),
        "rows": [
            {key: value for key, value in record.items() if key not in {"acq", "ink"}}
            for record in records
        ],
    }))
    environment_path = OUT / "environment_versions.json"
    environment_path.write_bytes(stable_json(package_versions()))
    event_log_path = OUT / "export_events.json"

    all_metric_pass = all(
        check["pass"] for group in metric_checks.values() for check in group.values()
    )
    all_pca_pass = all(value <= ARRAY_TOLERANCE for value in pca_errors.values())
    all_residual_pass = all(
        value <= ARRAY_TOLERANCE for group in residual_errors.values() for value in group.values()
    )
    criteria_match = criteria == reproduction_result["criteria"]
    export_result = {
        "schema": "blind-pal-saghog-v1.6-transform-export",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "selected_representation": selected,
        "source_root": source_root,
        "target_root": target_root,
        "writer_split_sha256": sha256_bytes(split_manifest_raw),
        "metrics": {"validation": validation_metrics, "test": test_metrics, "nuisance": nuisance_metrics},
        "metric_checks": metric_checks,
        "pca_max_absolute_errors": pca_errors,
        "residual_max_absolute_errors": residual_errors,
        "permutation": permutation,
        "k_calibration": k_calibration,
        "perturbation": perturbation,
        "criteria": criteria,
        "criteria_match_reproduction": criteria_match,
        "counts": {part: int(masks[part].sum()) for part in masks},
        "acceptance": {
            "checkpoint_matches": checkpoint_sha == persisted["saghog_v15_best.pt"]["sha256"],
            "split_matches": sha256_bytes(split_manifest_raw) == reproduction_manifest["comparison"]["checks"]["writer_split"]["observed_sha256"],
            "metrics_within_tolerance": all_metric_pass,
            "pca_roundtrip": all_pca_pass,
            "residual_roundtrip": all_residual_pass,
            "selected_representation_frozen": selected == "resid_combined",
            "criteria_unchanged": criteria_match,
        },
        "seal": {"voynich_opened": False, "davis_labels_loaded": False, "f115r_loaded": False},
    }
    export_result["acceptance"]["all_pass"] = all(export_result["acceptance"].values())
    export_result_path = OUT / "v16_transform_export_result.json"
    export_result_path.write_bytes(stable_json(export_result))
    event_log_path.write_bytes(stable_json(events))

    persisted_export: dict[str, Any] = {}
    for path in [transform_path, feature_path, page_manifest_path, environment_path, event_log_path, export_result_path]:
        object_path = f"{target_root}/artifacts/{path.name}"
        event("upload_start", file=path.name, bytes=path.stat().st_size)
        persisted_export[path.name] = storage.put_file_verified(object_path, path)
        event("upload_done", file=path.name, sha256=persisted_export[path.name]["sha256"])

    final_manifest = {
        "schema": "blind-pal-saghog-v1.6-transform-export-persistence",
        "source_root": source_root, "target_root": target_root,
        "accepted": export_result["acceptance"]["all_pass"],
        "export_result": export_result,
        "persisted": persisted_export,
    }
    manifest_meta = storage.put_verified(
        f"{target_root}/manifest/transform_export_manifest.json", stable_json(final_manifest)
    )
    final_manifest["manifest_object"] = manifest_meta
    print("V16_TRANSFORM_EXPORT_RESULT " + json.dumps(final_manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
