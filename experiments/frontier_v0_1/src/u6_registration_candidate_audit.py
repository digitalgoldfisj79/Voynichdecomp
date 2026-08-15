from __future__ import annotations

import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

PIPELINE_REPO = "Digitalgoldfish79/voynich-dinov3-pipeline"
TARGETS = ["f32r", "f39r", "f40r"]
BRIDGE = "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/vtps_hf_bridge_20260814"
SECRET = "frontier-u6-stageb-20260815"
BRIDGE_ID = "u6-stageb-20260815-registration-candidate-audit"


def post(obj: dict, phase: str) -> None:
    r = requests.post(
        BRIDGE,
        json={
            "secret": SECRET,
            "id": BRIDGE_ID,
            "payload": json.dumps(obj, sort_keys=True, default=str),
            "meta": {"phase": phase},
        },
        timeout=120,
    )
    r.raise_for_status()


def reg_to_dict(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    d = {}
    for name in [
        "folio", "canvas_label", "service_id", "matches", "inliers",
        "inlier_ratio", "median_reproj_px", "p95_reproj_px", "deriv_px",
        "H_deriv", "deriv_scale", "passed", "reason"
    ]:
        if hasattr(x, name):
            d[name] = getattr(x, name)
    return d


def main() -> int:
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    info = api.repo_info(repo_id=PIPELINE_REPO, repo_type="dataset")
    revision = info.sha
    root = snapshot_download(
        PIPELINE_REPO,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    sys.path.insert(0, root)
    from vdino3 import cfg, register, sources

    source = inspect.getsource(register.register_folio)
    cfg_values = {
        k: getattr(cfg, k)
        for k in [
            "REG_DERIVATIVE_PX", "REG_MIN_INLIERS", "REG_MIN_INLIER_RATIO",
            "REG_MAX_MEDIAN_REPROJ_PX", "REG_SIFT_NFEATURES", "REG_CLAHE_CLIP",
            "REG_CLAHE_GRID", "REG_RATIO_TEST", "REG_USAC_CONF", "REG_USAC_REPROJ"
        ]
    }
    out = {
        "schema": "u6-registration-candidate-audit-v0.1",
        "pipeline_repo": PIPELINE_REPO,
        "pipeline_revision": revision,
        "targets": TARGETS,
        "register_folio_source": source,
        "cfg": cfg_values,
        "started_unix": time.time(),
        "target_opened": False,
        "true_retention_read": False,
        "results": {},
    }
    post(out, "source-freeze")

    manifest = sources.yale_manifest()
    canvases = sources.yale_canvases(manifest)
    for folio in TARGETS:
        try:
            candidate_meta = register.candidate_service_ids(folio, canvases)
            best, scores = register.register_folio(folio, canvases, max_candidates=6)
            out["results"][folio] = {
                "candidate_service_ids": candidate_meta,
                "best": reg_to_dict(best),
                "scores": [reg_to_dict(x) for x in scores],
            }
        except Exception as exc:
            out["results"][folio] = {
                "error": repr(exc),
                "traceback": traceback.format_exc()[-12000:],
            }
        post(out, f"after-{folio}")

    out["finished_unix"] = time.time()
    out["all_candidates_passing"] = {
        folio: [
            x for x in rec.get("scores", [])
            if isinstance(x, dict) and bool(x.get("passed"))
        ]
        for folio, rec in out["results"].items()
    }
    post(out, "complete")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
