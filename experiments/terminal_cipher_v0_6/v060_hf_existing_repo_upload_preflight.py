#!/usr/bin/env python3
"""Verify upload permission to an already-existing HF dataset repository."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time

from huggingface_hub import HfApi

REPO_ID = "Digitalgoldfish79/v060-terminal-checkpoints"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not injected")
    api = HfApi(token=token)
    identity = api.whoami()
    api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    payload = {
        "probe": "v060-existing-repo-upload",
        "repository": REPO_ID,
        "identity": identity.get("name"),
        "unix_time": time.time(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    final = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path_in_repo = "preflight/v060_existing_repo_upload_probe.json"
    url = api.upload_file(
        path_or_fileobj=io.BytesIO(final),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Verify v0.6 checkpoint upload permission",
    )
    print(
        "V060_HF_EXISTING_UPLOAD_OK",
        json.dumps(
            {
                "identity": identity.get("name"),
                "repository": REPO_ID,
                "path": path_in_repo,
                "sha256": payload["sha256"],
                "url": str(url),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
