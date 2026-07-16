#!/usr/bin/env python3
"""Cheap mandatory preflight for v0.6 neural checkpoint persistence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="Digitalgoldfish79/v060-terminal-checkpoints",
    )
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not present in the job environment")

    api = HfApi(token=token)
    identity = api.whoami()
    name = identity.get("name", "UNKNOWN")
    auth = identity.get("auth", {}) if isinstance(identity, dict) else {}
    access = auth.get("accessToken", {}) if isinstance(auth, dict) else {}
    role = access.get("role") if isinstance(access, dict) else None
    print(
        "V060_HF_PREFLIGHT_IDENTITY",
        json.dumps({"name": name, "token_role": role}, sort_keys=True),
        flush=True,
    )

    if args.create:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
    else:
        api.repo_info(repo_id=args.repo_id, repo_type="dataset")

    payload = f"v060 checkpoint permission probe {time.time_ns()}\n".encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = f"permission_probes/{digest}.txt"
    url = api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo=path,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Verify v0.6 checkpoint write permission",
    )
    print(
        "V060_HF_PREFLIGHT_PASS",
        json.dumps(
            {
                "repo_id": args.repo_id,
                "path": path,
                "sha256": digest,
                "upload_url": str(url),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
