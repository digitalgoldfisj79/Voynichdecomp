#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time

from huggingface_hub import HfApi

REPO_ID = "Digitalgoldfish79/voynich-saghog-v15-reproduction-20260717"
RUN_ID = "v15-reproduction-20260717"


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing")

    api = HfApi(token=token)
    identity = api.whoami()
    payload = {
        "schema": "blind-pal-saghog-persistence-preflight-v1",
        "run_id": RUN_ID,
        "timestamp_unix": time.time(),
        "authenticated_as": identity.get("name"),
        "repo_id": REPO_ID,
    }
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=f"write_tests/{RUN_ID}.json",
        path_or_fileobj=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=f"Write-permission preflight for {RUN_ID}",
    )
    print("SAGHOG_WRITE_TEST_OK " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
