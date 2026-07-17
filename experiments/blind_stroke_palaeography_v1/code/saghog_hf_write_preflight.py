#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time

from huggingface_hub import HfApi

REPO_ID = "Digitalgoldfish79/voynich-saghog-v15-reproduction-20260717"
RUN_ID = "v15-reproduction-20260717"


def discussion_num_from_pr_url(pr_url: str) -> int:
    match = re.search(r"/discussions/(\d+)(?:$|[?#])", pr_url)
    if not match:
        raise RuntimeError(f"cannot parse discussion number from PR URL: {pr_url}")
    return int(match.group(1))


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
    info = api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=f"write_tests/{RUN_ID}.json",
        path_or_fileobj=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=f"Write-permission preflight for {RUN_ID}",
        create_pr=True,
    )
    pr_url = str(info.pr_url or "")
    if not pr_url:
        raise RuntimeError("PR-based upload did not return a pull-request URL")
    discussion_num = discussion_num_from_pr_url(pr_url)
    api.merge_pull_request(
        repo_id=REPO_ID,
        repo_type="dataset",
        discussion_num=discussion_num,
        comment="Automatic merge after SAGHOG persistence write preflight.",
    )
    payload["pr_url"] = pr_url
    payload["discussion_num"] = discussion_num
    payload["merged"] = True
    print("SAGHOG_WRITE_TEST_OK " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
