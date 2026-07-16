#!/usr/bin/env python3
"""Cheap exact-path persistence preflight for frozen S3 ensemble seed 1732."""
from __future__ import annotations

import hashlib
import json
import os

import requests

SIGNER = os.environ.get(
    "V060_SUPABASE_SIGNER_URL",
    "https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/v060-checkpoint-sign",
)
PATH = (
    "v060/s3/preflight-seed1732/u30000/"
    "s3_neural_seed1732_u30000.pt.part-99999"
)
PAYLOAD = b"v060-seed1732-exact-path-preflight\n"


def signed_url(action: str) -> str:
    response = requests.post(
        SIGNER,
        json={"action": action, "path": PATH},
        timeout=30,
    )
    response.raise_for_status()
    url = response.json().get("signedUrl")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"missing signed URL for action={action}")
    return url


def main() -> None:
    upload = requests.put(
        signed_url("upload"),
        data=PAYLOAD,
        headers={"content-type": "application/octet-stream"},
        timeout=30,
    )
    upload.raise_for_status()
    download = requests.get(signed_url("download"), timeout=30)
    download.raise_for_status()
    if download.content != PAYLOAD:
        raise RuntimeError("seed-1732 exact-path roundtrip mismatch")
    print(
        "V060_S3_SEED1732_NAMESPACE_PREFLIGHT_PASS",
        json.dumps(
            {
                "bytes": len(PAYLOAD),
                "path": PATH,
                "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
