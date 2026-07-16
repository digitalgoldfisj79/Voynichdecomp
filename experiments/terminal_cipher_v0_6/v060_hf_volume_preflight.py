#!/usr/bin/env python3
"""Verify that the mounted Hugging Face checkpoint dataset is writable."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    root = Path("/checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "v060-hf-volume-write",
        "unix_time": time.time(),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    target = root / "v060_volume_write_probe.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    reread = json.loads(target.read_text(encoding="utf-8"))
    assert reread["sha256"] == payload["sha256"]
    print("V060_HF_VOLUME_PREFLIGHT_OK", json.dumps({"path": str(target), "sha256": payload["sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
