#!/usr/bin/env python3
"""Run one acquisition script with deterministic Wikimedia API pacing.

This launcher changes transport mechanics only. It serializes requests, honours
HTTP 429 Retry-After values, and converts MediaWiki/Wikibase JSON error payloads
into retryable URL errors. Source ordering, eligibility and normalization remain
inside the invoked frozen acquisition script.
"""
from __future__ import annotations

import json
import runpy
import sys
import threading
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

_ORIGINAL_URLOPEN = urllib.request.urlopen
_ORIGINAL_JSON_LOAD = json.load
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_MIN_INTERVAL_SECONDS = 1.50
_MAX_429_RETRIES = 10


def retry_after_seconds(error: urllib.error.HTTPError, attempt: int) -> float:
    value = error.headers.get("Retry-After") if error.headers else None
    if value:
        try:
            return max(1.0, float(value))
        except ValueError:
            try:
                target = parsedate_to_datetime(value).timestamp()
                return max(1.0, target - time.time())
            except Exception:
                pass
    return min(180.0, 20.0 * (2 ** attempt))


def paced_urlopen(url, data=None, timeout=None, *args, **kwargs):
    global _LAST_REQUEST
    for attempt in range(_MAX_429_RETRIES + 1):
        with _LOCK:
            delay = _MIN_INTERVAL_SECONDS - (time.monotonic() - _LAST_REQUEST)
            if delay > 0:
                time.sleep(delay)
            _LAST_REQUEST = time.monotonic()
        try:
            return _ORIGINAL_URLOPEN(url, data=data, timeout=timeout, *args, **kwargs)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt >= _MAX_429_RETRIES:
                raise
            time.sleep(retry_after_seconds(exc, attempt))
    raise AssertionError("unreachable")


def checked_json_load(fp, *args, **kwargs):
    value = _ORIGINAL_JSON_LOAD(fp, *args, **kwargs)
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        error = value["error"]
        code = str(error.get("code", "api_error"))
        info = str(error.get("info", error))
        raise urllib.error.URLError(f"MediaWiki API error {code}: {info}")
    return value


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: acquisition_launcher.py SCRIPT [SCRIPT_ARGS ...]")
    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    urllib.request.urlopen = paced_urlopen
    json.load = checked_json_load
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
