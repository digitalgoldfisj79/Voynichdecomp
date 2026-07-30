"""Process-wide network pacing for the corpus-acquisition environment.

Python imports this module automatically from the experiment working directory.
It only wraps urllib.request.urlopen. Scientific selection and text processing are
unchanged; the wrapper serializes public-API requests and retries HTTP 429 using
the server's Retry-After value when supplied.
"""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

_ORIGINAL_URLOPEN = urllib.request.urlopen
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_MIN_INTERVAL_SECONDS = 0.80
_MAX_429_RETRIES = 8


def _retry_after_seconds(error: urllib.error.HTTPError, attempt: int) -> float:
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
    return min(120.0, 15.0 * (2 ** attempt))


def _paced_urlopen(url, data=None, timeout=None, *args, **kwargs):
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
            time.sleep(_retry_after_seconds(exc, attempt))
    raise AssertionError("unreachable")


urllib.request.urlopen = _paced_urlopen
