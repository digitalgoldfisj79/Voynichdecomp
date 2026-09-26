#!/usr/bin/env python3
"""Load the compressed, reviewable f116v preflight implementation."""
from __future__ import annotations

import base64
from pathlib import Path
import zlib

_payload_path = Path(__file__).with_name("run_pipeline.py.b85")
_source = zlib.decompress(base64.b85decode(_payload_path.read_text(encoding="ascii"))).decode("utf-8")
exec(compile(_source, str(Path(__file__).with_name("run_pipeline_impl.py")), "exec"), globals(), globals())
