#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v070_oracle_source_transfer as oracle

# Build and hash-verify all corpus/model assets once in the parent. Forked
# workers inherit the immutable registry and never race on shared cache files.
oracle.p.get_assets(Path("/tmp/v"))

if __name__ == "__main__":
    oracle.main()
