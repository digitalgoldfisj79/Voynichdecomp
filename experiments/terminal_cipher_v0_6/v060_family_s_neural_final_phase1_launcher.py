#!/usr/bin/env python3
"""Minimal launcher for the frozen S3 final-evaluation GPU phase."""
from __future__ import annotations

import sys
from pathlib import Path

import v060_family_s_neural_final_evaluate as evaluation

if __name__ == "__main__":
    repo = Path(sys.argv[1])
    args = type("Args", (), {
        "repo": repo,
        "signer_url": evaluation.DEFAULT_SIGNER_URL,
    })()
    evaluation.phase1(args)
