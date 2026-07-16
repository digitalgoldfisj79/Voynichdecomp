#!/usr/bin/env python3
"""Execution-only smoke wrapper for the fixed S3 neural transducer.

This wrapper suppresses Hub upload so container/GPU correctness can be tested
without forwarding credentials. It does not change model, data, optimizer or
loss code.
"""
from __future__ import annotations

import sys

import v060_family_s_neural_train as train


class _LocalOnlyApi:
    def __init__(self, *args, **kwargs):
        pass

    def create_repo(self, *args, **kwargs):
        return None

    def upload_file(self, *args, **kwargs):
        return "local-smoke-only"


train.HfApi = _LocalOnlyApi
sys.argv = [
    "v060_family_s_neural_train.py",
    "--repo", "/tmp/v",
    "--seed", "1731",
    "--updates", "5",
    "--batch-per-rank", "2",
    "--warmup", "2",
    "--filename", "s3_smoke_seed1731.pt",
]
train.main()
