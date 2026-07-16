#!/usr/bin/env python3
"""Build and round-trip the exact frozen S3 checkpoint payload on CPU."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
from v060_family_s_neural_common import NeuralTransducer, model_config
from v060_supabase_checkpoint_transport import persist_checkpoint, transport_from_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--filename", default="s3_neural_seed1731_u30000.pt")
    parser.add_argument(
        "--object-prefix",
        default="v060/s3/preflight-seed1731/u30000",
    )
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-s3-checkpoint-preflight",
    )
    language = languages["en"]
    config = model_config(len(language.alphabet))
    torch.manual_seed(args.seed)
    model = NeuralTransducer(**config)
    payload = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": config,
        "seed": args.seed,
        "updates": 30000,
        "completed_updates": 30000,
        "effective_batch": 32,
        "examples": 960000,
        "alphabet": list(language.alphabet),
        "corpus_manifest": str(root / "corpus_manifest_v050.json"),
    }
    with tempfile.TemporaryDirectory(prefix="v060-s3-preflight-") as tmp:
        checkpoint = Path(tmp) / args.filename
        torch.save(payload, checkpoint)
        manifest = persist_checkpoint(
            checkpoint,
            object_prefix=args.object_prefix,
            verify_roundtrip=True,
            **transport_from_environment(),
        )
    print(
        "V060_S3_EXACT_CHECKPOINT_ROUNDTRIP_PASS",
        json.dumps(
            {
                "filename": args.filename,
                "checkpoint_bytes": manifest["original_bytes"],
                "checkpoint_sha256": manifest["original_sha256"],
                "parts": len(manifest["parts"]),
                "manifest_object_path": manifest["manifest_object_path"],
                "roundtrip_verified": manifest["roundtrip_verified"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
