#!/usr/bin/env python3
"""Distributed fixed-update training for one final S3 neural-transducer seed."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
from v060_family_s_neural_common import (
    NeuralTransducer,
    SyntheticGenerator,
    collate,
    model_config,
)


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def learning_rate(update: int, updates: int, warmup: int, peak: float) -> float:
    if update < warmup:
        return peak * (update + 1) / max(1, warmup)
    progress = (update - warmup) / max(1, updates - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--batch-per-rank", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--upload-repo", default="Digitalgoldfish79/v060-terminal-checkpoints")
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32 - 1))
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / f"v060-s3-neural-{args.seed}-{rank}",
    )
    language = languages["en"]
    generator = SyntheticGenerator(
        language,
        core.stable_seed("v060-s3-neural-generator", args.seed, rank),
        plaintext_length=384,
    )
    config = model_config(len(language.alphabet))
    model = NeuralTransducer(**config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        fused=True,
    )
    ce_loss = torch.nn.CrossEntropyLoss()
    boundary_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    started = time.perf_counter()
    rolling_plain = 0.0
    rolling_boundary = 0.0

    for update in range(args.updates):
        lr = learning_rate(update, args.updates, args.warmup, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        examples = [generator.sample() for _ in range(args.batch_per_rank)]
        batch = collate(examples, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, boundary_logits = model(
                batch["source"],
                batch["line_flags"],
                batch["source_padding"],
                batch["target"],
            )
            plain = ce_loss(
                logits.reshape(-1, logits.shape[-1]), batch["target"].reshape(-1)
            )
            boundary_raw = boundary_loss(boundary_logits, batch["boundary"])
            mask = (~batch["source_padding"]).float()
            boundary = (boundary_raw * mask).sum() / mask.sum().clamp_min(1.0)
            loss = plain + 0.3 * boundary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rolling_plain += float(plain.detach())
        rolling_boundary += float(boundary.detach())

        if (update + 1) % 250 == 0:
            metrics = torch.tensor(
                [rolling_plain / 250.0, rolling_boundary / 250.0],
                dtype=torch.float64,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics /= world_size
            if rank == 0:
                print(
                    "V060_S3_NEURAL_TRAIN",
                    json.dumps(
                        {
                            "seed": args.seed,
                            "update": update + 1,
                            "updates": args.updates,
                            "plain_loss": float(metrics[0]),
                            "boundary_loss": float(metrics[1]),
                            "learning_rate": lr,
                            "elapsed_seconds": time.perf_counter() - started,
                            "examples_per_model": (update + 1)
                            * args.batch_per_rank
                            * world_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            rolling_plain = 0.0
            rolling_boundary = 0.0

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        underlying = model.module if isinstance(model, DDP) else model
        checkpoint = {
            "state_dict": {key: value.detach().cpu() for key, value in underlying.state_dict().items()},
            "config": config,
            "seed": args.seed,
            "updates": args.updates,
            "effective_batch": args.batch_per_rank * world_size,
            "examples": args.updates * args.batch_per_rank * world_size,
            "alphabet": list(language.alphabet),
            "corpus_manifest": str(root / "corpus_manifest_v050.json"),
        }
        local_path = Path("/tmp") / args.filename
        torch.save(checkpoint, local_path)
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(
            args.upload_repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        url = api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=args.filename,
            repo_id=args.upload_repo,
            repo_type="dataset",
            commit_message=f"Upload fixed S3 neural checkpoint seed {args.seed}",
        )
        print(
            "V060_S3_NEURAL_COMPLETE",
            json.dumps(
                {
                    "seed": args.seed,
                    "updates": args.updates,
                    "examples": checkpoint["examples"],
                    "effective_batch": checkpoint["effective_batch"],
                    "filename": args.filename,
                    "upload_url": url,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
