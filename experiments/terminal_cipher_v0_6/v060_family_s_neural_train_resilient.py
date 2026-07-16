#!/usr/bin/env python3
"""Resilient fixed-update S3 neural training with mandatory Hub write preflight."""
from __future__ import annotations

import argparse
import hashlib
import io
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


def upload_with_retries(
    api: HfApi,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    message: str,
    attempts: int = 5,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            url = api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=message,
            )
            return str(url)
        except Exception as exc:  # preserve training against transient Hub failures
            last_error = exc
            print(
                "V060_S3_CHECKPOINT_UPLOAD_RETRY",
                json.dumps(
                    {
                        "attempt": attempt,
                        "attempts": attempts,
                        "path": path_in_repo,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(60, 2 ** attempt))
    assert last_error is not None
    raise last_error


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict,
    language,
    root: Path,
    seed: int,
    updates: int,
    completed_updates: int,
    batch_per_rank: int,
    world_size: int,
) -> dict:
    underlying = model.module if isinstance(model, DDP) else model
    return {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in underlying.state_dict().items()
        },
        "config": config,
        "seed": seed,
        "updates": updates,
        "completed_updates": completed_updates,
        "effective_batch": batch_per_rank * world_size,
        "examples": completed_updates * batch_per_rank * world_size,
        "alphabet": list(language.alphabet),
        "corpus_manifest": str(root / "corpus_manifest_v050.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--batch-per-rank", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--upload-repo",
        default="Digitalgoldfish79/v060-terminal-checkpoints",
    )
    parser.add_argument("--filename", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    api: HfApi | None = None

    # Fail within seconds, before corpus loading or training, unless this exact
    # injected token can both see and write the already-created target dataset.
    if rank == 0:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not present in the training job")
        api = HfApi(token=token)
        api.repo_info(repo_id=args.upload_repo, repo_type="dataset")
        probe = f"training preflight {time.time_ns()}\n".encode()
        digest = hashlib.sha256(probe).hexdigest()
        probe_url = api.upload_file(
            path_or_fileobj=io.BytesIO(probe),
            path_in_repo=f"permission_probes/training-{digest}.txt",
            repo_id=args.upload_repo,
            repo_type="dataset",
            commit_message="Verify S3 training checkpoint write permission",
        )
        print(
            "V060_S3_TRAINING_PREFLIGHT_PASS",
            json.dumps(
                {
                    "repo_id": args.upload_repo,
                    "probe_sha256": digest,
                    "upload_url": str(probe_url),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()

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
    boundary_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    started = time.perf_counter()
    rolling_plain = 0.0
    rolling_boundary = 0.0

    final_url: str | None = None
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
            boundary_raw = boundary_loss_fn(boundary_logits, batch["boundary"])
            mask = (~batch["source_padding"]).float()
            boundary = (boundary_raw * mask).sum() / mask.sum().clamp_min(1.0)
            loss = plain + 0.3 * boundary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rolling_plain += float(plain.detach())
        rolling_boundary += float(boundary.detach())

        completed = update + 1
        if completed % 250 == 0:
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
                            "update": completed,
                            "updates": args.updates,
                            "plain_loss": float(metrics[0]),
                            "boundary_loss": float(metrics[1]),
                            "learning_rate": lr,
                            "elapsed_seconds": time.perf_counter() - started,
                            "examples_per_model": completed
                            * args.batch_per_rank
                            * world_size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            rolling_plain = 0.0
            rolling_boundary = 0.0

        checkpoint_due = (
            args.checkpoint_every > 0
            and completed % args.checkpoint_every == 0
        ) or completed == args.updates
        if checkpoint_due:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                assert api is not None
                payload = checkpoint_payload(
                    model,
                    config,
                    language,
                    root,
                    args.seed,
                    args.updates,
                    completed,
                    args.batch_per_rank,
                    world_size,
                )
                if completed == args.updates:
                    path_in_repo = args.filename
                else:
                    source = Path(args.filename)
                    path_in_repo = (
                        f"recovery/{source.stem}.u{completed:05d}{source.suffix}"
                    )
                local_path = Path("/tmp") / Path(path_in_repo).name
                torch.save(payload, local_path)
                url = upload_with_retries(
                    api,
                    local_path,
                    path_in_repo,
                    args.upload_repo,
                    f"Upload S3 checkpoint seed {args.seed} update {completed}",
                )
                print(
                    "V060_S3_NEURAL_CHECKPOINT",
                    json.dumps(
                        {
                            "seed": args.seed,
                            "completed_updates": completed,
                            "examples": payload["examples"],
                            "path": path_in_repo,
                            "upload_url": url,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if completed == args.updates:
                    final_url = url
            if world_size > 1:
                dist.barrier()

    if rank == 0:
        print(
            "V060_S3_NEURAL_COMPLETE",
            json.dumps(
                {
                    "seed": args.seed,
                    "updates": args.updates,
                    "examples": args.updates * args.batch_per_rank * world_size,
                    "effective_batch": args.batch_per_rank * world_size,
                    "filename": args.filename,
                    "upload_url": final_url,
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
