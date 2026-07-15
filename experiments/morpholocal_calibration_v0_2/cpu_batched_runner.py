#!/usr/bin/env python3
"""CPU fallback for the batched morpholocal development accelerator.

Uses the same sufficient-statistic candidate scorer and deterministic batched
annealing as gpu_runner.py, but executes independent trials across CPU worker
processes. This avoids GPU allocator delays while retaining the rewritten
optimizer. It is a development sensitivity, not a frozen formal result.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]


def load_gpu_runner():
    path = HERE / "gpu_runner.py"
    spec = importlib.util.spec_from_file_location("morpholocal_gpu_runner_cpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import gpu_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cpu_batched_anneal_factory(gr):
    def cpu_batched_anneal(chains: list[dict[str, Any]], device: torch.device, steps: int):
        if not chains:
            return []
        dtype = torch.float64
        n_cells = len(chains[0]["base"])
        max_units = max(len(row["stationary"]) for row in chains)
        batch = len(chains)

        pair_np = np.stack([row["pair"] for row in chains]).astype(np.float64)
        uni_np = np.stack([row["uni"] for row in chains]).astype(np.float64)
        assignment_np = np.stack([row["base"] for row in chains]).astype(np.int64)
        log_t_np = np.full((batch, max_units, max_units), -1e12, dtype=np.float64)
        log_s_np = np.full((batch, max_units), -1e12, dtype=np.float64)
        prop_a = np.empty((steps, batch), dtype=np.int64)
        prop_b = np.empty((steps, batch), dtype=np.int64)
        prop_u = np.empty((steps, batch), dtype=np.float64)

        for index, row in enumerate(chains):
            units = len(row["stationary"])
            log_t_np[index, :units, :units] = np.log2(np.clip(row["transition"], 1e-300, None))
            log_s_np[index, :units] = np.log2(np.clip(row["stationary"], 1e-300, None))
            a, b, u = gr._chain_randomness(int(row["chain_seed"]), steps, n_cells)
            prop_a[:, index] = a
            prop_b[:, index] = b
            prop_u[:, index] = u

        pair = torch.as_tensor(pair_np, dtype=dtype, device=device)
        uni = torch.as_tensor(uni_np, dtype=dtype, device=device)
        current = torch.as_tensor(assignment_np, dtype=torch.long, device=device)
        log_t = torch.as_tensor(log_t_np, dtype=dtype, device=device)
        log_s = torch.as_tensor(log_s_np, dtype=dtype, device=device)
        prop_a_t = torch.as_tensor(prop_a, dtype=torch.long, device=device)
        prop_b_t = torch.as_tensor(prop_b, dtype=torch.long, device=device)
        prop_u_t = torch.as_tensor(prop_u, dtype=dtype, device=device)
        indices = torch.arange(batch, device=device)

        score = gr.objective_batch(pair, uni, current, log_t, log_s)
        best_score = score.clone()
        best = current.clone()

        for step in range(steps):
            a = prop_a_t[step]
            b = prop_b_t[step]
            value_a = current[indices, a]
            value_b = current[indices, b]
            different = value_a != value_b
            candidate = current.clone()
            candidate[indices, a] = value_b
            candidate[indices, b] = value_a
            candidate_score = gr.objective_batch(pair, uni, candidate, log_t, log_s)
            delta = candidate_score - score
            fraction = step / max(1, steps - 1)
            temperature = 6.0 * (0.05 / 6.0) ** fraction
            probability = torch.exp(torch.clamp(delta / max(temperature, 1e-12), max=0.0))
            accept = different & ((delta >= 0.0) | (prop_u_t[step] < probability))
            current = torch.where(accept[:, None], candidate, current)
            score = torch.where(accept, candidate_score, score)
            improve = score > best_score
            best = torch.where(improve[:, None], current, best)
            best_score = torch.where(improve, score, best_score)

        best_np = best.numpy()
        score_np = best_score.numpy()
        return [
            {"mapping": tuple(int(x) for x in best_np[i]), "score": float(score_np[i])}
            for i in range(batch)
        ]

    return cpu_batched_anneal


def run_one(repo_text: str, task: dict[str, Any], steps: int, restarts: int):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    gr = load_gpu_runner()
    gr.batched_anneal = cpu_batched_anneal_factory(gr)
    repo = Path(repo_text)
    impl = gr.load_impl(repo)
    device = torch.device("cpu")
    impl.candidate_fit = functools.partial(gr.gpu_candidate_fit, impl, device)
    record_path, ci_path = impl.locate_data(repo)
    registry = impl.build_surface_registry(pickle.load(record_path.open("rb")))
    external = impl.build_external_models(pickle.load(ci_path.open("rb")))

    kind = str(task["kind"])
    index = int(task["index"])
    seed = int(task["seed"])
    if kind == "positive":
        scenario = impl.scenario_for_index(index)
        events, keys, sizes = impl.generate_cipher_trial(
            registry, external, seed,
            str(scenario["key_scheme"]), int(scenario["null_count"]),
            str(scenario["size_profile"]), str(scenario["external_profile"]),
            str(scenario["selection_policy"]), str(scenario["selector"]),
        )
        truth = {**scenario, "keys": keys, "class_sizes": list(sizes)}
        result = impl.score_trial(events, registry, external, seed ^ 0xC1F3, steps, restarts, truth)
        result.update({"trial_type": "positive", "trial_index": index, "seed": seed})
    else:
        family = impl.CONTROL_FAMILIES[index % len(impl.CONTROL_FAMILIES)]
        selector = impl.SELECTORS[(index // len(impl.CONTROL_FAMILIES)) % len(impl.SELECTORS)]
        events = impl.generate_control_trial(registry, external, seed, family, selector)
        result = impl.score_trial(events, registry, external, seed ^ 0xF00D, steps, restarts, None)
        result.update({
            "trial_type": "control", "control_family": family,
            "true_selector": selector, "trial_index": index, "seed": seed,
            "false_positive": bool(result["cipher_selected"]),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--positives", type=int, default=96)
    parser.add_argument("--controls", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--output", type=Path, default=Path("CPU_BATCHED_DEVELOPMENT_FACTORIAL.json"))
    args = parser.parse_args()

    gr = load_gpu_runner()
    gr.batched_anneal = cpu_batched_anneal_factory(gr)
    impl = gr.load_impl(args.repo)
    validation = gr.validate(impl, args.repo, torch.device("cpu"))
    print("CPU_BATCHED_VALIDATION", json.dumps(validation, sort_keys=True), flush=True)

    tasks = [
        {"kind": "positive", "index": i, "seed": args.seed + 100000 + i * 7919}
        for i in range(args.positives)
    ] + [
        {"kind": "control", "index": i, "seed": args.seed + 900000 + i * 104729}
        for i in range(args.controls)
    ]
    started = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(run_one, str(args.repo), task, args.steps, args.restarts): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            task = futures[future]
            print(
                f"CPU_BATCHED_PROGRESS completed={completed}/{len(tasks)} "
                f"kind={task['kind']} index={task['index']} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    results.sort(key=lambda row: (row["trial_type"], int(row["trial_index"])))
    summary = impl.aggregate(results)
    payload = {
        "programme": impl.VERSION,
        "development_accelerator": "torch-cpu-batched-anneal-v0.2",
        "formal_seed": args.seed,
        "data_hashes": impl.EXPECTED_HASHES,
        "parameters": {
            "positives": args.positives, "controls": args.controls,
            "anneal_steps": args.steps, "anneal_restarts": args.restarts,
            "workers": args.workers, "torch_version": torch.__version__,
            "candidate_grid_unchanged": True,
            "optimizer_changed_from_scalar_cpu": True,
        },
        "summary": summary,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
