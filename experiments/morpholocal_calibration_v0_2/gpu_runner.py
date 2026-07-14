#!/usr/bin/env python3
"""Batched multi-GPU development runner for morpholocal calibration v0.2.

This is a development accelerator, not a frozen result. It preserves the
candidate grid and scoring conventions, but replaces scalar simulated annealing
with deterministic batched annealing and scores candidate models from sufficient
statistics rather than rescanning the full training corpus for each candidate.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]


def load_impl(repo: Path):
    wrapper = repo / "experiments/morpholocal_calibration_v0_2/morpholocal_gate.py"
    impl = repo / "experiments/morpholocal_calibration_v0_2/morpholocal_gate_impl.py"
    patcher = repo / "experiments/morpholocal_calibration_v0_2/apply_development_patch.py"
    if not impl.exists():
        subprocess.run(
            [
                sys.executable, str(wrapper), "--repo", str(repo),
                "--positives", "0", "--controls", "0", "--workers", "1",
                "--steps", "1", "--restarts", "1", "--seed", "1",
                "--output", "/tmp/morpholocal-bootstrap.json",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run([sys.executable, str(patcher), str(impl)], check=True)
    name = f"morpholocal_gpu_impl_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, impl)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import morpholocal implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def objective_batch(
    pair: torch.Tensor,
    uni: torch.Tensor,
    assignment: torch.Tensor,
    log_transition: torch.Tensor,
    log_stationary: torch.Tensor,
) -> torch.Tensor:
    batch = assignment.shape[0]
    bidx = torch.arange(batch, device=assignment.device)[:, None, None]
    left = assignment[:, :, None]
    right = assignment[:, None, :]
    transition_values = log_transition[bidx, left, right]
    score = (pair * transition_values).sum(dim=(1, 2))
    stationary_values = torch.gather(log_stationary, 1, assignment)
    score = score + 0.25 * (uni * stationary_values).sum(dim=1)
    return score


def _chain_randomness(seed: int, steps: int, n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed & ((1 << 63) - 1))
    a = rng.integers(0, n_cells, size=steps, dtype=np.int64)
    b = rng.integers(0, n_cells - 1, size=steps, dtype=np.int64)
    b = b + (b >= a)
    u = rng.random(steps)
    return a, b, u


def batched_anneal(chains: list[dict[str, Any]], device: torch.device, steps: int):
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
        a, b, u = _chain_randomness(int(row["chain_seed"]), steps, n_cells)
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

    score = objective_batch(pair, uni, current, log_t, log_s)
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
        candidate_score = objective_batch(pair, uni, candidate, log_t, log_s)
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

    torch.cuda.synchronize(device)
    best_np = best.cpu().numpy()
    score_np = best_score.cpu().numpy()
    return [
        {"mapping": tuple(int(x) for x in best_np[i]), "score": float(score_np[i])}
        for i in range(batch)
    ]


def sufficient_statistics(module, train, scheme: str, n_cells: int):
    pairs, unis = module.cell_statistics(train, scheme, n_cells)
    starts = {label: np.zeros(n_cells, dtype=float) for label in pairs}
    for line in module.lines_from_events(train):
        if line:
            event = line[0]
            label = module.key_label(event, scheme)
            starts[label][event.cell] += 1.0
    return pairs, unis, starts


def fast_cost_components(module, assignments, pairs, unis, starts, transition, stationary):
    log_t = np.log2(np.clip(transition, 1e-300, None))
    log_s = np.log2(np.clip(stationary, 1e-300, None))
    external_bits = 0.0
    emission_bits = 0.0
    for label in sorted(assignments):
        mapping = np.asarray(assignments[label], dtype=int)
        external_bits -= float((starts[label] * log_s[mapping]).sum())
        external_bits -= float((pairs[label] * log_t[np.ix_(mapping, mapping)]).sum())
        for unit in sorted(set(int(x) for x in mapping)):
            support = np.flatnonzero(mapping == unit)
            emission_bits += module.kt_bits([int(unis[label][c]) for c in support])
    return external_bits, emission_bits


def structural_report(module, sizes, null_count: int, key_count: int, external_profile: str):
    record = {
        "codec_version": module.CODEC_VERSION,
        "class_sizes": list(sizes) + ([null_count] if null_count else []),
        "num_states": key_count,
        "outdegrees": [1] * key_count,
        "transition_counts": [],
        "emission_counts": [],
        "latent_path_mode": "none",
        "external_model_count": len(module.PROFILE_NAMES),
        "external_model_index": module.PROFILE_NAMES.index(external_profile),
    }
    return module.cost_model(record)


def gpu_candidate_fit(module, device: torch.device, train, registry, external, seed: int, steps: int, restarts: int):
    selected_selector, selector_scores = module.select_selector(train, registry)
    selector_bits = module.selector_nll(train, registry, selected_selector)
    configurations: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []

    for scheme in module.KEY_SCHEMES:
        pairs, unis, starts = sufficient_statistics(module, train, scheme, len(registry.cells))
        for null_count in module.NULL_COUNTS:
            for size_profile in module.SIZE_PROFILES:
                sizes = module.class_sizes(size_profile, len(registry.cells) - null_count)
                for external_profile in module.PROFILE_NAMES:
                    transition, stationary = module.extend_external_model(external, external_profile, null_count)
                    config_id = len(configurations)
                    configurations.append({
                        "id": config_id,
                        "scheme": scheme,
                        "null_count": null_count,
                        "size_profile": size_profile,
                        "sizes": sizes,
                        "external_profile": external_profile,
                        "transition": transition,
                        "stationary": stationary,
                        "pairs": pairs,
                        "unis": unis,
                        "starts": starts,
                    })
                    for key_index, label in enumerate(sorted(pairs)):
                        anneal_seed = (
                            seed ^ (key_index * 0x9E3779B1)
                            ^ module.stable_seed(scheme, null_count, size_profile, external_profile)
                        )
                        base = module.initial_assignment(unis[label], sizes, null_count, stationary)
                        for restart in range(restarts):
                            current = list(base)
                            chain_seed = anneal_seed ^ ((restart + 1) * 0xD1B54A32D192ED03)
                            if restart:
                                random.Random(chain_seed).shuffle(current)
                            chains.append({
                                "config_id": config_id,
                                "label": label,
                                "restart": restart,
                                "chain_seed": chain_seed,
                                "pair": pairs[label],
                                "uni": unis[label],
                                "transition": transition,
                                "stationary": stationary,
                                "base": current,
                            })

    outputs = batched_anneal(chains, device, steps)
    winners: dict[tuple[int, str], dict[str, Any]] = {}
    for chain, output in zip(chains, outputs):
        key = (int(chain["config_id"]), str(chain["label"]))
        previous = winners.get(key)
        if previous is None or (output["score"], output["mapping"]) > (
            previous["score"], previous["mapping"]
        ):
            winners[key] = output

    best = None
    for config in configurations:
        labels = sorted(config["pairs"])
        assignments = {
            label: winners[(config["id"], label)]["mapping"] for label in labels
        }
        objective = sum(winners[(config["id"], label)]["score"] for label in labels)
        external_bits, emission_bits = fast_cost_components(
            module, assignments, config["pairs"], config["unis"], config["starts"],
            config["transition"], config["stationary"],
        )
        report = structural_report(
            module, config["sizes"], config["null_count"], len(assignments),
            config["external_profile"],
        )
        selection_score = (
            report.partition_bits * len(assignments)
            + report.topology_bits
            + report.external_model_index_bits
            + external_bits + emission_bits + selector_bits
        )
        row = {
            "scheme": config["scheme"],
            "null_count": config["null_count"],
            "size_profile": config["size_profile"],
            "external_profile": config["external_profile"],
            "sizes": config["sizes"],
            "assignments": assignments,
            "selector": selected_selector,
            "selector_scores": selector_scores,
            "objective": objective,
            "selection_score": selection_score,
            "optimizer": "torch-batched-anneal-v0.2-development",
        }
        if best is None or (row["selection_score"], json.dumps(assignments, sort_keys=True)) < (
            best["selection_score"], json.dumps(best["assignments"], sort_keys=True)
        ):
            best = row
    if best is None:
        raise RuntimeError("no candidate model was fitted")
    return best


def validate(module, repo: Path, device: torch.device) -> dict[str, float]:
    rng = np.random.default_rng(20260714)
    rows = []
    for index in range(32):
        n_cells = 24
        units = 13
        pair = rng.poisson(3.0, size=(n_cells, n_cells)).astype(float)
        uni = rng.poisson(20.0, size=n_cells).astype(float)
        transition = rng.random((units, units))
        transition /= transition.sum(axis=1, keepdims=True)
        stationary = rng.random(units)
        stationary /= stationary.sum()
        assignment = rng.integers(0, units, size=n_cells).tolist()
        chain = {
            "pair": pair, "uni": uni, "transition": transition,
            "stationary": stationary, "base": assignment,
            "chain_seed": index + 1,
        }
        pair_t = torch.as_tensor(pair[None], dtype=torch.float64, device=device)
        uni_t = torch.as_tensor(uni[None], dtype=torch.float64, device=device)
        assignment_t = torch.as_tensor([assignment], dtype=torch.long, device=device)
        log_t = torch.as_tensor(np.log2(transition)[None], dtype=torch.float64, device=device)
        log_s = torch.as_tensor(np.log2(stationary)[None], dtype=torch.float64, device=device)
        gpu = float(objective_batch(pair_t, uni_t, assignment_t, log_t, log_s)[0].cpu())
        cpu = module.objective_assignment(
            pair, uni, assignment, np.log2(transition), np.log2(stationary)
        )
        rows.append(abs(cpu - gpu))

    record_path, ci_path = module.locate_data(repo)
    records = pickle.load(record_path.open("rb"))
    ci = pickle.load(ci_path.open("rb"))
    registry = module.build_surface_registry(records)
    external = module.build_external_models(ci)
    scenario = module.scenario_for_index(1)
    events, keys, _ = module.generate_cipher_trial(
        registry, external, 616161,
        str(scenario["key_scheme"]), int(scenario["null_count"]),
        str(scenario["size_profile"]), str(scenario["external_profile"]),
        str(scenario["selection_policy"]), str(scenario["selector"]),
    )
    train = [event for event in events if not event.test]
    transition, stationary = module.extend_external_model(
        external, str(scenario["external_profile"]), int(scenario["null_count"])
    )
    cpu_total, cpu_external, cpu_emission = module.cipher_train_nll(
        train, keys, str(scenario["key_scheme"]), transition, stationary,
        registry, str(scenario["selector"]),
    )
    pairs, unis, starts = sufficient_statistics(
        module, train, str(scenario["key_scheme"]), len(registry.cells)
    )
    fast_external, fast_emission = fast_cost_components(
        module, keys, pairs, unis, starts, transition, stationary
    )
    token_bits = module.selector_nll(train, registry, str(scenario["selector"]))
    fast_total = fast_external + fast_emission + token_bits
    report = {
        "objective_max_abs_error": max(rows),
        "external_bits_abs_error": abs(cpu_external - fast_external),
        "emission_bits_abs_error": abs(cpu_emission - fast_emission),
        "train_total_abs_error": abs(cpu_total - fast_total),
    }
    if max(report.values()) > 1e-7:
        raise RuntimeError(f"GPU validation failed: {report}")
    return report


def worker(repo_text: str, rank: int, tasks: list[dict[str, Any]], steps: int, restarts: int, shard_path: str):
    repo = Path(repo_text)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.use_deterministic_algorithms(True)
    module = load_impl(repo)
    module.candidate_fit = functools.partial(gpu_candidate_fit, module, device)
    record_path, ci_path = module.locate_data(repo)
    registry = module.build_surface_registry(pickle.load(record_path.open("rb")))
    external = module.build_external_models(pickle.load(ci_path.open("rb")))
    results = []
    started = time.time()
    for position, task in enumerate(tasks, start=1):
        kind = task["kind"]
        index = int(task["index"])
        seed = int(task["seed"])
        if kind == "positive":
            scenario = module.scenario_for_index(index)
            events, keys, sizes = module.generate_cipher_trial(
                registry, external, seed,
                str(scenario["key_scheme"]), int(scenario["null_count"]),
                str(scenario["size_profile"]), str(scenario["external_profile"]),
                str(scenario["selection_policy"]), str(scenario["selector"]),
            )
            truth = {**scenario, "keys": keys, "class_sizes": list(sizes)}
            result = module.score_trial(
                events, registry, external, seed ^ 0xC1F3, steps, restarts, truth
            )
            result.update({"trial_type": "positive", "trial_index": index, "seed": seed})
        else:
            family = module.CONTROL_FAMILIES[index % len(module.CONTROL_FAMILIES)]
            selector = module.SELECTORS[(index // len(module.CONTROL_FAMILIES)) % len(module.SELECTORS)]
            events = module.generate_control_trial(registry, external, seed, family, selector)
            result = module.score_trial(
                events, registry, external, seed ^ 0xF00D, steps, restarts, None
            )
            result.update({
                "trial_type": "control", "control_family": family,
                "true_selector": selector, "trial_index": index, "seed": seed,
                "false_positive": bool(result["cipher_selected"]),
            })
        results.append(result)
        print(
            f"GPU_PROGRESS rank={rank} completed={position}/{len(tasks)} "
            f"kind={kind} index={index} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    Path(shard_path).write_text(json.dumps(results, sort_keys=True) + "\n")


def run_multi_gpu(repo: Path, positives: int, controls: int, steps: int, restarts: int, seed: int, output: Path):
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("no CUDA devices are visible")
    gpu_count = min(8, count)
    tasks = [
        {"kind": "positive", "index": i, "seed": seed + 100000 + i * 7919}
        for i in range(positives)
    ] + [
        {"kind": "control", "index": i, "seed": seed + 900000 + i * 104729}
        for i in range(controls)
    ]
    shards = [[] for _ in range(gpu_count)]
    for index, task in enumerate(tasks):
        shards[index % gpu_count].append(task)

    ctx = mp.get_context("spawn")
    processes = []
    shard_paths = []
    for rank in range(gpu_count):
        shard_path = f"/tmp/morpholocal_gpu_shard_{rank}.json"
        shard_paths.append(shard_path)
        process = ctx.Process(
            target=worker,
            args=(str(repo), rank, shards[rank], steps, restarts, shard_path),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU worker failed with exit code {process.exitcode}")

    results = []
    for shard_path in shard_paths:
        results.extend(json.loads(Path(shard_path).read_text()))
    results.sort(key=lambda row: (row["trial_type"], int(row["trial_index"])))
    module = load_impl(repo)
    summary = module.aggregate(results)
    payload = {
        "programme": module.VERSION,
        "development_accelerator": "torch-batched-anneal-v0.2",
        "formal_seed": seed,
        "data_hashes": module.EXPECTED_HASHES,
        "parameters": {
            "positives": positives, "controls": controls,
            "anneal_steps": steps, "anneal_restarts": restarts,
            "gpu_count": gpu_count,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
            "torch_version": torch.__version__,
            "candidate_grid_unchanged": True,
            "optimizer_changed_from_scalar_cpu": True,
        },
        "summary": summary,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--positives", type=int, default=96)
    parser.add_argument("--controls", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, default=Path("GPU_DEVELOPMENT_FACTORIAL.json"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    module = load_impl(args.repo)
    device = torch.device("cuda:0")
    validation = validate(module, args.repo, device)
    print("GPU_VALIDATION", json.dumps(validation, sort_keys=True), flush=True)
    if not args.validate_only:
        run_multi_gpu(
            args.repo, args.positives, args.controls,
            args.steps, args.restarts, args.seed, args.output,
        )


if __name__ == "__main__":
    main()
