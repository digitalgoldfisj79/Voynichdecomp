#!/usr/bin/env python3
"""Synthetic-trained graph decoder for morpholocal calibration v0.3."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
FAST_PATH = HERE / "tournament_fast.py"
spec = importlib.util.spec_from_file_location("v03_fast_neural", FAST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import tournament_fast.py")
fast = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fast
spec.loader.exec_module(fast)
base = fast.base

MAX_UNITS = 13
GLOBAL_DIM = MAX_UNITS * MAX_UNITS + MAX_UNITS + 6
_MODEL_CACHE: dict[str, tuple[nn.Module, dict[str, Any]]] = {}


def normalized_rows(pair: np.ndarray) -> np.ndarray:
    return pair / np.clip(pair.sum(axis=1, keepdims=True), 1e-12, None)


def feature_arrays(pair, uni, transition, stationary, null_count, size_profile, external_profile):
    pair = np.asarray(pair, dtype=np.float32)
    uni = np.asarray(uni, dtype=np.float32)
    out = normalized_rows(pair)
    inn = normalized_rows(pair.T)
    row_sum = pair.sum(axis=1)
    col_sum = pair.sum(axis=0)
    row_entropy = -(out * np.log2(np.clip(out, 1e-12, None))).sum(axis=1)
    col_entropy = -(inn * np.log2(np.clip(inn, 1e-12, None))).sum(axis=1)
    node = np.stack([
        np.log1p(uni), np.log1p(row_sum), np.log1p(col_sum), row_entropy, col_entropy,
    ], axis=1).astype(np.float32)
    node = (node - node.mean(axis=0, keepdims=True)) / np.clip(node.std(axis=0, keepdims=True), 1e-5, None)

    padded_t = np.zeros((MAX_UNITS, MAX_UNITS), dtype=np.float32)
    padded_s = np.zeros(MAX_UNITS, dtype=np.float32)
    n = transition.shape[0]
    padded_t[:n, :n] = transition
    padded_s[:n] = stationary
    extra = np.array([
        float(null_count > 0), float(size_profile == "unequal"),
        float(external_profile == "word_heavy"), float(external_profile == "balanced"),
        float(external_profile == "letter_heavy"), float(n) / MAX_UNITS,
    ], dtype=np.float32)
    global_features = np.concatenate([padded_t.reshape(-1), padded_s, extra]).astype(np.float32)
    return node, out.astype(np.float32), inn.astype(np.float32), global_features


class GraphKeyNet(nn.Module):
    def __init__(self, hidden=160, layers=5, dropout=0.05):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.global_net = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            ) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.output = nn.Linear(hidden, MAX_UNITS)

    def forward(self, node, out, inn, global_features):
        h = self.node(node)
        g = self.global_net(global_features)[:, None, :]
        for layer, norm in zip(self.layers, self.norms):
            message_out = torch.bmm(out, h)
            message_in = torch.bmm(inn, h)
            update = layer(torch.cat([h, message_out, message_in, g.expand_as(h)], dim=-1))
            h = norm(h + update)
        return self.output(h)


def sections_for_registry(registry):
    sections = sorted({section for section, _ in registry.context_position_counts})
    if not sections:
        raise RuntimeError("surface registry has no sections")
    return sections


def simulate_statistics(module, registry, external, seed, scenario, token_count):
    rng = random.Random(seed)
    n_cells = len(registry.cells)
    sizes = module.class_sizes(scenario["size_profile"], n_cells - scenario["null_count"])
    labels = ("GLOBAL",) if scenario["key_scheme"] == "global" else ("A", "B")
    keys = {label: module.make_key(rng, sizes, scenario["null_count"], n_cells) for label in labels}
    transition, stationary = module.extend_external_model(
        external, scenario["external_profile"], scenario["null_count"]
    )
    sections = sections_for_registry(registry)
    by_currier = defaultdict(list)
    for section in sections:
        by_currier[module.currier_for_section(section)].append(section)
    line_lengths = tuple(int(x) for x in registry.line_lengths if int(x) > 0) or (8, 9, 10)
    pairs = {label: np.zeros((n_cells, n_cells), dtype=np.float32) for label in labels}
    unis = {label: np.zeros(n_cells, dtype=np.float32) for label in labels}
    cyclic_state = defaultdict(int)
    produced = 0
    line_id = 0
    while produced < token_count:
        if scenario["key_scheme"] == "global":
            currier = "A" if line_id % 2 == 0 else "B"
            section_pool = by_currier.get(currier) or sections
            label = "GLOBAL"
        else:
            label = "A" if line_id % 2 == 0 else "B"
            currier = label
            section_pool = by_currier.get(label) or sections
        section = section_pool[rng.randrange(len(section_pool))]
        assignment = keys[label]
        line_length = min(line_lengths[rng.randrange(len(line_lengths))], token_count - produced)
        previous_unit = None
        previous_cell = None
        line_cells = []
        for position_index in range(line_length):
            position = "FIRST" if position_index == 0 else "LAST" if position_index == line_length - 1 else "MID"
            if previous_unit is None:
                unit = module.weighted_choice(rng, tuple(range(len(stationary))), stationary)
            else:
                unit = module.weighted_choice(rng, tuple(range(transition.shape[1])), transition[previous_unit])
            candidates = [c for c, u in enumerate(assignment) if u == unit]
            policy = scenario["selection_policy"]
            if policy == "iid_uniform":
                cell = candidates[rng.randrange(len(candidates))]
            elif policy == "frequency_weighted":
                cell = module.weighted_choice(
                    rng, candidates,
                    [module.context_weight(registry, c, section, position) for c in candidates],
                )
            elif policy == "cyclic":
                slot = cyclic_state[(label, unit)] % len(candidates)
                cell = sorted(candidates)[slot]
                cyclic_state[(label, unit)] += 1
            elif policy == "sticky_line_reset":
                if previous_unit == unit and previous_cell in candidates and rng.random() < 0.75:
                    cell = int(previous_cell)
                else:
                    cell = module.weighted_choice(
                        rng, candidates,
                        [module.context_weight(registry, c, section, position) for c in candidates],
                    )
            else:
                raise ValueError(policy)
            unis[label][cell] += 1
            if line_cells:
                pairs[label][line_cells[-1], cell] += 1
            line_cells.append(cell)
            previous_unit, previous_cell = unit, cell
            produced += 1
        line_id += 1
    return pairs, unis, keys, sizes, transition, stationary


def generate_dataset(repo: Path, trials: int, seed: int, min_tokens: int, max_tokens: int):
    _, module = base.load_v02(repo)
    record_path, ci_path = module.locate_data(repo)
    registry = module.build_surface_registry(pickle.load(record_path.open("rb")))
    external = module.build_external_models(pickle.load(ci_path.open("rb")))
    nodes, outs, inns, globals_, targets, counts = [], [], [], [], [], []
    rng = random.Random(seed)
    for trial in range(trials):
        scenario = module.scenario_for_index(rng.randrange(96))
        token_count = rng.randint(min_tokens, max_tokens)
        pairs, unis, keys, sizes, transition, stationary = simulate_statistics(
            module, registry, external, seed + 1000003 * trial, scenario, token_count
        )
        for label in sorted(keys):
            node, out, inn, global_features = feature_arrays(
                pairs[label], unis[label], transition, stationary,
                scenario["null_count"], scenario["size_profile"], scenario["external_profile"],
            )
            target = np.asarray(keys[label], dtype=np.int64)
            unit_counts = np.bincount(target, minlength=MAX_UNITS).astype(np.int64)
            perm_rng = np.random.default_rng(seed + 7919 * trial + (0 if label == "GLOBAL" else ord(label)))
            perm = perm_rng.permutation(len(target))
            nodes.append(node[perm])
            outs.append(out[np.ix_(perm, perm)])
            inns.append(inn[np.ix_(perm, perm)])
            globals_.append(global_features)
            targets.append(target[perm])
            counts.append(unit_counts)
        if (trial + 1) % 250 == 0:
            print(f"NEURAL_DATA trial={trial+1}/{trials} examples={len(targets)}", flush=True)
    return (
        np.stack(nodes), np.stack(outs), np.stack(inns), np.stack(globals_),
        np.stack(targets), np.stack(counts),
    )


def constrained_assignment(logits: np.ndarray, counts: np.ndarray) -> tuple[int, ...]:
    slots = []
    for unit, count in enumerate(counts):
        slots.extend([unit] * int(count))
    if len(slots) != logits.shape[0]:
        raise RuntimeError(f"class counts sum to {len(slots)}, expected {logits.shape[0]}")
    score = logits[:, np.asarray(slots, dtype=np.int64)]
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-score)
        mapping = np.empty(logits.shape[0], dtype=np.int64)
        mapping[rows] = np.asarray(slots, dtype=np.int64)[cols]
    except Exception:
        mapping = np.full(logits.shape[0], -1, dtype=np.int64)
        available = list(range(len(slots)))
        choices = sorted(
            ((float(score[cell, slot]), cell, slot) for cell in range(score.shape[0]) for slot in range(score.shape[1])),
            reverse=True,
        )
        used_cells = set()
        for _, cell, slot_index in choices:
            if cell in used_cells or slot_index not in available:
                continue
            mapping[cell] = slots[slot_index]
            used_cells.add(cell)
            available.remove(slot_index)
            if not available:
                break
    return tuple(int(x) for x in mapping)


def train_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=base.DEFAULT_REPO)
    ap.add_argument("--trials", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=330033)
    ap.add_argument("--min-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    arrays = generate_dataset(args.repo, args.trials, args.seed, args.min_tokens, args.max_tokens)
    node, out, inn, global_features, target, counts = arrays
    generator = np.random.default_rng(args.seed ^ 0x515151)
    order = generator.permutation(len(target))
    split = int(round(len(order) * 0.9))
    train_idx, val_idx = order[:split], order[split:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphKeyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    tensors = [torch.from_numpy(x) for x in (node, out, inn, global_features, target, counts)]
    train_ds = TensorDataset(*(x[train_idx] for x in tensors))
    val_ds = TensorDataset(*(x[val_idx] for x in tensors))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_cells = 0
        for node_b, out_b, inn_b, global_b, target_b, _ in train_loader:
            node_b, out_b, inn_b = node_b.to(device), out_b.to(device), inn_b.to(device)
            global_b, target_b = global_b.to(device), target_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(node_b, out_b, inn_b, global_b)
            loss = loss_fn(logits.reshape(-1, MAX_UNITS), target_b.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss) * target_b.numel()
            train_cells += target_b.numel()
        model.eval()
        val_loss = 0.0
        val_cells = 0
        raw_correct = 0
        constrained_correct = 0
        examples = 0
        with torch.no_grad():
            for node_b, out_b, inn_b, global_b, target_b, counts_b in val_loader:
                logits = model(node_b.to(device), out_b.to(device), inn_b.to(device), global_b.to(device))
                target_dev = target_b.to(device)
                loss = loss_fn(logits.reshape(-1, MAX_UNITS), target_dev.reshape(-1))
                val_loss += float(loss) * target_b.numel()
                val_cells += target_b.numel()
                raw_correct += int((logits.argmax(-1).cpu() == target_b).sum())
                for logit_row, target_row, count_row in zip(logits.cpu().numpy(), target_b.numpy(), counts_b.numpy()):
                    pred = np.asarray(constrained_assignment(logit_row, count_row))
                    constrained_correct += int((pred == target_row).sum())
                    examples += 1
        scheduler.step()
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, train_cells),
            "val_loss": val_loss / max(1, val_cells),
            "raw_accuracy": raw_correct / max(1, val_cells),
            "constrained_accuracy": constrained_correct / max(1, examples * 24),
        }
        history.append(metrics)
        print("NEURAL_EPOCH", json.dumps(metrics, sort_keys=True), flush=True)
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "format": "morpholocal-v0.3-graph-key-net",
        "model_config": {"hidden": 160, "layers": 5, "dropout": 0.05},
        "training": {
            "trials": args.trials, "examples": len(target), "seed": args.seed,
            "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
            "epochs": args.epochs, "batch_size": args.batch_size,
        },
        "history": history, "state_dict": best_state,
    }
    torch.save(payload, args.output)
    print("NEURAL_TRAIN_DONE", json.dumps({
        "output": str(args.output), "best_val_loss": best_val,
        "final": history[-1], "examples": len(target), "device": str(device),
    }, sort_keys=True), flush=True)


def load_model(path: str):
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = GraphKeyNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _MODEL_CACHE[path] = (model, payload)
    return model, payload


def neural_solver(module, events, label, pair, uni, sizes, null_count, transition, stationary, registry, seed, cfg):
    model_path = str(cfg["model_path"])
    model, payload = load_model(model_path)
    node, out, inn, global_features = feature_arrays(
        pair, uni, transition, stationary, null_count,
        cfg.get("_size_profile", "balanced"), cfg.get("_external_profile", "balanced"),
    )
    with torch.no_grad():
        logits = model(
            torch.from_numpy(node[None]), torch.from_numpy(out[None]),
            torch.from_numpy(inn[None]), torch.from_numpy(global_features[None]),
        )[0].numpy()
    _, count_dict = base.mapping_counts(module, uni, sizes, null_count, stationary)
    count_array = np.zeros(MAX_UNITS, dtype=np.int64)
    for unit, count in count_dict.items():
        count_array[int(unit)] = int(count)
    mapping = constrained_assignment(logits, count_array)
    policy_scores = {p: base.label_policy_nll(module, events, mapping, p, registry, label) for p in base.POLICIES}
    policy = min(policy_scores, key=lambda p: (policy_scores[p], p))
    mapping, _ = base.refine_swaps(
        module, events, label, pair, uni, mapping, transition, stationary,
        policy, registry, seed ^ 0x4E455552414C,
        int(cfg.get("refine_iterations", 250)), float(cfg.get("refine_temperature", 0.5)),
    )
    return mapping, {
        "solver": "neural", "policy_hint": policy,
        "training_examples": payload["training"]["examples"],
    }


def patched_fit_candidate(module, gr, train, registry, external, seed, solver_name, cfg):
    selected_selector, selector_scores = module.select_selector(train, registry)
    best = None
    for scheme in module.KEY_SCHEMES:
        pairs, unis, starts = gr.sufficient_statistics(module, train, scheme, len(registry.cells))
        for null_count in module.NULL_COUNTS:
            for size_profile in module.SIZE_PROFILES:
                sizes = module.class_sizes(size_profile, len(registry.cells) - null_count)
                for external_profile in module.PROFILE_NAMES:
                    transition, stationary = module.extend_external_model(external, external_profile, null_count)
                    local_cfg = dict(cfg)
                    local_cfg["_size_profile"] = size_profile
                    local_cfg["_external_profile"] = external_profile
                    assignments, solver_meta = {}, {}
                    for key_index, label in enumerate(sorted(pairs)):
                        mapping, meta = neural_solver(
                            module, base.label_events(module, train, scheme, label), label,
                            pairs[label], unis[label], sizes, null_count, transition,
                            stationary, registry,
                            seed ^ (key_index * 0x9E3779B1) ^ module.stable_seed(
                                "neural", scheme, null_count, size_profile, external_profile
                            ), local_cfg,
                        )
                        assignments[label] = tuple(mapping)
                        solver_meta[label] = meta
                    policy, policy_scores = base.infer_policy(module, train, assignments, scheme, registry)
                    external_bits, _ = gr.fast_cost_components(
                        module, assignments, pairs, unis, starts, transition, stationary
                    )
                    report = gr.structural_report(module, sizes, null_count, len(assignments), external_profile)
                    selection_score = (
                        report.partition_bits * len(assignments) + report.topology_bits
                        + report.external_model_index_bits + external_bits
                        + policy_scores[policy] + math.log2(len(base.POLICIES))
                        + module.selector_nll(train, registry, selected_selector)
                    )
                    row = {
                        "scheme": scheme, "null_count": null_count,
                        "size_profile": size_profile, "external_profile": external_profile,
                        "sizes": sizes, "assignments": assignments,
                        "selector": selected_selector, "selector_scores": selector_scores,
                        "policy": policy, "policy_scores": policy_scores,
                        "selection_score": float(selection_score),
                        "solver": "neural", "solver_meta": solver_meta,
                    }
                    tie = json.dumps(assignments, sort_keys=True)
                    if best is None or (selection_score, tie) < (
                        best["selection_score"], json.dumps(best["assignments"], sort_keys=True)
                    ):
                        best = row
    if best is None:
        raise RuntimeError("neural candidate fit failed")
    return best


def run_main(argv):
    base.SOLVERS["neural"] = neural_solver
    base.fit_candidate = patched_fit_candidate
    sys.argv = [sys.argv[0], *argv]
    base.main()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "run"}:
        raise SystemExit("usage: neural_runner.py {train|run} ...")
    if sys.argv[1] == "train":
        train_main(sys.argv[2:])
    else:
        run_main(sys.argv[2:])


if __name__ == "__main__":
    main()
