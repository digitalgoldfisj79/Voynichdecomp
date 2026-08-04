#!/usr/bin/env python3
"""Remove the scale-run host-memory spike without changing model semantics."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXPECTED_INPUT_SHA256 = "cdb97e78005ff8b00006598df90d68fbdad6fa6dae554df7fa034f51779b23b4"

OLD_EXTRACT = '''def extract_dino_features(encoder, processor, views: np.ndarray, lengths: torch.Tensor,
                          device: torch.device, batch_size: int) -> torch.Tensor:
    n = views.shape[0]
    all_features = []
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    gh, gw = HEIGHT // PATCH, WIDTH // PATCH
    flat = views.reshape(n * VIEWS, HEIGHT, WIDTH, 3)
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        pixels = torch.from_numpy(batch).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        pixels = (pixels - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = encoder(pixel_values=pixels, interpolate_pos_encoding=True)
            tokens = output.last_hidden_state[:, -(gh * gw):, :].reshape(len(batch), gh, gw, -1)
            seq = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        all_features.append(seq.detach().cpu().to(torch.float16))
        if (start // batch_size) % 25 == 0:
            print("DINO_FEATURES", min(start + len(batch), len(flat)), "/", len(flat), flush=True)
    return torch.cat(all_features, 0).reshape(n, VIEWS, TIME_STEPS, -1)
'''

NEW_EXTRACT = '''def extract_dino_features(encoder, processor, views: np.ndarray, lengths: torch.Tensor,
                          device: torch.device, batch_size: int) -> torch.Tensor:
    n = views.shape[0]
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    gh, gw = HEIGHT // PATCH, WIDTH // PATCH
    flat = views.reshape(n * VIEWS, HEIGHT, WIDTH, 3)
    feature_store = None
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        pixels = torch.from_numpy(batch).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        pixels = (pixels - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = encoder(pixel_values=pixels, interpolate_pos_encoding=True)
            tokens = output.last_hidden_state[:, -(gh * gw):, :].reshape(len(batch), gh, gw, -1)
            seq = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        seq_cpu = seq.detach().cpu().to(torch.float16)
        if feature_store is None:
            feature_store = torch.empty(
                (len(flat), seq_cpu.shape[1], seq_cpu.shape[2]), dtype=torch.float16)
        feature_store[start:start + len(batch)].copy_(seq_cpu)
        del pixels, output, tokens, seq, seq_cpu
        if (start // batch_size) % 25 == 0:
            print("DINO_FEATURES", min(start + len(batch), len(flat)), "/", len(flat), flush=True)
    if feature_store is None:
        return torch.empty((n, VIEWS, TIME_STEPS, 0), dtype=torch.float16)
    return feature_store.reshape(n, VIEWS, TIME_STEPS, -1)
'''

OLD_VIEWS = '''    views = make_all_views(samples, args.seed)
    lengths = {k: torch.tensor([s.valid_steps for s in v], dtype=torch.long)
               for k, v in samples.items()}
'''

NEW_VIEWS = '''    views = make_all_views(samples, args.seed)
    # The rendered three-view tensor is now authoritative; release the source
    # line arrays before allocating the multi-gigabyte DINO feature store.
    for rows in samples.values():
        for sample in rows:
            sample.image = np.empty((0,), dtype=np.uint8)
    import gc
    gc.collect()
    lengths = {k: torch.tensor([s.valid_steps for s in v], dtype=torch.long)
               for k, v in samples.items()}
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_memory_safe_features.py TRAIN_V02_PY")
    path = Path(sys.argv[1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"unexpected input SHA-256: {digest}")
    text = path.read_text(encoding="utf-8")
    if text.count(OLD_EXTRACT) != 1:
        raise RuntimeError("feature-extractor patch site mismatch")
    if text.count(OLD_VIEWS) != 1:
        raise RuntimeError("source-image release patch site mismatch")
    text = text.replace(OLD_EXTRACT, NEW_EXTRACT, 1).replace(OLD_VIEWS, NEW_VIEWS, 1)
    path.write_text(text, encoding="utf-8")
    print("MEMORY_PATCH_OK", hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
