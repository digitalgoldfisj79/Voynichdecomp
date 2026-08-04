#!/usr/bin/env python3
"""Compare a pixel HTR baseline with a pixel + frozen-DINOv3 fusion model."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import train_pilot as base
import train_pilot_v2 as corrected_sampler

base.collect_samples = corrected_sampler.collect_samples_balanced


class PixelSequence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input is ink-positive: 0 is parchment, 1 is black ink.
        x = self.net(images)  # B, 128, 8, 64
        x = x.mean(dim=2).transpose(1, 2)  # B, 64, 128
        return x


class HybridCTC(nn.Module):
    def __init__(self, dino_dim: int, vocab_size: int, use_dino: bool) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.pixel = PixelSequence()
        if use_dino:
            self.pixel_proj = nn.Linear(128, 160)
            self.dino_proj = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 160), nn.GELU())
            sequence_dim = 320
        else:
            self.pixel_proj = nn.Linear(128, 320)
            self.dino_proj = None
            sequence_dim = 320
        self.sequence_norm = nn.LayerNorm(sequence_dim)
        self.rnn = nn.GRU(
            sequence_dim,
            192,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )
        self.classifier = nn.Linear(384, vocab_size)
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[0] = -2.0

    def forward(
        self,
        images: torch.Tensor,
        dino: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pixel = self.pixel_proj(self.pixel(images))
        if self.use_dino:
            x = torch.cat([pixel, self.dino_proj(dino)], dim=-1)
        else:
            x = pixel
        x = self.sequence_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        return self.classifier(x)


def image_batch(samples: Sequence[base.Sample], indices: Sequence[int], device: torch.device) -> torch.Tensor:
    arr = np.stack([samples[i].image for i in indices])
    x = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(1) / 255.0
    return 1.0 - x


def encode_targets(texts: Sequence[str], char_to_id: dict[str, int], device: torch.device):
    encoded = [torch.tensor([char_to_id[c] for c in text], dtype=torch.long) for text in texts]
    lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
    return torch.cat(encoded).to(device), lengths


def greedy_decode(ids, id_to_char: dict[int, str]) -> str:
    out = []
    previous = None
    for value in ids:
        value = int(value)
        if value != 0 and value != previous:
            out.append(id_to_char.get(value, "�"))
        previous = value
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def evaluate(model, features, lengths, samples, id_to_char, device, batch_size):
    model.eval()
    edits = chars = ns_edits = ns_chars = exact = 0
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            idx = list(range(start, min(len(samples), start + batch_size)))
            images = image_batch(samples, idx, device)
            dino = features[idx].to(device=device, dtype=torch.float32)
            lens = lengths[idx].to(device)
            logits = model(images, dino, lens)
            best = logits.argmax(-1).cpu().numpy()
            for j, sample_idx in enumerate(idx):
                sample = samples[sample_idx]
                pred = greedy_decode(best[j, : int(lengths[sample_idx])], id_to_char)
                edits += edit_distance(sample.text, pred)
                chars += len(sample.text)
                ref_ns, pred_ns = sample.text.replace(" ", ""), pred.replace(" ", "")
                ns_edits += edit_distance(ref_ns, pred_ns)
                ns_chars += len(ref_ns)
                exact += pred == sample.text
                rows.append({"shelfmark": sample.shelfmark, "reference": sample.text, "prediction": pred})
    return {
        "cer": edits / max(1, chars),
        "cer_no_spaces": ns_edits / max(1, ns_chars),
        "exact_line_accuracy": exact / max(1, len(samples)),
        "lines": len(samples),
    }, rows


def train_arm(
    name,
    use_dino,
    features,
    lengths,
    samples,
    char_to_id,
    id_to_char,
    device,
    epochs,
    batch_size,
    seed,
):
    base.seed_everything(seed)
    model = HybridCTC(features["train"].shape[-1], len(char_to_id) + 1, use_dino).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_dev = float("inf")
    patience = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(samples["train"]), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            images = image_batch(samples["train"], idx, device)
            dino = features["train"][idx].to(device=device, dtype=torch.float32)
            lens = lengths["train"][idx].to(device)
            texts = [samples["train"][i].text for i in idx]
            targets, target_lengths = encode_targets(texts, char_to_id, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, dino, lens)
            log_probs = logits.log_softmax(-1).transpose(0, 1)
            loss = ctc(log_probs, targets, lens, target_lengths)
            # Temporary anti-collapse regularizer. It is identical in both arms
            # and decays to zero after epoch 8.
            weight = max(0.0, 0.08 * (1.0 - (epoch - 1) / 8.0))
            if weight:
                blank_prob = logits.softmax(-1)[..., 0]
                valid = torch.arange(logits.shape[1], device=device)[None, :] < lens[:, None]
                loss = loss + weight * blank_prob[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev, _ = evaluate(
            model, features["dev"], lengths["dev"], samples["dev"], id_to_char, device, batch_size
        )
        row = {"arm": name, "epoch": epoch, "loss": float(np.mean(losses)), **dev}
        history.append(row)
        print("EPOCH", json.dumps(row, ensure_ascii=False), flush=True)
        if dev["cer"] < best_dev - 1e-4:
            best_dev = dev["cer"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break

    if best_state is None:
        raise RuntimeError(f"{name}: no best state")
    model.load_state_dict(best_state)
    dev, dev_rows = evaluate(model, features["dev"], lengths["dev"], samples["dev"], id_to_char, device, batch_size)
    test, test_rows = evaluate(model, features["test"], lengths["test"], samples["test"], id_to_char, device, batch_size)

    blank = base.Sample("blank", "BLANK", "", np.full((base.HEIGHT, base.WIDTH), 255, np.uint8), base.TIME_STEPS, 0, "", "")
    blank_dino = torch.zeros((1, base.TIME_STEPS, features["train"].shape[-1]), dtype=torch.float16)
    blank_lengths = torch.tensor([base.TIME_STEPS], dtype=torch.long)
    blank_logits = model(
        image_batch([blank], [0], device),
        blank_dino.to(device=device, dtype=torch.float32),
        blank_lengths.to(device),
    )
    blank_prediction = greedy_decode(blank_logits.argmax(-1)[0].detach().cpu().tolist(), id_to_char)
    return model, {"dev": dev, "test": test, "blank_prediction": blank_prediction, "history": history}, dev_rows, test_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("hybrid_results"))
    p.add_argument("--train", type=int, default=512)
    p.add_argument("--dev", type=int, default=96)
    p.add_argument("--test", type=int, default=96)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--max-scan", type=int, default=60000)
    p.add_argument("--seed", type=int, default=20260804)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from huggingface_hub import dataset_info, model_info
    from transformers import AutoImageProcessor, AutoModel

    data_meta = dataset_info(base.DATA_REPO)
    dino_meta = model_info(base.DINO_REPO)
    samples, manifest = base.collect_samples(args.train, args.dev, args.test, args.seed, 48, args.max_scan)
    manifest.update({"dataset_revision": data_meta.sha, "dino_revision": dino_meta.sha})
    (args.output / "DATA_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DATA", manifest["counts"], manifest["shelfmark_counts"], flush=True)

    char_to_id, id_to_char = base.make_vocab(samples)
    processor = AutoImageProcessor.from_pretrained(base.DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN"))
    encoder = AutoModel.from_pretrained(base.DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN")).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    features, lengths = {}, {}
    for split in ("train", "dev", "test"):
        features[split], lengths[split] = base.extract_features(
            encoder, processor, samples[split], device, args.feature_batch_size
        )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    arm_results = {}
    predictions = {}
    states = {}
    for arm, use_dino in (("CNN_ONLY", False), ("CNN_DINOV3", True)):
        model, result, dev_rows, test_rows = train_arm(
            arm, use_dino, features, lengths, samples, char_to_id, id_to_char,
            device, args.epochs, args.batch_size, args.seed,
        )
        arm_results[arm] = result
        predictions[arm] = {"dev": dev_rows, "test": test_rows}
        states[arm] = {k: v.cpu() for k, v in model.state_dict().items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cnn_cer = arm_results["CNN_ONLY"]["test"]["cer"]
    fused_cer = arm_results["CNN_DINOV3"]["test"]["cer"]
    fused_dev = arm_results["CNN_DINOV3"]["dev"]["cer"]
    blank_nonspace = len(arm_results["CNN_DINOV3"]["blank_prediction"].replace(" ", ""))
    pilot_pass = fused_dev < 0.90 and fused_cer <= cnn_cer - 0.02 and blank_nonspace <= 2

    result = {
        "status": "COMPLETE",
        "verdict": "HYBRID_DINOV3_CATMUS_PASS" if pilot_pass else "HYBRID_DINOV3_CATMUS_FAIL",
        "pilot_pass": pilot_pass,
        "dataset": {"repo": base.DATA_REPO, "revision": data_meta.sha, **manifest["counts"]},
        "shelfmark_counts": manifest["shelfmark_counts"],
        "dino": {"repo": base.DINO_REPO, "revision": dino_meta.sha, "frozen": True},
        "arms": arm_results,
        "fused_minus_cnn_test_cer": fused_cer - cnn_cer,
        "device": str(device),
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "PREDICTIONS.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"states": states, "vocab": {"char_to_id": char_to_id}, "result": result}, args.output / "hybrid_models.pt")

    md = [
        "# Hybrid pixel + DINOv3 CATMuS result",
        "",
        f"- Verdict: **{result['verdict']}**",
        f"- Train/dev/test: **{args.train}/{args.dev}/{args.test}**",
        f"- Shelfmarks: **{manifest['shelfmark_counts']}**",
        "",
        "| Arm | Dev CER | Test CER | Test CER no spaces | Blank prediction |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ("CNN_ONLY", "CNN_DINOV3"):
        r = arm_results[arm]
        md.append(f"| {arm} | {r['dev']['cer']:.4f} | {r['test']['cer']:.4f} | {r['test']['cer_no_spaces']:.4f} | `{r['blank_prediction']}` |")
    md += [
        "",
        f"Fused minus CNN-only test CER: **{result['fused_minus_cnn_test_cer']:+.4f}**.",
        "",
        "No dictionary, language model, abbreviation expansion or word correction was used.",
    ]
    (args.output / "RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("HYBRID_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
