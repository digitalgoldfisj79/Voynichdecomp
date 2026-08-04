#!/usr/bin/env python3
"""Frozen-DINOv3 + CATMuS grapheme CTC feasibility pilot.

The script constructs a new manuscript-disjoint split from CATMuS shelfmarks,
extracts frozen DINOv3 patch features, trains only a small 2D-to-1D BiGRU CTC
head, and reports untrained-versus-trained character error rates. It never
uses a dictionary or language model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Iterable, Sequence
import unicodedata

import numpy as np
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DINO_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DATA_REPO = "CATMuS/medieval"
HEIGHT = 128
WIDTH = 1024
PATCH = 16
TIME_STEPS = WIDTH // PATCH


@dataclass
class Sample:
    split: str
    shelfmark: str
    text: str
    image: np.ndarray  # grayscale uint8, HEIGHT x WIDTH
    valid_steps: int
    century: int
    script_type: str
    language: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ctc_required_steps(text: str) -> int:
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def split_for_shelfmark(shelfmark: str) -> str:
    value = int(hashlib.sha256(shelfmark.encode("utf-8")).hexdigest()[:12], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "dev"
    return "test"


def prepare_line(image: Image.Image) -> tuple[np.ndarray, int]:
    image = ImageOps.exif_transpose(image).convert("L")
    # Avoid giving blank margins most of the canvas when source lines contain
    # scanner padding. Cropping is based only on pixels, not OCR.
    arr = np.asarray(image)
    threshold = min(245, int(np.percentile(arr, 92)))
    ys, xs = np.where(arr < threshold)
    if len(xs) >= 8:
        x0, x1 = max(0, int(xs.min()) - 8), min(arr.shape[1], int(xs.max()) + 9)
        y0, y1 = max(0, int(ys.min()) - 4), min(arr.shape[0], int(ys.max()) + 5)
        image = image.crop((x0, y0, x1, y1))
    scale = min(HEIGHT / max(1, image.height), WIDTH / max(1, image.width))
    new_w = max(PATCH, min(WIDTH, int(round(image.width * scale))))
    new_h = max(PATCH, min(HEIGHT, int(round(image.height * scale))))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (WIDTH, HEIGHT), 255)
    y = (HEIGHT - new_h) // 2
    canvas.paste(image, (0, y))
    valid_steps = max(1, min(TIME_STEPS, math.ceil(new_w / PATCH)))
    return np.asarray(canvas, dtype=np.uint8), valid_steps


def collect_samples(
    train_n: int,
    dev_n: int,
    test_n: int,
    seed: int,
    max_chars: int,
    max_scan: int,
) -> tuple[dict[str, list[Sample]], dict]:
    from datasets import load_dataset

    quotas = {"train": train_n, "dev": dev_n, "test": test_n}
    per_shelf_cap = {"train": 32, "dev": 16, "test": 16}
    samples: dict[str, list[Sample]] = {k: [] for k in quotas}
    shelf_counts: dict[str, Counter] = {k: Counter() for k in quotas}
    rejected = Counter()

    stream = load_dataset(DATA_REPO, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=5000)

    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > max_scan:
            break
        if all(len(samples[k]) >= quotas[k] for k in quotas):
            break
        text = normalize_text(row.get("text", ""))
        if row.get("line_type") != "DefaultLine":
            rejected["line_type"] += 1
            continue
        if row.get("century") not in (14, 15, 16):
            rejected["century"] += 1
            continue
        if not (8 <= len(text) <= max_chars):
            rejected["length"] += 1
            continue
        shelfmark = str(row.get("shelfmark") or "UNKNOWN")
        split = split_for_shelfmark(shelfmark)
        if len(samples[split]) >= quotas[split]:
            rejected["quota"] += 1
            continue
        if shelf_counts[split][shelfmark] >= per_shelf_cap[split]:
            rejected["shelf_cap"] += 1
            continue
        try:
            image, valid_steps = prepare_line(row["im"])
        except Exception:
            rejected["decode"] += 1
            continue
        if ctc_required_steps(text) > valid_steps:
            rejected["ctc_infeasible"] += 1
            continue
        samples[split].append(
            Sample(
                split=split,
                shelfmark=shelfmark,
                text=text,
                image=image,
                valid_steps=valid_steps,
                century=int(row["century"]),
                script_type=str(row.get("script_type") or ""),
                language=str(row.get("language") or ""),
            )
        )
        shelf_counts[split][shelfmark] += 1

    short = {k: quotas[k] - len(samples[k]) for k in quotas if len(samples[k]) < quotas[k]}
    if short:
        raise RuntimeError(f"Could not fill shelfmark-disjoint quotas after {scanned} rows: {short}")

    shelves = {k: {s.shelfmark for s in v} for k, v in samples.items()}
    assert shelves["train"].isdisjoint(shelves["dev"])
    assert shelves["train"].isdisjoint(shelves["test"])
    assert shelves["dev"].isdisjoint(shelves["test"])

    manifest = {
        "scanned_rows": scanned,
        "counts": {k: len(v) for k, v in samples.items()},
        "shelfmark_counts": {k: len(shelves[k]) for k in shelves},
        "shelfmarks": {k: sorted(shelves[k]) for k in shelves},
        "rejected": dict(rejected),
        "centuries": {k: dict(Counter(s.century for s in v)) for k, v in samples.items()},
        "scripts": {k: dict(Counter(s.script_type for s in v)) for k, v in samples.items()},
        "languages": {k: dict(Counter(s.language for s in v)) for k, v in samples.items()},
    }
    return samples, manifest


def make_vocab(samples: dict[str, list[Sample]]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted({c for split in samples.values() for sample in split for c in sample.text})
    char_to_id = {c: i + 1 for i, c in enumerate(chars)}  # 0 is CTC blank
    id_to_char = {i: c for c, i in char_to_id.items()}
    return char_to_id, id_to_char


@torch.inference_mode()
def extract_features(
    encoder: nn.Module,
    processor,
    samples: Sequence[Sample],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_features = []
    all_lengths = []
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    gh, gw = HEIGHT // PATCH, WIDTH // PATCH
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        arr = np.stack([s.image for s in batch])
        pixels = torch.from_numpy(arr).to(device=device, dtype=torch.float32).unsqueeze(1) / 255.0
        pixels = pixels.repeat(1, 3, 1, 1)
        pixels = (pixels - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = encoder(pixel_values=pixels, interpolate_pos_encoding=True)
            tokens = output.last_hidden_state[:, -(gh * gw) :, :]
            tokens = tokens.reshape(len(batch), gh, gw, -1)
            # Mean retains broad stroke context; maximum retains thin/high-response strokes.
            sequence = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)
        all_features.append(sequence.detach().cpu().to(torch.float16))
        all_lengths.extend(s.valid_steps for s in batch)
        print(f"FEATURES {start + len(batch)}/{len(samples)}", flush=True)
    return torch.cat(all_features, dim=0), torch.tensor(all_lengths, dtype=torch.long)


class CTCHead(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=192,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )
        self.classifier = nn.Linear(384, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(x))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        return self.classifier(x)


def encode_targets(texts: Sequence[str], char_to_id: dict[str, int], device: torch.device):
    encoded = [torch.tensor([char_to_id[c] for c in text], dtype=torch.long) for text in texts]
    lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
    concat = torch.cat(encoded).to(device)
    return concat, lengths


def greedy_decode(ids: Sequence[int], id_to_char: dict[int, str]) -> str:
    out = []
    previous = None
    for idx in ids:
        idx = int(idx)
        if idx != 0 and idx != previous:
            out.append(id_to_char.get(idx, "�"))
        previous = idx
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


def evaluate(
    head: nn.Module,
    features: torch.Tensor,
    lengths: torch.Tensor,
    samples: Sequence[Sample],
    id_to_char: dict[int, str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    head.eval()
    total_edits = total_chars = 0
    ns_edits = ns_chars = 0
    exact = 0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            end = min(len(samples), start + batch_size)
            x = features[start:end].to(device=device, dtype=torch.float32)
            lens = lengths[start:end].to(device)
            logits = head(x, lens)
            best = logits.argmax(dim=-1).cpu().numpy()
            for offset, sample in enumerate(samples[start:end]):
                pred = greedy_decode(best[offset, : int(lengths[start + offset])], id_to_char)
                total_edits += edit_distance(sample.text, pred)
                total_chars += len(sample.text)
                ref_ns = sample.text.replace(" ", "")
                pred_ns = pred.replace(" ", "")
                ns_edits += edit_distance(ref_ns, pred_ns)
                ns_chars += len(ref_ns)
                exact += pred == sample.text
                predictions.append({
                    "shelfmark": sample.shelfmark,
                    "reference": sample.text,
                    "prediction": pred,
                    "century": sample.century,
                    "script_type": sample.script_type,
                    "language": sample.language,
                })
    metrics = {
        "cer": total_edits / max(1, total_chars),
        "cer_no_spaces": ns_edits / max(1, ns_chars),
        "exact_line_accuracy": exact / max(1, len(samples)),
        "lines": len(samples),
    }
    return metrics, predictions


def train_head(
    head: nn.Module,
    train_features: torch.Tensor,
    train_lengths: torch.Tensor,
    train_samples: Sequence[Sample],
    dev_features: torch.Tensor,
    dev_lengths: torch.Tensor,
    dev_samples: Sequence[Sample],
    char_to_id: dict[str, int],
    id_to_char: dict[int, str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Module, list[dict]]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_cer = float("inf")
    patience = 0
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            x = train_features[idx].to(device=device, dtype=torch.float32)
            lens = train_lengths[idx].to(device)
            texts = [train_samples[i].text for i in idx]
            targets, target_lengths = encode_targets(texts, char_to_id, device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(x, lens)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = loss_fn(log_probs, targets, lens, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        dev_metrics, _ = evaluate(
            head, dev_features, dev_lengths, dev_samples, id_to_char, device, batch_size
        )
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **dev_metrics}
        history.append(row)
        print("EPOCH", json.dumps(row, ensure_ascii=False), flush=True)
        if dev_metrics["cer"] < best_cer - 1e-4:
            best_cer = dev_metrics["cer"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    if best_state is None:
        raise RuntimeError("No finite best model state")
    head.load_state_dict(best_state)
    return head, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--dev", type=int, default=64)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=48)
    parser.add_argument("--max-scan", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device, flush=True)

    from huggingface_hub import dataset_info, model_info
    from transformers import AutoImageProcessor, AutoModel

    data_meta = dataset_info(DATA_REPO)
    dino_meta = model_info(DINO_REPO)
    print("REVISIONS", DATA_REPO, data_meta.sha, DINO_REPO, dino_meta.sha, flush=True)

    samples, manifest = collect_samples(
        args.train, args.dev, args.test, args.seed, args.max_chars, args.max_scan
    )
    manifest.update({
        "dataset": DATA_REPO,
        "dataset_revision": data_meta.sha,
        "seed": args.seed,
        "eligibility": {"centuries": [14, 15, 16], "min_chars": 8, "max_chars": args.max_chars},
    })
    (args.output / "DATA_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DATA", json.dumps(manifest["counts"]), manifest["shelfmark_counts"], flush=True)

    char_to_id, id_to_char = make_vocab(samples)
    vocab = {"blank_id": 0, "char_to_id": char_to_id}
    (args.output / "VOCAB.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    processor = AutoImageProcessor.from_pretrained(DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN"))
    encoder = AutoModel.from_pretrained(DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN"))
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    features = {}
    lengths = {}
    for split in ("train", "dev", "test"):
        features[split], lengths[split] = extract_features(
            encoder, processor, samples[split], device, args.feature_batch_size
        )
        torch.save({"features": features[split], "lengths": lengths[split]}, args.output / f"{split}_features.pt")
    input_dim = features["train"].shape[-1]
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = CTCHead(input_dim=input_dim, vocab_size=len(char_to_id) + 1).to(device)
    untrained_dev, _ = evaluate(
        head, features["dev"], lengths["dev"], samples["dev"], id_to_char, device, args.batch_size
    )
    untrained_test, _ = evaluate(
        head, features["test"], lengths["test"], samples["test"], id_to_char, device, args.batch_size
    )

    head, history = train_head(
        head,
        features["train"], lengths["train"], samples["train"],
        features["dev"], lengths["dev"], samples["dev"],
        char_to_id, id_to_char, device, args.epochs, args.batch_size, args.seed,
    )
    trained_dev, dev_predictions = evaluate(
        head, features["dev"], lengths["dev"], samples["dev"], id_to_char, device, args.batch_size
    )
    trained_test, test_predictions = evaluate(
        head, features["test"], lengths["test"], samples["test"], id_to_char, device, args.batch_size
    )

    # Blank control uses the same frozen image encoder preprocessing, but the
    # trained head is evaluated only after the scientific metrics are frozen.
    blank_sample = Sample("blank", "BLANK", "", np.full((HEIGHT, WIDTH), 255, np.uint8), TIME_STEPS, 0, "", "")
    # Encoder was released; reload once for this single control to keep training memory bounded.
    encoder = AutoModel.from_pretrained(DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN")).to(device).eval()
    blank_features, blank_lengths = extract_features(encoder, processor, [blank_sample], device, 1)
    blank_logits = head(blank_features.to(device=device, dtype=torch.float32), blank_lengths.to(device))
    blank_ids = blank_logits.argmax(dim=-1)[0, : int(blank_lengths[0])].detach().cpu().tolist()
    blank_prediction = greedy_decode(blank_ids, id_to_char)
    del encoder

    pilot_pass = (
        trained_test["cer"] <= untrained_test["cer"] - 0.10
        and trained_dev["cer"] < 0.95
        and all(math.isfinite(x["loss"]) for x in history)
    )
    result = {
        "status": "COMPLETE",
        "verdict": "DINOV3_CATMUS_PREFLIGHT_PASS" if pilot_pass else "DINOV3_CATMUS_PREFLIGHT_FAIL",
        "pilot_pass": pilot_pass,
        "architecture": {
            "encoder": DINO_REPO,
            "encoder_revision": dino_meta.sha,
            "encoder_frozen": True,
            "input": [HEIGHT, WIDTH],
            "time_steps": TIME_STEPS,
            "feature_dim": input_dim,
            "ctc_vocab_size_including_blank": len(char_to_id) + 1,
        },
        "dataset": {"repo": DATA_REPO, "revision": data_meta.sha, **manifest["counts"]},
        "untrained": {"dev": untrained_dev, "test": untrained_test},
        "trained": {"dev": trained_dev, "test": trained_test},
        "blank_prediction": blank_prediction,
        "history": history,
        "device": str(device),
        "torch": torch.__version__,
    }
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
            "input_dim": input_dim,
            "vocab": vocab,
            "architecture": result["architecture"],
        },
        args.output / "dinov3_catmus_ctc_head.pt",
    )
    (args.output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "PREDICTIONS.tsv").open("w", encoding="utf-8") as f:
        f.write("split\tshelfmark\tcentury\tscript_type\tlanguage\treference\tprediction\n")
        for split, rows in (("dev", dev_predictions), ("test", test_predictions)):
            for row in rows:
                values = [split, row["shelfmark"], str(row["century"]), row["script_type"], row["language"], row["reference"], row["prediction"]]
                f.write("\t".join(v.replace("\t", " ").replace("\n", " ") for v in values) + "\n")

    md = [
        "# DINOv3–CATMuS CTC preflight",
        "",
        f"- Verdict: **{result['verdict']}**",
        f"- Train/dev/test: **{args.train}/{args.dev}/{args.test}**",
        f"- Shelfmarks: **{manifest['shelfmark_counts']}**",
        f"- Untrained test CER: **{untrained_test['cer']:.4f}**",
        f"- Trained dev CER: **{trained_dev['cer']:.4f}**",
        f"- Trained test CER: **{trained_test['cer']:.4f}**",
        f"- Trained test CER without spaces: **{trained_test['cer_no_spaces']:.4f}**",
        f"- Exact test-line accuracy: **{trained_test['exact_line_accuracy']:.4f}**",
        f"- Blank prediction: `{blank_prediction}`",
        "",
        "The encoder remained frozen. No dictionary, language model, abbreviation expansion or word correction was used.",
    ]
    (args.output / "RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("DINOV3_CATMUS_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
