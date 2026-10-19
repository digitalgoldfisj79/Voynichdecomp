#!/usr/bin/env python3
"""Generate preregistered positive or blank multispectral page controls."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["positive", "blank"], required=True)
    p.add_argument("--seed", type=int, default=116)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + (1 if args.mode == "blank" else 0))
    h, w = 700, 520
    yy, xx = np.mgrid[:h, :w]
    page = np.full((h, w), 0.86, np.float32)
    page += 0.025 * np.sin(xx / 37) + 0.018 * np.cos(yy / 61) + rng.normal(0, 0.008, (h, w))
    page_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(page_mask, (30, 20), (490, 680), 255, -1)
    base = np.full((h, w), 0.10, np.float32)
    base[page_mask > 0] = page[page_mask > 0]

    visible = np.zeros((h, w), np.uint8)
    for y in [70, 105, 140, 175, 210, 245]:
        cv2.putText(visible, "michiton oladabas 8ar", (55, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
    hidden = np.zeros((h, w), np.uint8)
    for y in [370, 405, 440, 475, 510, 545]:
        cv2.putText(hidden, "qokedy chedy aiin", (70, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.48, 255, 1, cv2.LINE_AA)

    stain = np.exp(-(((xx - 290) / 190) ** 2 + ((yy - 475) / 170) ** 2)) * 0.25
    wavelengths = [365, 395, 450, 530, 625, 700, 850, 1050]
    hidden_strengths = [0.00, 0.04, 0.08, 0.16, 0.22, 0.28, 0.34, 0.25]
    if args.mode == "blank":
        hidden_strengths = [0.0] * len(wavelengths)

    for i, (wavelength, hidden_strength) in enumerate(zip(wavelengths, hidden_strengths)):
        image = base.copy()
        visible_strength = 0.48 + 0.12 * np.sin(i)
        image[visible > 0] -= visible_strength * (visible[visible > 0] / 255)
        image[hidden > 0] -= hidden_strength * (hidden[hidden > 0] / 255)
        image -= stain * (0.7 + 0.1 * np.cos(i))
        image += rng.normal(0, 0.008 + 0.001 * i, (h, w))
        transform = np.float32([[1, 0, (i % 3) - 1], [0, 1, (i // 3) - 1]])
        image = cv2.warpAffine(image, transform, (w, h), borderValue=0.08)
        Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(
            args.output / f"Voynich_116v_MB{wavelength:04d}_{i:03d}.jpg", quality=95
        )
    np.save(args.output / "CONTROL_hidden_ground_truth.npy", hidden)


if __name__ == "__main__":
    main()
