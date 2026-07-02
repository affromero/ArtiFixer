# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the quality-vs-memory plot from the DL3DV config sweep JSON.

Reads dl3dv_config_sweep.json (one point per inference configuration:
quality on the held-out gap, measured peak GPU memory, wall time) and
writes quality_vs_memory.png used in the README.

Usage:
    python docs/benchmarks/plot_quality_vs_memory.py \
        [--sweep docs/benchmarks/dl3dv_config_sweep.json] \
        [--out docs/benchmarks/quality_vs_memory.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

GPU_TIERS = ((24, "24 GB\n4090/3090"), (48, "48 GB\nL40S/A6000"), (80, "80 GB\nH100/A100"))


def point_color(point: dict) -> str:
    if point.get("budget_gb"):
        return "tab:green"
    return "tab:orange" if point["quantization"] == "fp8" else "tab:blue"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=HERE / "dl3dv_config_sweep.json")
    parser.add_argument("--out", type=Path, default=HERE / "quality_vs_memory.png")
    args = parser.parse_args()

    sweep = json.loads(args.sweep.read_text())
    measured = [p for p in sweep["points"] if p["status"] == "measured"]
    estimated = [p for p in sweep["points"] if p["status"] == "estimated"]

    fig, (ax_psnr, ax_lpips) = plt.subplots(
        2, 1, figsize=(9.0, 7.8), sharex=True, height_ratios=[3, 2]
    )

    for gpu_gb, name in GPU_TIERS:
        for ax in (ax_psnr, ax_lpips):
            ax.axvline(gpu_gb, color="0.85", lw=1, zorder=0)
        ax_psnr.annotate(
            name, (gpu_gb, 0.02), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7.5, color="0.45",
        )

    # Alternate label offsets in peak order so neighboring labels don't collide.
    label_offsets = {}
    for rank, p in enumerate(sorted(measured, key=lambda q: q["peak_gb"])):
        label_offsets[p["label"]] = (0, 13) if rank % 2 == 0 else (0, -26)

    for p in measured:
        marker = "*" if p.get("budget_gb") else "o"
        size = 230 if marker == "*" else 75
        ax_psnr.scatter(p["peak_gb"], p["psnr"], s=size, marker=marker, color=point_color(p), zorder=3)
        ax_lpips.scatter(p["peak_gb"], p["lpips"], s=size, marker=marker, color=point_color(p), zorder=3)
        label = p["label"]
        if p.get("budget_gb"):
            label += f"\n(enforced {p['budget_gb']:.0f} GB budget)"
        ax_psnr.annotate(
            label, (p["peak_gb"], p["psnr"]),
            textcoords="offset points", xytext=label_offsets[p["label"]],
            ha="center", fontsize=8,
        )

    psnr_lo = min(p["psnr"] for p in measured)
    psnr_hi = max(p["psnr"] for p in measured)
    pad = max(0.12, 0.45 * (psnr_hi - psnr_lo))
    for p in estimated:
        y = psnr_lo - 0.45 * pad
        ax_psnr.scatter(p["peak_gb"], y, s=110, marker="X", facecolors="none", edgecolors="tab:red", zorder=3)
        ax_psnr.annotate(
            f"{p['label']}\nest. ~{p['peak_gb']:.0f} GB — OOMs on 80 GB",
            (p["peak_gb"], y),
            textcoords="offset points", xytext=(0, 13), ha="center", fontsize=8, color="tab:red",
        )

    ax_psnr.set_ylabel("PSNR on held-out gap (dB) ↑")
    ax_lpips.set_ylabel("LPIPS ↓")
    ax_lpips.set_xlabel("measured peak GPU memory (GiB, torch allocator)")
    ax_psnr.set_title(sweep["title"], fontsize=10, pad=12)
    ax_psnr.set_ylim(psnr_lo - pad, psnr_hi + pad)

    for ax in (ax_psnr, ax_lpips):
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 95)

    handles = [
        plt.Line2D([], [], color="tab:blue", marker="o", ls="", label="bf16"),
        plt.Line2D([], [], color="tab:orange", marker="o", ls="", label="fp8 weights"),
        plt.Line2D([], [], color="tab:green", marker="*", ms=14, ls="", label="fp8 + enforced 24 GB budget"),
        plt.Line2D([], [], color="tab:red", marker="X", mfc="none", ls="", label="estimated (exceeds 80 GB)"),
    ]
    ax_psnr.legend(handles=handles, loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
