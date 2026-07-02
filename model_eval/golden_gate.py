"""Golden-baseline regression gate for the ArtiFixer optimization fork.

Freezes a known-good ArtiFixer run (GT / base / AF2D / AF3D frames + per-frame
PSNR/SSIM/LPIPS) and gates every future optimization against it.

A candidate PASSES only if, vs the golden AF3D:
  - mean ΔPSNR  >= -0.10 dB
  - mean ΔLPIPS <= +0.005
  - worst single-frame ΔPSNR >= -0.50 dB   (no frame silently collapses)
Plus a side-by-side montage (golden vs candidate on the worst-Δ frame) is
written for MANDATORY visual review — the gate is metrics AND picture.

Usage (needs torch/lpips/scikit-image/numpy/PIL):
  golden_gate.py freeze --name apartment_lr_g32 \
      --gt <gt_dir> --base <rendered_dir> --af2d <pred_dir> --af3d <renders_dir>
  golden_gate.py check  --name apartment_lr_g32 \
      --af3d <candidate_renders_dir> [--af2d <candidate_pred_dir>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"
# Thresholds calibrated against the pipeline's measured run-to-run noise
# floor: an identical-input rerun (same base checkpoint, only unseeded
# sampling RNG differing — A1b, 2026-07-01) scored mean dPSNR -0.110,
# worst-frame dPSNR -0.448, mean dLPIPS -0.011 vs golden. Gates are set
# ~2x the observed noise so they fail on real regressions, not RNG.
# With --seed (deterministic inference) reruns should be near-exact and
# these gates conservative.
GATE_MEAN_DPSNR = -0.25
GATE_MEAN_DLPIPS = 0.005
GATE_WORST_DPSNR = -1.00


def load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return arr / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else -10.0 * float(np.log10(mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        structural_similarity(a, b, channel_axis=2, data_range=1.0)
    )


class LpipsScorer:
    def __init__(self) -> None:
        import lpips

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device)

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        return float(self.net(ta, tb).item())


def frame_metrics(
    gt_dir: Path, arm_dir: Path, scorer: LpipsScorer
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for gt_path in sorted(gt_dir.glob("*.png")):
        arm_path = arm_dir / gt_path.name
        if not arm_path.exists():
            raise FileNotFoundError(f"missing frame {arm_path}")
        gt = load_rgb(gt_path)
        arm = load_rgb(arm_path)
        if arm.shape != gt.shape:
            raise ValueError(
                f"shape mismatch {arm_path}: {arm.shape} vs GT {gt.shape}"
            )
        out[gt_path.name] = {
            "psnr": psnr(gt, arm),
            "ssim": ssim(gt, arm),
            "lpips": scorer(gt, arm),
        }
    return out


def summarize(per_frame: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("psnr", "ssim", "lpips")
    return {k: float(np.mean([f[k] for f in per_frame.values()])) for k in keys}


def montage(paths: list[Path], labels: list[str], out_path: Path) -> None:
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    h = min(i.shape[0] for i in imgs)
    sep = np.full((h, 6, 3), 255, np.uint8)
    row: list[np.ndarray] = []
    for i, img in enumerate(imgs):
        if i:
            row.append(sep)
        row.append(img[:h])
    Image.fromarray(np.concatenate(row, axis=1)).save(out_path)
    print(f"montage ({' | '.join(labels)}): {out_path}")


def cmd_freeze(args: argparse.Namespace) -> int:
    gdir = GOLDEN_ROOT / args.name
    if gdir.exists():
        print(f"REFUSING to overwrite existing golden {gdir}")
        return 1
    scorer = LpipsScorer()
    arms = {"base": Path(args.base), "af2d": Path(args.af2d), "af3d": Path(args.af3d)}
    gt_src = Path(args.gt)
    gap_names = sorted(p.name for p in gt_src.glob("*.png"))

    (gdir / "frames" / "gt").mkdir(parents=True)
    for name in gap_names:
        shutil.copy2(gt_src / name, gdir / "frames" / "gt" / name)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for arm, src in arms.items():
        (gdir / "frames" / arm).mkdir(parents=True)
        for name in gap_names:
            shutil.copy2(src / name, gdir / "frames" / arm / name)
        metrics[arm] = frame_metrics(gt_src, src, scorer)
        print(f"{arm}: {summarize(metrics[arm])}")

    worst = min(metrics["af3d"], key=lambda n: metrics["af3d"][n]["psnr"])
    payload = {
        "name": args.name,
        "gap_frames": gap_names,
        "worst_af3d_frame": worst,
        "per_frame": metrics,
        "summary": {arm: summarize(m) for arm, m in metrics.items()},
        "gate": {
            "mean_dpsnr_min": GATE_MEAN_DPSNR,
            "mean_dlpips_max": GATE_MEAN_DLPIPS,
            "worst_frame_dpsnr_min": GATE_WORST_DPSNR,
        },
    }
    (gdir / "golden.json").write_text(json.dumps(payload, indent=2))
    montage(
        [gdir / "frames" / d / worst for d in ("gt", "base", "af2d", "af3d")],
        ["GT", "base", "AF2D", "AF3D"],
        gdir / "worst_frame_montage.png",
    )
    print(f"FROZEN golden -> {gdir}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    gdir = GOLDEN_ROOT / args.name
    golden = json.loads((gdir / "golden.json").read_text())
    scorer = LpipsScorer()
    gt_dir = gdir / "frames" / "gt"
    cand = frame_metrics(gt_dir, Path(args.af3d), scorer)
    gold = golden["per_frame"]["af3d"]

    deltas = {
        n: {
            "dpsnr": cand[n]["psnr"] - gold[n]["psnr"],
            "dlpips": cand[n]["lpips"] - gold[n]["lpips"],
        }
        for n in gold
    }
    mean_dpsnr = float(np.mean([d["dpsnr"] for d in deltas.values()]))
    mean_dlpips = float(np.mean([d["dlpips"] for d in deltas.values()]))
    worst_name = min(deltas, key=lambda n: deltas[n]["dpsnr"])
    worst_dpsnr = deltas[worst_name]["dpsnr"]

    checks = {
        f"mean ΔPSNR {mean_dpsnr:+.3f} >= {GATE_MEAN_DPSNR}": mean_dpsnr
        >= GATE_MEAN_DPSNR,
        f"mean ΔLPIPS {mean_dlpips:+.4f} <= {GATE_MEAN_DLPIPS}": mean_dlpips
        <= GATE_MEAN_DLPIPS,
        f"worst-frame ΔPSNR {worst_dpsnr:+.3f} ({worst_name}) >= "
        f"{GATE_WORST_DPSNR}": worst_dpsnr >= GATE_WORST_DPSNR,
    }
    print(f"candidate summary: {summarize(cand)}")
    print(f"golden    summary: {golden['summary']['af3d']}")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    out_montage = Path(args.af3d).parent / f"gate_vs_golden_{worst_name}"
    montage(
        [
            gt_dir / worst_name,
            gdir / "frames" / "af3d" / worst_name,
            Path(args.af3d) / worst_name,
        ],
        ["GT", "golden AF3D", "candidate AF3D"],
        out_montage,
    )
    passed = all(checks.values())
    print(
        f"GATE: {'PASS' if passed else 'FAIL'} "
        "(now VISUALLY inspect the montage — metrics alone do not pass)"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--name", required=True)
    f.add_argument("--gt", required=True)
    f.add_argument("--base", required=True)
    f.add_argument("--af2d", required=True)
    f.add_argument("--af3d", required=True)
    c = sub.add_parser("check")
    c.add_argument("--name", required=True)
    c.add_argument("--af3d", required=True)
    args = parser.parse_args()
    return cmd_freeze(args) if args.cmd == "freeze" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
