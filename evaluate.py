from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PIL import Image
import torch

from metrics import (
    IMAGE_EXTS,
    compute_fid_pocold_style_from_real_paths,
    compute_lpips_pocold_style_from_pairs,
    compute_psnr_pocold_style_from_pairs,
    compute_ssim_pocold_style_from_pairs,
    list_images,
)


def build_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_resolution_dir(base_path: str | Path, resolution: int) -> Path:
    base = Path(base_path)
    candidates = [base / str(resolution), base]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(candidates[0])


def list_direct_image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def image_size_for_resolution(resolution: int) -> tuple[int, int]:
    if resolution == 256:
        return 256, 176
    return 512, 352


def pil_size_for_resolution(resolution: int) -> tuple[int, int]:
    height, width = image_size_for_resolution(resolution)
    return width, height


def assert_image_sizes(paths: list[Path], expected_size: tuple[int, int], label: str, max_checks: int = 32) -> None:
    for path in paths[:max_checks]:
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(
                    f"{label} image size mismatch at {path}: expected {expected_size}, got {image.size}. "
                    "Use the GT/FID-real directory that matches --resolution."
                )


def resolve_fid_real_paths(training_path: str | Path, fid_real_path: str | Path | None, resolution: int) -> list[Path]:
    candidates = []
    if fid_real_path:
        base = Path(fid_real_path)
        candidates.extend([base / str(resolution), base])
    else:
        base = Path(training_path)
        candidates.extend(
            [
                base / f"fid_real_{resolution}x{176 if resolution == 256 else 352}",
                base / "train_highres",
                base / "train",
                base,
            ]
        )

    for candidate in candidates:
        if candidate.is_dir():
            paths = list_direct_image_paths(candidate)
            if paths:
                return paths
            paths = list_images(candidate)
            if paths:
                return paths
    raise FileNotFoundError(
        f"Could not find FID real images from {fid_real_path or training_path} for resolution {resolution}"
    )


def match_common_images(gt_dir: Path, pred_dir: Path) -> list[tuple[Path, Path]]:
    gt_files = {p.name: p for p in list_direct_image_paths(gt_dir)}
    pred_files = {p.name: p for p in list_direct_image_paths(pred_dir)}
    common = sorted(set(gt_files) & set(pred_files))
    return [(pred_files[name], gt_files[name]) for name in common]


def evaluate_whole_image(img_path, gt_path, training_path, fid_real_path, device, resolution=256, skip_fid=False):
    gt_dir = resolve_resolution_dir(gt_path, resolution)
    pred_dir = resolve_resolution_dir(img_path, resolution)
    pairs = match_common_images(gt_dir, pred_dir)
    if not pairs:
        raise ValueError(f"No matching prediction/GT files found under {gt_dir} and {pred_dir}")

    gt_count = len(list_direct_image_paths(gt_dir))
    pred_count = len(list_direct_image_paths(pred_dir))
    skipped = max(gt_count, pred_count) - len(pairs)
    if skipped > 0:
        print(f"Skipping {skipped} missing prediction images for {resolution} resolution.")

    eval_size = image_size_for_resolution(resolution)
    expected_pil_size = pil_size_for_resolution(resolution)
    assert_image_sizes([gt for _, gt in pairs], expected_pil_size, f"{resolution} GT")
    lpips_score = compute_lpips_pocold_style_from_pairs(pairs, batch_size=64, device=device, eval_size=eval_size)
    ssim_score, ssim_256_score = compute_ssim_pocold_style_from_pairs(pairs, eval_size=eval_size)
    psnr_score = compute_psnr_pocold_style_from_pairs(pairs, eval_size=eval_size)

    print(f"Evaluation Results on {resolution} resolution:")
    print(f"LPIPS: {lpips_score}")
    print(f"SSIM: {ssim_score}")
    print(f"SSIM_256: {ssim_256_score}")
    print(f"PSNR: {psnr_score}")
    if skip_fid:
        print("FID: skipped")
    else:
        fid_real_paths = resolve_fid_real_paths(training_path, fid_real_path, resolution)
        assert_image_sizes(fid_real_paths, expected_pil_size, f"{resolution} FID real")
        score_fid = compute_fid_pocold_style_from_real_paths(
            pred_paths=[pred for pred, _ in pairs],
            real_paths=fid_real_paths,
            batch_size=64,
            device=device,
            eval_size=eval_size,
        )
        print(f"FID: {score_fid}")
    print("------------------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="./results/gt/")
    parser.add_argument("--img_path", type=str, default="./results/mcld/")
    parser.add_argument("--training_path", type=str, default="./dataset/fashion/")
    parser.add_argument("--fid_real_path", type=str, default="")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip_fid", action="store_true")
    args = parser.parse_args()

    device = build_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This evaluation script currently expects a CUDA device.")

    resolutions = [args.resolution] if args.resolution is not None else [256, 512]
    for resolution in resolutions:
        evaluate_whole_image(
            args.img_path,
            args.gt_path,
            args.training_path,
            args.fid_real_path or None,
            device,
            resolution=resolution,
            skip_fid=args.skip_fid,
        )


if __name__ == "__main__":
    main()
