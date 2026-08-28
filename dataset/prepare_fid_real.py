from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
try:
    PIL_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    PIL_BICUBIC = Image.BICUBIC


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_pil_image(image: Image.Image, size: int | tuple[int, int]) -> Image.Image:
    if isinstance(size, int):
        size = (size, size)
    return image.resize(size, PIL_BICUBIC)


def list_source_images(source_dir: str | Path) -> list[Path]:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    return sorted([p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def default_source_dir(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    train_highres = dataset_root / "train_highres"
    if train_highres.exists():
        return train_highres
    return dataset_root / "img"


def output_name(path: Path, source_dir: Path) -> str:
    rel = path.relative_to(source_dir)
    stem = "_".join([*rel.parts[:-1], rel.stem]) if len(rel.parts) > 1 else rel.stem
    return f"{stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="dataset/deepfashion")
    parser.add_argument("--source_dir", default="")
    parser.add_argument("--output_dir", default="dataset/deepfashion/fid_real_256x176")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 176], metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()

    source_dir = Path(args.source_dir) if args.source_dir else default_source_dir(args.dataset_root)
    output_dir = ensure_dir(args.output_dir)
    height, width = args.resolution
    paths = list_source_images(source_dir)
    if not paths:
        raise ValueError(f"No source images found in {source_dir}")

    for path in tqdm(paths, desc="prepare fid real"):
        image = Image.open(path).convert("RGB")
        image = resize_pil_image(image, (width, height))
        image.save(output_dir / output_name(path, source_dir))

    print(f"Saved {len(paths)} resized FID real images to {output_dir}")


if __name__ == "__main__":
    main()
