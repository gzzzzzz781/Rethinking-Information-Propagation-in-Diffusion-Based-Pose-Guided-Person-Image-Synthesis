from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_diffusion.utils import ensure_dir, resize_pil_image


def image_key(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts[1:] if path.parts and path.parts[0] == "img" else path.parts
    if not parts:
        return path.stem
    return "_".join([*parts[:-1], Path(parts[-1]).stem])


def pair_output_name(source_rel: str, target_rel: str) -> str:
    return f"{image_key(source_rel)}_to_{image_key(target_rel)}.png"


def resolve_target_path(dataset_root: Path, target_rel: str) -> Path:
    target_path = dataset_root / target_rel
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    return target_path


def build_resolution_dirs(output_dir: Path) -> dict[int, Path]:
    return {
        256: ensure_dir(output_dir / "256"),
        512: ensure_dir(output_dir / "512"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="dataset/deepfashion")
    parser.add_argument("--pairs_file", default="test_pairs.txt")
    parser.add_argument("--output_dir", default="outputs/gt")
    args = parser.parse_args()

    from metrics import parse_pairs_file

    dataset_root = Path(args.dataset_root)
    output_dir = ensure_dir(args.output_dir)
    resolution_dirs = build_resolution_dirs(output_dir)
    pairs = parse_pairs_file(dataset_root, args.pairs_file)

    sizes = {
        256: (176, 256),
        512: (352, 512),
    }

    count = 0
    for target_rel, source_rel_list in tqdm(pairs, desc="prepare gt"):
        target_path = resolve_target_path(dataset_root, target_rel)
        image = Image.open(target_path).convert("RGB")

        for source_rel in source_rel_list:
            output_name = pair_output_name(source_rel, target_rel)
            for resolution, pil_size in sizes.items():
                resized = resize_pil_image(image, pil_size)
                resized.save(resolution_dirs[resolution] / output_name)
            count += 1

    print(f"Saved {count} paired GT images to {output_dir / '256'} and {output_dir / '512'}")


if __name__ == "__main__":
    main()
