from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .pose import load_pose_map
from .utils import resize_pil_image


@dataclass
class PairItem:
    target_image: str
    source_image: str


@dataclass
class ImageItem:
    image_path: str


def _resolve_root_path(root: Path, rel_path: str) -> Path:
    return root / rel_path


def _densepose_path_from_image(root: Path, image_path: str) -> Path:
    rel = Path(image_path)
    if rel.parts and rel.parts[0] == "img":
        rel = Path(*rel.parts[1:])
    densepose_name = "-".join([*rel.parent.parts, rel.stem]) + "_densepose.png"
    return root / "densepose" / densepose_name


class PoseTransferDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        pairs_file: str = "train_pairs.txt",
        resolution: int = 256,
    ):
        self.root = Path(root)
        self.pairs_file = self.root / pairs_file
        self.resolution = int(resolution)
        self.skipped_missing_pose = 0
        self.items = self._load_pairs()

    def _load_pairs(self) -> list[PairItem]:
        if not self.pairs_file.exists():
            raise FileNotFoundError(self.pairs_file)
        items: list[PairItem] = []
        for line in self.pairs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            target_image = parts[0]
            if not _densepose_path_from_image(self.root, target_image).exists():
                self.skipped_missing_pose += len(parts) - 1
                continue
            for source_image in parts[1:]:
                items.append(
                    PairItem(
                        target_image=target_image,
                        source_image=source_image,
                    )
                )
        if not items:
            raise ValueError(f"No valid pairs found in {self.pairs_file}")
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, rel_path: str) -> Image.Image:
        path = _resolve_root_path(self.root, rel_path)
        with Image.open(path) as img:
            return img.convert("RGB")

    def _image_to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = resize_pil_image(img, (self.resolution, self.resolution))
        tensor = TF.to_tensor(img)
        return tensor.mul(2.0).sub(1.0)

    def _pose_to_tensor(self, image_path: str, img_size: tuple[int, int]) -> torch.Tensor:
        pose_path = _densepose_path_from_image(self.root, image_path)
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        return load_pose_map(pose_path, size=(self.resolution, self.resolution))

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        source_rel = item.source_image
        target_rel = item.target_image

        target_img = self._load_image(target_rel)
        source_img = self._load_image(source_rel)

        target_pose = self._pose_to_tensor(item.target_image, target_img.size)

        target_img = self._image_to_tensor(target_img)
        source_img = self._image_to_tensor(source_img)
        if torch.rand(()) < 0.5:
            target_img = torch.flip(target_img, dims=[2])
            source_img = torch.flip(source_img, dims=[2])
            target_pose = torch.flip(target_pose, dims=[2])

        return {
            "source_image": source_img,
            "target_image": target_img,
            "target_pose": target_pose,
            "source_path": source_rel,
            "target_path": item.target_image,
        }
