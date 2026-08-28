from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

try:
    PIL_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    PIL_BICUBIC = Image.BICUBIC


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_tensor_image(tensor: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((tensor.clamp(-1, 1) + 1) * 0.5, nrow=nrow)
    save_image(grid, str(path))


def resize_pil_image(image: Image.Image, size: int | tuple[int, int]) -> Image.Image:
    if isinstance(size, int):
        size = (size, size)
    return image.resize(size, PIL_BICUBIC)


def resize_tensor_image(
    tensor: torch.Tensor,
    size: int | tuple[int, int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    return TF.resize(tensor, list(size), interpolation=interpolation, antialias=True)
