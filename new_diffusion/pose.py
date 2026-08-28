from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def load_pose_map(path: str | Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if size is not None:
                img = img.resize((size[1], size[0]), resample=Image.BILINEAR)
            return TF.to_tensor(img).clamp(0.0, 1.0)

    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(arr)
        if tensor.ndim != 3:
            raise ValueError(f"Expected HWC pose array in {path}, got {tensor.shape}")
        tensor = tensor.permute(2, 0, 1)
        if size is not None and tuple(tensor.shape[-2:]) != tuple(size):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return tensor.clamp(0.0, 1.0)

    raise ValueError(f"Unsupported densepose format: {path}")
