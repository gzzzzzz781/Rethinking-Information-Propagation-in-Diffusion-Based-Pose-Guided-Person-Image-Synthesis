from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .refiner_model import build_refiner
from new_diffusion.utils import ensure_dir, load_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_device(device_arg: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "auto" else device_arg)
    if device_arg == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", dtype=get_weight_dtype(mixed_precision))
    return torch.cuda.amp.autocast(dtype=get_weight_dtype(mixed_precision))


def parse_resolution(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("Use --resolution SIZE or --resolution HEIGHT WIDTH.")


def load_rgb01_image(path: str | Path, resolution: tuple[int, int]) -> torch.Tensor:
    height, width = resolution
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return TF.to_tensor(image)


def save_rgb01_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.detach().cpu().clamp(0.0, 1.0)).save(path)


def infer_model_config(ckpt: dict) -> dict[str, object]:
    args = ckpt.get("args", {})
    model_type = str(args.get("model_type", "window_restormer"))
    if model_type != "window_restormer":
        raise ValueError(f"Unsupported refiner checkpoint model_type: {model_type}")
    return {
        "model_dim": int(args.get("model_dim", 48)),
        "num_blocks": tuple(args.get("num_blocks", [1, 2, 2, 4])),
        "num_refinement_blocks": int(args.get("num_refinement_blocks", 2)),
        "heads": tuple(args.get("heads", [1, 2, 4, 8])),
        "ffn_expansion_factor": float(args.get("ffn_expansion_factor", 2.0)),
        "model_bias": bool(args.get("model_bias", False)),
        "window_size": int(args.get("window_size", 8)),
        "residual_scale": float(args.get("residual_scale", 0.1)),
    }


def build_model_from_checkpoint(ckpt: dict, device: torch.device):
    config = infer_model_config(ckpt)
    model = build_refiner(**config).to(device)
    state_dict = ckpt["model"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


class RefinerPredictionDataset(Dataset):
    def __init__(self, input_dir: str | Path, resolution: tuple[int, int], recursive: bool = True):
        self.input_dir = Path(input_dir)
        self.resolution = resolution
        pattern = "**/*" if recursive else "*"
        self.paths = sorted(
            path for path in self.input_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No input images found in {self.input_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        image = load_rgb01_image(path, self.resolution)
        rel_path = path.relative_to(self.input_dir)
        return {"image": image, "rel_path": str(rel_path)}


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    mixed_precision: str,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    with autocast_context(device, mixed_precision):
        refined = model(images)
    return refined.detach().cpu()


def run_single(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, resolution: tuple[int, int]) -> None:
    image = load_rgb01_image(args.input, resolution).unsqueeze(0)
    refined = predict_batch(model, image, device, args.mixed_precision)[0]
    save_rgb01_image(refined, args.output)


def run_batch(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, resolution: tuple[int, int]) -> None:
    output_dir = ensure_dir(args.output_dir)
    dataset = RefinerPredictionDataset(args.input_dir, resolution, recursive=not args.no_recursive)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    for batch in tqdm(loader, desc="predict refiner"):
        refined = predict_batch(model, batch["image"], device, args.mixed_precision)
        for image, rel_path in zip(refined, batch["rel_path"]):
            save_path = output_dir / Path(rel_path)
            save_rgb01_image(image, save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/refiner/last.pt")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="outputs/refiner.png")
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--output_dir", default="outputs/refiner_batch")
    parser.add_argument("--resolution", type=int, nargs="+", default=[512], help="Use SIZE or HEIGHT WIDTH.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--no_recursive", action="store_true")
    args = parser.parse_args()

    if bool(args.input) == bool(args.input_dir):
        raise ValueError("Use either --input for single-image inference or --input_dir for batch inference.")
    if args.input_dir and not args.output_dir:
        raise ValueError("Batch inference requires --output_dir.")

    resolution = parse_resolution(args.resolution)
    device = build_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model, config = build_model_from_checkpoint(ckpt, device)
    print("Loaded refiner config: " + ", ".join(f"{key}={value}" for key, value in config.items()))

    if args.input:
        run_single(args, model, device, resolution)
    else:
        run_batch(args, model, device, resolution)


if __name__ == "__main__":
    main()
