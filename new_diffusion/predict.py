from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .autoencoder import DiffusersAutoencoder
from .diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from .dataset import _densepose_path_from_image
from .unet import PoseTransferUNet
from .pose import load_pose_map
from .utils import ensure_dir, load_checkpoint, resize_pil_image, save_tensor_image


INFERENCE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _load_image(path: str | Path, resolution: int) -> torch.Tensor:
    img = resize_pil_image(Image.open(path).convert("RGB"), (resolution, resolution))
    return TF.to_tensor(img).mul(2.0).sub(1.0)


def build_device(device_arg: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "auto" else device_arg)
    if device_arg == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def get_inference_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name not in INFERENCE_DTYPES:
        raise ValueError(f"Unsupported dtype={dtype_name!r}; choose from {tuple(INFERENCE_DTYPES)}.")
    dtype = INFERENCE_DTYPES[dtype_name]
    if dtype != torch.float32 and device.type != "cuda":
        raise ValueError(f"--dtype {dtype_name} requires a CUDA device; got {device}.")
    if dtype == torch.bfloat16:
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_supported:
            raise RuntimeError("--dtype bf16 requires a CUDA GPU with BF16 support.")
    return dtype


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU prediction requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), local_rank, world_size
    return build_device(args.device), local_rank, world_size


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def parse_pairs_file(dataset_root: str | Path, pairs_file: str = "test_pairs.txt") -> list[tuple[str, list[str]]]:
    dataset_root = Path(dataset_root)
    path = dataset_root / pairs_file
    if not path.exists():
        raise FileNotFoundError(path)
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1:]))
    if not pairs:
        raise ValueError(f"No valid pairs found in {path}")
    return pairs


def _image_key(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts[1:] if path.parts and path.parts[0] == "img" else path.parts
    if not parts:
        return path.stem
    return "_".join([*parts[:-1], Path(parts[-1]).stem])


def _pair_output_name(source_rel: str, target_rel: str) -> str:
    return f"{_image_key(source_rel)}_to_{_image_key(target_rel)}.png"


def _output_is_complete(path: str | Path) -> bool:
    """Return whether a previously written prediction can safely be reused."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    # A killed process can leave a truncated PNG. Verify it before skipping.
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


class PredictionPairDataset(Dataset):
    def __init__(self, dataset_root: str | Path, pairs_file: str, resolution: int):
        self.dataset_root = Path(dataset_root)
        self.resolution = int(resolution)
        self.skipped_missing_pose = 0
        self.items = []

        for target_rel, source_rel_list in parse_pairs_file(self.dataset_root, pairs_file):
            target_pose_path = _densepose_path_from_image(self.dataset_root, target_rel)
            if not target_pose_path.exists():
                self.skipped_missing_pose += len(source_rel_list)
                continue
            for source_rel in source_rel_list:
                self.items.append(
                    {
                        "source_path": self.dataset_root / source_rel,
                        "pose_path": target_pose_path,
                        "output_name": _pair_output_name(source_rel, target_rel),
                    }
                )
        if not self.items:
            raise ValueError(f"No valid prediction pairs found in {self.dataset_root / pairs_file}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        source = _load_image(item["source_path"], self.resolution)
        pose = load_pose_map(item["pose_path"], size=(self.resolution, self.resolution))
        return {
            "source": source,
            "pose": pose,
            "output_name": item["output_name"],
        }


def build_diffusion_model(args: argparse.Namespace) -> nn.Module:
    return PoseTransferUNet(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        pose_channels=3,
    )


def build_models(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module, GaussianDiffusion]:
    dtype_name = getattr(args, "dtype", "fp32")
    weight_dtype = get_inference_dtype(dtype_name, device)
    autoencoder = DiffusersAutoencoder(
        name_or_path=args.sd_vae_name_or_path,
        scaling_factor=args.latent_scaling_factor,
    ).to(device=device, dtype=weight_dtype)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    args.latent_channels = int(getattr(autoencoder, "latent_channels", args.latent_channels))

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_diffusion_model(args).to(dtype=weight_dtype)
    state = ckpt.get("model") or ckpt
    model.load_state_dict(state, strict=True)
    del state, ckpt
    model = model.to(device).eval()

    betas = cosine_beta_schedule(args.timesteps) if args.schedule == "cosine" else linear_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, objective=args.diffusion_objective).to(device)
    return autoencoder, model, diffusion


@torch.no_grad()
def sample_from_inputs(
    autoencoder: DiffusersAutoencoder,
    model: nn.Module,
    diffusion: GaussianDiffusion,
    source: torch.Tensor,
    pose: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    weight_dtype = next(model.parameters()).dtype
    source = source.to(device=device, dtype=weight_dtype)
    pose = pose.to(device=device, dtype=weight_dtype)
    with autocast_context(device, weight_dtype):
        source_latent = autoencoder.encode(source, sample_posterior=False).latent
        cond = {"source_image": source, "source_latent": source_latent, "target_pose": pose}
        downsample_factor = int(getattr(autoencoder, "downsample_factor", 4))
        latent_h = max(source.shape[-2] // downsample_factor, 1)
        latent_w = max(source.shape[-1] // downsample_factor, 1)
        latent_shape = (source.shape[0], args.latent_channels, latent_h, latent_w)

        if args.sampler == "ddpm":
            sample_latent = diffusion.sample(
                model,
                latent_shape,
                cond,
                guidance_scale=1.0,
                source_guidance_scale=args.source_guidance_scale,
                pose_guidance_scale=args.pose_guidance_scale,
                device=device,
                show_progress=bool(args.pose) and not bool(args.dataset_root) and is_main_process(),
                progress_desc="ddpm sampling",
            )
        else:
            sample_latent = diffusion.ddim_sample(
                model,
                latent_shape,
                cond,
                steps=args.steps,
                eta=0.0,
                guidance_scale=1.0,
                source_guidance_scale=args.source_guidance_scale,
                pose_guidance_scale=args.pose_guidance_scale,
                device=device,
                show_progress=bool(args.pose) and not bool(args.dataset_root) and is_main_process(),
                progress_desc="ddim sampling",
            )
        return autoencoder.decode(sample_latent.to(dtype=weight_dtype))


def run_single(args: argparse.Namespace, autoencoder: DiffusersAutoencoder, model: nn.Module, diffusion: GaussianDiffusion, device: torch.device) -> None:
    if args.resume and _output_is_complete(args.output):
        if is_main_process():
            print(f"Skipping existing prediction: {args.output}")
        return
    source = _load_image(args.source, args.resolution).unsqueeze(0)
    pose = load_pose_map(args.pose, size=(args.resolution, args.resolution)).unsqueeze(0)
    sample = sample_from_inputs(autoencoder, model, diffusion, source, pose, args, device)
    save_tensor_image(sample, args.output, nrow=1)


def run_batch(
    args: argparse.Namespace,
    autoencoder: DiffusersAutoencoder,
    model: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    if not args.dataset_root:
        raise ValueError("Batch prediction requires --dataset_root.")
    out_dir = ensure_dir(args.output_dir)
    dataset = PredictionPairDataset(args.dataset_root, args.pairs_file, args.resolution)
    if is_main_process() and dataset.skipped_missing_pose > 0:
        print(f"Skipped {dataset.skipped_missing_pose} prediction pairs with missing densepose files.")

    if args.resume:
        pending_items = [
            item for item in dataset.items
            if not _output_is_complete(out_dir / str(item["output_name"]))
        ]
        skipped = len(dataset.items) - len(pending_items)
        dataset.items = pending_items
        if is_main_process() and skipped:
            print(f"Resume: skipped {skipped} existing predictions in {out_dir}.")
        if not dataset.items:
            if is_main_process():
                print("Resume: all predictions are already complete.")
            return

    indices = list(range(len(dataset)))[rank::world_size]
    dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    for batch in tqdm(loader, desc="predict"):
        source_batch = batch["source"]
        pose_batch = batch["pose"]
        samples = sample_from_inputs(autoencoder, model, diffusion, source_batch, pose_batch, args, device)
        for sample, output_name in zip(samples, batch["output_name"]):
            save_tensor_image(sample.unsqueeze(0), out_dir / output_name, nrow=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/new_diffusion/last.pt")
    parser.add_argument("--source", default="test.jpg")
    parser.add_argument("--pose", default="")
    parser.add_argument("--output", default="outputs/new_diffusion.png")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--pairs_file", default="test_pairs.txt")
    parser.add_argument("--output_dir", default="outputs/new_diffusion_test_pairs")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prediction files that already exist and pass PNG verification. Useful for resuming an interrupted batch.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=512, choices=[512])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=list(INFERENCE_DTYPES),
        default="fp32",
        help="Inference precision. fp16/bf16 require CUDA and keep model/VAE weights in 16-bit precision.",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--diffusion_objective", choices=["eps", "v"], default="eps")
    parser.add_argument("--sd_vae_name_or_path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--base_channels", type=int, default=256)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--source_guidance_scale", type=float, default=3.0)
    parser.add_argument("--pose_guidance_scale", type=float, default=3.0)
    args = parser.parse_args()
    device, rank, world_size = setup_distributed(args)
    set_seed(args.seed + rank if args.seed >= 0 else args.seed)
    autoencoder, model, diffusion = build_models(args, device)
    if is_main_process():
        print(f"Using device={device}, dtype={args.dtype}")

    if args.dataset_root:
        run_batch(args, autoencoder, model, diffusion, device, rank, world_size)
    else:
        if not args.pose:
            raise ValueError("Single-image prediction requires --pose.")
        if is_main_process():
            run_single(args, autoencoder, model, diffusion, device)

    cleanup_distributed()


if __name__ == "__main__":
    main()
