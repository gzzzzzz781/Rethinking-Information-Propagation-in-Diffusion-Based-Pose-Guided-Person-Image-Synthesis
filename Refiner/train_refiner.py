from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .refiner_model import build_refiner
from metrics import (
    build_fid_pocold_style_model,
    build_lpips_pocold_style_model,
    compute_lpips_pocold_style_from_tensors,
    compute_psnr_pocold_style_from_tensors,
    compute_ssim_pocold_style_from_tensors,
    extract_fid_features_pocold_style_from_tensors,
    frechet_distance,
    list_images,
    load_image_tensor,
)
from new_diffusion.utils import ensure_dir, load_checkpoint, save_tensor_image


METRIC_SIZES = {
    256: (256, 176),
    512: (512, 352),
}


def build_device(device_arg: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and device_arg == "auto" else device_arg)
    if device_arg == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    return device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), local_rank
    return build_device(args.device), local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_data_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def metric_size_for_resolution(resolution: int) -> tuple[int, int]:
    try:
        return METRIC_SIZES[resolution]
    except KeyError as exc:
        raise ValueError("Refiner training supports --resolution 256 or --resolution 512.") from exc


def parse_pairs_file(dataset_root: str | Path, pairs_file: str = "train_pairs.txt") -> list[tuple[str, list[str]]]:
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


def pair_output_name(source_rel: str, target_rel: str) -> str:
    return f"{_image_key(source_rel)}_to_{_image_key(target_rel)}.png"


def resolve_fid_real_dir(
    dataset_root: str | Path,
    fid_real_dir: str,
    metric_size: tuple[int, int],
) -> Path:
    if fid_real_dir:
        path = Path(fid_real_dir)
    else:
        height, width = metric_size
        path = Path(dataset_root) / f"fid_real_{height}x{width}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def update_fid_feature_stats(
    features: torch.Tensor,
    count: int,
    feature_sum: torch.Tensor | None,
    feature_square_sum: torch.Tensor | None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    features = features.double()
    if feature_sum is None:
        feature_sum = features.sum(dim=0)
        feature_square_sum = features.t().matmul(features)
    else:
        feature_sum = feature_sum + features.sum(dim=0)
        feature_square_sum = feature_square_sum + features.t().matmul(features)
    return count + int(features.shape[0]), feature_sum, feature_square_sum


def compute_feature_stats_from_sums(
    count: int,
    feature_sum: torch.Tensor,
    feature_square_sum: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 1:
        raise ValueError("FID requires at least two prediction features.")
    mean = feature_sum / float(count)
    cov = (feature_square_sum - float(count) * torch.outer(mean, mean)) / float(count - 1)
    return mean.cpu().numpy(), cov.cpu().numpy()


@torch.no_grad()
def compute_fid_real_stats_distributed(
    paths: list[Path],
    extractor: nn.Module,
    batch_size: int,
    device: torch.device,
    eval_size: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("FID real statistics require non-empty image paths.")

    if is_distributed():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_paths = paths[rank::world_size]
    else:
        local_paths = paths

    batch_size = max(1, batch_size)
    feature_count = 0
    feature_sum = None
    feature_square_sum = None
    pbar = tqdm(
        range(0, len(local_paths), batch_size),
        desc="fid real",
        leave=False,
        disable=not is_main_process(),
    )
    for start in pbar:
        batch_paths = local_paths[start:start + batch_size]
        batch = torch.stack([load_image_tensor(path, size=eval_size) for path in batch_paths], dim=0).to(device)
        features = extractor(batch).detach()
        feature_count, feature_sum, feature_square_sum = update_fid_feature_stats(
            features,
            feature_count,
            feature_sum,
            feature_square_sum,
        )

    if feature_sum is None or feature_square_sum is None:
        feature_sum = torch.zeros(2048, device=device, dtype=torch.float64)
        feature_square_sum = torch.zeros(2048, 2048, device=device, dtype=torch.float64)

    if is_distributed():
        count_tensor = torch.tensor([float(feature_count)], device=device, dtype=torch.float64)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        feature_count = int(count_tensor.item())
        dist.all_reduce(feature_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_square_sum, op=dist.ReduceOp.SUM)

    return compute_feature_stats_from_sums(feature_count, feature_sum, feature_square_sum)


def load_rgb01_image(path: str | Path, resolution: tuple[int, int]) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms import functional as TF

    height, width = resolution
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return TF.to_tensor(image)


def build_lpips_loss(device: torch.device, net: str) -> nn.Module:
    try:
        import lpips
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "LPIPS is not installed. Install the `lpips` package or set --lpips_weight 0."
        ) from exc

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    for param in loss_fn.parameters():
        param.requires_grad_(False)
    return loss_fn


def set_optimizer_lr(optimizer: AdamW, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr
    warmup_steps = max(int(warmup_steps), 0)
    step = max(int(step), 0)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return base_lr

    decay_steps = max(total_steps - warmup_steps - 1, 1)
    decay_progress = min(step - warmup_steps, decay_steps) / float(decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


class DiffusionRefinerDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        pred_dir: str | Path,
        pairs_file: str = "train_pairs.txt",
        resolution: tuple[int, int] = (256, 256),
        random_hflip: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.pred_dir = Path(pred_dir)
        self.resolution = resolution
        self.random_hflip = random_hflip
        self.skipped_missing_pred = 0
        self.skipped_missing_gt = 0
        self.items = self._build_items(pairs_file)
        if not self.items:
            raise ValueError(f"No valid refiner training pairs found under {self.pred_dir}")

    def _build_items(self, pairs_file: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for target_rel, source_rel_list in parse_pairs_file(self.dataset_root, pairs_file):
            gt_path = self.dataset_root / target_rel
            for source_rel in source_rel_list:
                pred_path = self.pred_dir / pair_output_name(source_rel, target_rel)
                missing_paths = []
                if not pred_path.exists():
                    self.skipped_missing_pred += 1
                    missing_paths.append(pred_path)
                if not gt_path.exists():
                    self.skipped_missing_gt += 1
                    missing_paths.append(gt_path)
                if missing_paths:
                    continue
                items.append(
                    {
                        "pred_path": str(pred_path),
                        "gt_path": str(gt_path),
                        "name": pred_path.name,
                    }
                )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        pred = load_rgb01_image(item["pred_path"], self.resolution)
        gt = load_rgb01_image(item["gt_path"], self.resolution)
        if self.random_hflip and torch.rand(()) < 0.5:
            pred = torch.flip(pred, dims=[2])
            gt = torch.flip(gt, dims=[2])
        return {
            "pred": pred,
            "gt": gt,
            "name": item["name"],
        }


def save_preview(pred_in: torch.Tensor, refined: torch.Tensor, gt: torch.Tensor, path: str | Path) -> None:
    preview = torch.cat(
        [
            pred_in[:1].mul(2.0).sub(1.0),
            refined[:1].mul(2.0).sub(1.0),
            gt[:1].mul(2.0).sub(1.0),
        ],
        dim=0,
    )
    save_tensor_image(preview, path, nrow=3)


def build_model(args: argparse.Namespace) -> nn.Module:
    return build_refiner(
        model_dim=args.model_dim,
        num_blocks=tuple(args.num_blocks),
        num_refinement_blocks=args.num_refinement_blocks,
        heads=tuple(args.heads),
        ffn_expansion_factor=args.ffn_expansion_factor,
        model_bias=args.model_bias,
        window_size=args.window_size,
        residual_scale=args.residual_scale,
    )


@torch.no_grad()
def evaluate_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lpips_metric: nn.Module | None,
    fid_extractor: nn.Module | None,
    fid_real_stats: tuple[np.ndarray, np.ndarray] | None,
    metric_size: tuple[int, int],
) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_input_psnr = 0.0
    total_psnr = 0.0
    total_ssim_256 = 0.0
    total_lpips = 0.0
    total_images = 0
    total_elements = 0
    has_lpips = lpips_metric is not None
    has_fid = fid_extractor is not None
    fid_feature_count = 0
    fid_feature_sum = None
    fid_feature_square_sum = None

    for batch in tqdm(loader, desc="refiner val", leave=False):
        pred_in = batch["pred"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        refined = model(pred_in)
        batch_images = int(gt.shape[0])

        diff = (refined.float() - gt.float()).abs()
        total_l1 += diff.sum().item()
        total_elements += int(diff.numel())
        total_images += batch_images

        total_input_psnr += batch_images * compute_psnr_pocold_style_from_tensors(pred_in.float(), gt.float(), eval_size=metric_size)
        total_psnr += batch_images * compute_psnr_pocold_style_from_tensors(refined.float(), gt.float(), eval_size=metric_size)
        _, ssim_256 = compute_ssim_pocold_style_from_tensors(refined.float(), gt.float(), eval_size=metric_size)
        total_ssim_256 += batch_images * ssim_256
        if has_lpips:
            total_lpips += batch_images * compute_lpips_pocold_style_from_tensors(
                refined.float(),
                gt.float(),
                metric=lpips_metric,
                eval_size=metric_size,
            )
        if has_fid:
            fid_features = extract_fid_features_pocold_style_from_tensors(
                refined.float(),
                extractor=fid_extractor,
                eval_size=metric_size,
            )
            fid_feature_count, fid_feature_sum, fid_feature_square_sum = update_fid_feature_stats(
                fid_features,
                fid_feature_count,
                fid_feature_sum,
                fid_feature_square_sum,
            )

    model.train()
    if is_distributed():
        stats = torch.tensor(
            [
                total_l1,
                float(total_input_psnr),
                float(total_psnr),
                float(total_ssim_256),
                float(total_lpips),
                float(total_images),
                float(total_elements),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_l1 = float(stats[0].item())
        total_input_psnr = float(stats[1].item())
        total_psnr = float(stats[2].item())
        total_ssim_256 = float(stats[3].item())
        total_lpips = float(stats[4].item())
        total_images = int(stats[5].item())
        total_elements = int(stats[6].item())
        if has_fid:
            fid_count_tensor = torch.tensor([float(fid_feature_count)], device=device, dtype=torch.float64)
            dist.all_reduce(fid_count_tensor, op=dist.ReduceOp.SUM)
            fid_feature_count = int(fid_count_tensor.item())
            dist.all_reduce(fid_feature_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(fid_feature_square_sum, op=dist.ReduceOp.SUM)
    if total_images == 0 or total_elements == 0:
        raise ValueError("Validation loader is empty.")
    input_psnr = total_input_psnr / float(total_images)
    psnr = total_psnr / float(total_images)
    metrics = {
        "val_l1": total_l1 / float(total_elements),
        "input_psnr": input_psnr,
        "psnr": psnr,
        "psnr_gain": psnr - input_psnr,
        "ssim_256": total_ssim_256 / float(total_images),
    }
    if has_lpips:
        metrics["lpips"] = total_lpips / float(total_images)
    if has_fid:
        fid_tensor = torch.full((1,), float("nan"), device=device, dtype=torch.float64)
        if is_main_process():
            if fid_real_stats is None:
                raise ValueError("FID real statistics are required on the main process.")
            mu_pred, sigma_pred = compute_feature_stats_from_sums(
                fid_feature_count,
                fid_feature_sum,
                fid_feature_square_sum,
            )
            fid_tensor[0] = frechet_distance(fid_real_stats[0], fid_real_stats[1], mu_pred, sigma_pred)
        if is_distributed():
            dist.broadcast(fid_tensor, src=0)
        metrics["fid"] = float(fid_tensor.item())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="dataset/deepfashion")
    parser.add_argument("--pred_dir", required=True, help="Directory of diffusion outputs on the training set.")
    parser.add_argument("--pairs_file", default="train_pairs.txt")
    parser.add_argument("--val_pred_dir", required=True, help="Directory of diffusion outputs on the validation/test set.")
    parser.add_argument("--val_pairs_file", default="test_pairs.txt")
    parser.add_argument("--fid_real_dir", default="", help="Directory of real images for validation FID. Defaults to the directory matching --resolution.")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=2, help="Defaults to --batch_size when <= 0.")
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lpips_weight", type=float, default=0.2)
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA algorithms. This may reduce training speed.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4, help="Defaults to --num_workers when < 0.")
    parser.add_argument("--checkpoint_dir", default="./checkpoints/refiner_restormer")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=500)
    parser.add_argument("--skip_val_fid", action="store_true")
    parser.add_argument("--model_dim", type=int, default=24)
    parser.add_argument("--num_blocks", type=int, nargs=4, default=[1, 2, 2, 4])
    parser.add_argument("--num_refinement_blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, nargs=4, default=[1, 2, 4, 8])
    parser.add_argument("--ffn_expansion_factor", type=float, default=2.0)
    parser.add_argument("--model_bias", action="store_true")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=0.005)
    parser.add_argument("--random_hflip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    device, local_rank = setup_distributed(args)
    seed_everything(args.seed, deterministic=args.deterministic)
    process_rank = dist.get_rank() if is_distributed() else 0
    resolution = (args.resolution, args.resolution)
    metric_size = metric_size_for_resolution(args.resolution)
    out_dir = ensure_dir(args.checkpoint_dir)
    if is_main_process() and is_distributed():
        print(
            f"Distributed training enabled: {dist.get_world_size()} GPUs, "
            f"batch_size={args.batch_size} per GPU, effective_batch_size={args.batch_size * dist.get_world_size()}."
        )

    dataset = DiffusionRefinerDataset(
        dataset_root=args.dataset_root,
        pred_dir=args.pred_dir,
        pairs_file=args.pairs_file,
        resolution=resolution,
        random_hflip=args.random_hflip,
    )
    if is_main_process() and dataset.skipped_missing_pred > 0:
        print(f"Skipped {dataset.skipped_missing_pred} training pairs with missing diffusion outputs.")
    if is_main_process() and dataset.skipped_missing_gt > 0:
        print(f"Skipped {dataset.skipped_missing_gt} training pairs with missing GT images.")

    train_sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed)
        if is_distributed()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_data_generator(args.seed + process_rank),
    )

    val_dataset = DiffusionRefinerDataset(
        dataset_root=args.dataset_root,
        pred_dir=args.val_pred_dir,
        pairs_file=args.val_pairs_file,
        resolution=resolution,
        random_hflip=False,
    )
    if is_main_process() and val_dataset.skipped_missing_pred > 0:
        print(f"Skipped {val_dataset.skipped_missing_pred} validation pairs with missing diffusion outputs.")
    if is_main_process() and val_dataset.skipped_missing_gt > 0:
        print(f"Skipped {val_dataset.skipped_missing_gt} validation pairs with missing GT images.")

    val_batch_size = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_num_workers = args.val_num_workers if args.val_num_workers >= 0 else args.num_workers
    if is_distributed():
        rank_indices = range(dist.get_rank(), len(val_dataset), dist.get_world_size())
        val_data = Subset(val_dataset, rank_indices)
    else:
        val_data = val_dataset
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=val_num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=build_data_generator(args.seed + process_rank + 1),
    )

    model = build_model(args).to(device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.L1Loss()
    train_lpips_criterion = None
    if args.lpips_weight > 0:
        train_lpips_criterion = build_lpips_loss(device, args.lpips_net)
    val_lpips_metric = build_lpips_pocold_style_model(device=device, net_type="alex")
    fid_extractor = None
    fid_real_stats = None
    if not args.skip_val_fid:
        fid_extractor = build_fid_pocold_style_model(device=device)
        fid_real_dir = resolve_fid_real_dir(args.dataset_root, args.fid_real_dir, metric_size)
        fid_real_paths = list_images(fid_real_dir)
        if not fid_real_paths:
            raise ValueError(f"No FID real images found in {fid_real_dir}")
        if is_main_process():
            suffix = f" across {dist.get_world_size()} ranks" if is_distributed() else ""
            print(f"Computing FID real statistics from {len(fid_real_paths)} images in {fid_real_dir}{suffix}.")
        fid_real_stats = compute_fid_real_stats_distributed(
            fid_real_paths,
            extractor=fid_extractor,
            batch_size=args.fid_batch_size,
            device=device,
            eval_size=metric_size,
        )
    optimizer = AdamW(unwrap_model(model).parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(loader), 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
    set_optimizer_lr(
        optimizer,
        compute_lr(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps),
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"refiner epoch {epoch + 1}/{args.epochs}", disable=not is_main_process())
        for batch in pbar:
            pred_in = batch["pred"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            refined = model(pred_in)
            l1_loss = criterion(refined, gt)
            loss = l1_loss
            lpips_loss = None
            if train_lpips_criterion is not None:
                refined_lpips = refined.mul(2.0).sub(1.0)
                gt_lpips = gt.mul(2.0).sub(1.0)
                lpips_loss = train_lpips_criterion(refined_lpips, gt_lpips).mean()
                loss = loss + args.lpips_weight * lpips_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            set_optimizer_lr(
                optimizer,
                compute_lr(global_step, total_steps, args.lr, args.min_lr, args.warmup_steps),
            )
            if is_main_process():
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "l1": f"{l1_loss.item():.4f}",
                }
                if lpips_loss is not None:
                    postfix["lpips"] = f"{lpips_loss.item():.4f}"
                pbar.set_postfix(postfix)

            if is_main_process() and args.preview_every > 0 and global_step % args.preview_every == 0:
                save_preview(pred_in, refined, gt, out_dir / f"preview_{global_step:06d}.png")

        val_metrics = evaluate_metrics(
            unwrap_model(model),
            val_loader,
            device,
            val_lpips_metric,
            fid_extractor,
            fid_real_stats,
            metric_size,
        )
        if is_main_process():
            log_items = [
                f"val_l1={val_metrics['val_l1']:.6f}",
                f"input_psnr={val_metrics['input_psnr']:.4f}",
                f"psnr={val_metrics['psnr']:.4f}",
                f"psnr_gain={val_metrics['psnr_gain']:.4f}",
                f"ssim_256={val_metrics['ssim_256']:.4f}",
            ]
            if "lpips" in val_metrics:
                log_items.insert(4, f"lpips={val_metrics['lpips']:.4f}")
            if "fid" in val_metrics:
                log_items.append(f"fid={val_metrics['fid']:.4f}")
            print(f"epoch {epoch + 1}: " + ", ".join(log_items))

        if is_main_process():
            ckpt = {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
                **val_metrics,
            }
            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt, out_dir / f"epoch_{epoch + 1}.pt")
            torch.save(ckpt, out_dir / "last.pt")
        if is_distributed():
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
