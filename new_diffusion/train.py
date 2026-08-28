from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .autoencoder import DiffusersAutoencoder
from .dataset import PoseTransferDataset
from .diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from .unet import PoseTransferUNet
from .utils import ensure_dir, load_checkpoint, save_tensor_image


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


def reduce_mean(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = value.detach()
    if not is_distributed():
        return value
    reduced = value.to(device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()


def get_weight_dtype() -> torch.dtype:
    return torch.float32


def autocast_context():
    return nullcontext()


def build_grad_scaler():
    enabled = False
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
        dist.barrier()
        dist.destroy_process_group()


def build_epoch_cosine_scheduler(
    optimizer: AdamW,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
) -> LambdaLR:
    total_epochs = max(int(total_epochs), 1)
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int) -> float:
        if total_epochs <= 1:
            return 1.0
        progress = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi, dtype=torch.float32)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def set_diffusion_lr(
    optimizer: AdamW,
    base_lr: float,
    global_step: int,
    epoch: int,
    total_epochs: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.01,
) -> float:
    warmup_steps = max(int(warmup_steps), 1)
    if global_step < warmup_steps:
        lr_scale = float(global_step + 1) / float(warmup_steps)
    else:
        total_epochs = max(int(total_epochs), 1)
        if total_epochs <= 1:
            lr_scale = 1.0
        else:
            progress = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi, dtype=torch.float32)).item())
            lr_scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    lr = base_lr * lr_scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def load_autoencoder(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return DiffusersAutoencoder(
        name_or_path=args.sd_vae_name_or_path,
        scaling_factor=args.latent_scaling_factor,
    ).to(device)


def build_diffusion_model(args: argparse.Namespace) -> nn.Module:
    return PoseTransferUNet(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        pose_channels=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./dataset/deepfashion")
    parser.add_argument("--resolution", type=int, default=512, choices=[512])
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/new_diffusion")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--diffusion_objective", choices=["eps", "v"], default="eps")
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--noise_offset", type=float, default=0.05)
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sd_vae_name_or_path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--base_channels", type=int, default=256)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--source_drop_prob", type=float, default=0.15)
    parser.add_argument("--pose_drop_prob", type=float, default=0.15)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_source_guidance_scale", type=float, default=3.0)
    parser.add_argument("--sample_pose_guidance_scale", type=float, default=3.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    device, local_rank = setup_distributed(args)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    weight_dtype = get_weight_dtype()
    out_dir = ensure_dir(args.checkpoint_dir)

    autoencoder = load_autoencoder(args, device).to(device=device, dtype=weight_dtype)
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    latent_downsample_factor = int(getattr(autoencoder, "downsample_factor", 4))
    args.latent_channels = int(getattr(autoencoder, "latent_channels", args.latent_channels))
    if is_main_process():
        print(
            "Autoencoder config: "
            f"type=sd, "
            f"latent_channels={args.latent_channels}, "
            f"downsample_factor={latent_downsample_factor}, "
            f"scaling_factor={args.latent_scaling_factor}, "
            f"mixed_precision=no, "
            f"weight_dtype={weight_dtype}"
        )

    model = build_diffusion_model(args).to(device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    betas = cosine_beta_schedule(args.timesteps) if args.schedule == "cosine" else linear_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, objective=args.diffusion_objective).to(device)
    optim = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scaler = build_grad_scaler()

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)

    diff_dataset = PoseTransferDataset(
        args.dataset_root,
        "train_pairs.txt",
        resolution=args.resolution,
    )
    if is_main_process() and getattr(diff_dataset, "skipped_missing_pose", 0) > 0:
        print(f"Skipped {diff_dataset.skipped_missing_pose} training pairs with missing densepose files.")
    diff_sampler = DistributedSampler(diff_dataset, shuffle=True) if is_distributed() else None
    diff_loader = DataLoader(
        diff_dataset,
        batch_size=args.batch_size,
        shuffle=diff_sampler is None,
        sampler=diff_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    for epoch in range(start_epoch, args.epochs):
        if isinstance(diff_loader.sampler, DistributedSampler):
            diff_loader.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(diff_loader, desc=f"diff epoch {epoch + 1}/{args.epochs}", disable=not is_main_process())
        for batch in pbar:
            lr = set_diffusion_lr(
                optim,
                base_lr=args.lr,
                global_step=global_step,
                epoch=epoch,
                total_epochs=args.epochs,
                warmup_steps=1000,
                min_lr_ratio=args.min_lr_ratio,
            )
            target = batch["target_image"].to(device, non_blocking=True)
            source = batch["source_image"].to(device, non_blocking=True)
            pose = batch["target_pose"].to(device, non_blocking=True)

            with torch.no_grad():
                with autocast_context():
                    target_latent = autoencoder.encode(target.to(dtype=weight_dtype), sample_posterior=False).latent
                    source_latent = autoencoder.encode(source.to(dtype=weight_dtype), sample_posterior=False).latent

            cond = {"source_image": source, "source_latent": source_latent, "target_pose": pose}
            with autocast_context():
                main_step = diffusion.training_step(
                    model,
                    target_latent,
                    cond,
                    source_drop_prob=args.source_drop_prob,
                    pose_drop_prob=args.pose_drop_prob,
                    snr_gamma=args.snr_gamma,
                    noise_offset=args.noise_offset,
                )
            main_loss = main_step["loss"]
            loss = main_loss.float()

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            global_step += 1
            loss_log = reduce_mean(loss, device)
            main_loss_log = reduce_mean(main_loss, device)
            pbar.set_postfix(
                loss=f"{loss_log.item():.4f}",
                diff=f"{main_loss_log.item():.4f}",
                lr=f"{lr:.2e}",
            )

            if args.sample_every > 0 and global_step % args.sample_every == 0 and is_main_process():
                with torch.no_grad(), autocast_context():
                    sample_model = unwrap_model(model)
                    sample_was_training = sample_model.training
                    sample_model.eval()
                    try:
                        sample_latent = diffusion.ddim_sample(
                            sample_model,
                            tuple(target_latent[:1].shape),
                            {"source_image": source[:1], "source_latent": source_latent[:1], "target_pose": pose[:1]},
                            steps=args.sample_steps,
                            eta=0.0,
                            guidance_scale=1.0,
                            source_guidance_scale=(
                                None if args.sample_source_guidance_scale < 0 else args.sample_source_guidance_scale
                            ),
                            pose_guidance_scale=(
                                None if args.sample_pose_guidance_scale < 0 else args.sample_pose_guidance_scale
                            ),
                            device=device,
                        )
                    finally:
                        if sample_was_training:
                            sample_model.train()
                    sample = autoencoder.decode(sample_latent.to(dtype=weight_dtype))
                ddim_preview = torch.cat([source[:1], target[:1], sample], dim=0)
                save_tensor_image(ddim_preview, out_dir / f"ddim_preview_{global_step:06d}.png", nrow=3)

        ckpt = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "sd_vae_name_or_path": args.sd_vae_name_or_path,
        }
        if is_main_process():
            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt, out_dir / f"epoch_{epoch + 1}.pt")
            torch.save(ckpt, out_dir / "last.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()
