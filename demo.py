from __future__ import annotations

import argparse
from pathlib import Path

# NumPy must load before torch in this Windows environment to avoid loading
# two Intel OpenMP runtimes in the opposite order.
import numpy  # noqa: F401
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from new_diffusion.utils import load_checkpoint, resize_pil_image, save_tensor_image


def load_source_image(path: str | Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = resize_pil_image(image, (resolution, resolution))
    return TF.to_tensor(image).mul(2.0).sub(1.0)


def save_rgb01_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    TF.to_pil_image(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diffusion with optional Refiner post-processing.")
    parser.add_argument("--source", required=True, help="Source/person image.")
    parser.add_argument("--pose", required=True, help="Target DensePose image.")
    parser.add_argument("--diffusion_checkpoint", required=True)
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument(
        "--refine",
        "-refine",
        dest="refine",
        action="store_true",
        help="Enable Refiner post-processing (default).",
    )
    refine_group.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
        help="Save only the diffusion output.",
    )
    parser.set_defaults(refine=True)
    parser.add_argument(
        "--refiner_checkpoint",
        "--refiner-checkpoint",
        default="checkpoints/refiner/last.pt",
        help="Used only when Refiner post-processing is enabled.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        choices=[256, 512],
        help="Refiner resolution when enabled. Diffusion always runs at 512x512.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--diffusion_output", default="outputs/demo_diffusion.png")
    parser.add_argument("--refined_output", default="outputs/demo_refined.png")

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--diffusion_objective", choices=["eps", "v"], default="eps")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--source_guidance_scale", type=float, default=2.0)
    parser.add_argument("--pose_guidance_scale", type=float, default=2.0)
    parser.add_argument("--sd_vae_name_or_path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--base_channels", type=int, default=256)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_scaling_factor", type=float, default=0.18215)

    args = parser.parse_args()
    if args.refine and not Path(args.refiner_checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"Refiner checkpoint does not exist: {args.refiner_checkpoint}")

    from new_diffusion.pose import load_pose_map
    from new_diffusion.predict import (
        build_device,
        build_models as build_diffusion_models,
        sample_from_inputs,
        set_seed,
    )

    device = build_device(args.device)
    set_seed(args.seed)

    diffusion_resolution = 512
    source = load_source_image(args.source, diffusion_resolution).unsqueeze(0)
    pose = load_pose_map(args.pose, size=(diffusion_resolution, diffusion_resolution)).unsqueeze(0)

    diffusion_args = argparse.Namespace(
        checkpoint=args.diffusion_checkpoint,
        sd_vae_name_or_path=args.sd_vae_name_or_path,
        latent_scaling_factor=args.latent_scaling_factor,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        schedule=args.schedule,
        diffusion_objective=args.diffusion_objective,
        sampler=args.sampler,
        steps=args.steps,
        source_guidance_scale=args.source_guidance_scale,
        pose_guidance_scale=args.pose_guidance_scale,
        pose=args.pose,
        dataset_root="",
    )

    print(f"Using device: {device}")
    print("Loading diffusion model...")
    autoencoder, diffusion_model, diffusion = build_diffusion_models(diffusion_args, device)

    print("Running diffusion inference...")
    with torch.no_grad():
        diffusion_output = sample_from_inputs(
            autoencoder,
            diffusion_model,
            diffusion,
            source,
            pose,
            diffusion_args,
            device,
        )
    save_tensor_image(diffusion_output, args.diffusion_output, nrow=1)

    print(f"Diffusion output: {args.diffusion_output}")
    if args.refine:
        from Refiner.predict_refiner import build_model_from_checkpoint, predict_batch

        print("Loading Refiner model...")
        refiner_checkpoint = load_checkpoint(args.refiner_checkpoint, map_location=device)
        refiner_model, refiner_config = build_model_from_checkpoint(refiner_checkpoint, device)
        print("Loaded Refiner config: " + ", ".join(f"{key}={value}" for key, value in refiner_config.items()))

        print("Running Refiner inference...")
        diffusion_01 = (diffusion_output.clamp(-1.0, 1.0) + 1.0) * 0.5
        if args.resolution != diffusion_resolution:
            diffusion_01 = TF.resize(
                diffusion_01,
                [args.resolution, args.resolution],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        refined_output = predict_batch(refiner_model, diffusion_01, device, "no")
        save_rgb01_image(refined_output[0], args.refined_output)
        print(f"Refined output:   {args.refined_output}")
    else:
        print("Refiner post-processing: disabled")


if __name__ == "__main__":
    main()
