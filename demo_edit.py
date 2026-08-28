"""DensePose-guided masked appearance editing with the local diffusion model.

``--source`` donates clothing appearance. ``--reference`` is the target canvas
whose identity/background are retained. The 3-channel ``--pose`` DensePose and
the white ``--mask`` must both be aligned with the reference image.

Example::

    python demo_edit.py \
        --source donor.jpg \
        --reference target.jpg \
        --pose target_densepose.png \
        --mask target_upper_mask.png

During sampling, the known reference latent is re-noised and restored outside
the edit mask after every scheduler step. This is the optional editing branch
used by PoCoLD; it is separate from how DensePose is injected into the UNet.
Pass ``--refine`` to run the independently trained RGB Refiner afterward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, type=Path, help="Appearance donor image.")
    parser.add_argument("--reference", required=True, type=Path, help="Target image to edit.")
    parser.add_argument(
        "--pose",
        required=True,
        type=Path,
        help="Three-channel target DensePose image or HWC .npy array.",
    )
    parser.add_argument(
        "--mask",
        required=True,
        type=Path,
        help="Target-space grayscale mask: white is edited, black is preserved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "new_diffusion" / "last.pt",
    )
    parser.add_argument("--refine", action="store_true", help="Post-process the diffusion result with Refiner.")
    parser.add_argument(
        "--refiner-checkpoint",
        "--refiner_checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "refiner" / "last.pt",
        help="Used only when --refine is enabled.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "appearance_edit.png")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=ROOT / "outputs" / "appearance_edit_comparison.png",
    )
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--resolution", type=int, choices=(256, 512), default=512)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--steps", type=int, default=50, help="DDIM inference steps.")
    parser.add_argument("--source-guidance-scale", type=float, default=3.0)
    parser.add_argument("--pose-guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")

    # These must match the checkpoint training configuration.
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--diffusion-objective", choices=("eps", "v"), default="eps")
    parser.add_argument("--sd-vae-name-or-path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--latent-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the four inputs without loading the VAE or checkpoint.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def read_mask_array(path: Path, resolution: int | None = None) -> Any:
    import numpy as np
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = image.convert("L")
        if resolution is not None:
            image = image.resize((resolution, resolution), resampling.NEAREST)
        array = np.asarray(image, dtype=np.uint8)
    mask = array >= 128
    coverage = float(mask.mean())
    if coverage <= 0.0 or coverage >= 1.0:
        raise ValueError(f"Mask must contain both black and white pixels: {path}")
    return mask


def validate_inputs(source: Path, reference: Path, pose: Path, mask: Path) -> None:
    import numpy as np
    from PIL import Image

    for label, path in (("source", source), ("reference", reference)):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported {label} image extension: {path.suffix}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            print(f"{label.capitalize()} image: {image.size[0]}x{image.size[1]} {image.mode}")

    if pose.suffix.lower() in IMAGE_EXTENSIONS:
        with Image.open(pose) as image:
            if len(image.getbands()) < 3:
                raise ValueError(f"DensePose image must have three channels: {pose}")
            print(f"DensePose: {image.size[0]}x{image.size[1]} {image.mode} -> RGB")
    elif pose.suffix.lower() == ".npy":
        pose_array = np.load(pose, mmap_mode="r")
        if pose_array.ndim != 3 or pose_array.shape[2] != 3:
            raise ValueError(f"Expected DensePose shape (H, W, 3), got {pose_array.shape} in {pose}")
        print(f"DensePose: {pose_array.shape}, dtype={pose_array.dtype}")
    else:
        raise ValueError(f"Unsupported DensePose format: {pose.suffix}")

    coverage = float(read_mask_array(mask).mean())
    print(f"Edit-mask coverage: {coverage * 100:.1f}% (white = edit)")


def load_rgb_tensor(path: Path, resolution: int) -> Any:
    import torch
    from PIL import Image

    from new_diffusion.utils import resize_pil_image

    with Image.open(path) as image:
        image = resize_pil_image(image.convert("RGB"), (resolution, resolution))
        array = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32").copy())
    return array.permute(2, 0, 1).div(127.5).sub(1.0)


def load_mask_tensor(path: Path, resolution: int) -> Any:
    import numpy as np
    import torch

    mask = read_mask_array(path, resolution=resolution).astype(np.float32)
    return torch.from_numpy(mask)[None]


def tensor_to_image(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    array = array.add(1.0).mul(127.5).round().permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_image(mask: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = mask[0].detach().cpu().mul(255).numpy().astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def pose_to_image(pose: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = pose.detach().float().cpu().clamp(0.0, 1.0)
    array = array.mul(255).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_comparison(path: Path, panels: list[tuple[str, Any]], tile_size: int = 256) -> None:
    from PIL import Image, ImageDraw

    label_height = 28
    canvas = Image.new("RGB", (tile_size * len(panels), tile_size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * tile_size
        canvas.paste(image.resize((tile_size, tile_size)), (x, label_height))
        draw.text((x + 8, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if not 1 <= args.steps <= args.timesteps:
        raise ValueError("--steps must be between 1 and --timesteps")
    if args.source_guidance_scale < 0 or args.pose_guidance_scale < 0:
        raise ValueError("Guidance scales must be non-negative.")

    source_path = require_file(args.source, "Source image")
    reference_path = require_file(args.reference, "Reference image")
    pose_path = require_file(args.pose, "DensePose")
    mask_path = require_file(args.mask, "Edit mask")
    checkpoint_path = require_file(args.checkpoint, "Checkpoint")
    refiner_checkpoint_path = (
        require_file(args.refiner_checkpoint, "Refiner checkpoint") if args.refine else None
    )
    validate_inputs(source_path, reference_path, pose_path, mask_path)
    if args.validate_only:
        print(f"Checkpoint: {checkpoint_path}")
        if refiner_checkpoint_path is not None:
            print(f"Refiner checkpoint: {refiner_checkpoint_path}")
        print("Validation passed; model was not loaded.")
        return

    # NumPy must be imported before torch in this Windows environment to avoid
    # loading two Intel OpenMP runtimes in the opposite order.
    import numpy as np
    import torch
    import torch.nn.functional as F

    from new_diffusion.pose import load_pose_map
    from new_diffusion.predict import build_device, build_models, set_seed

    args.checkpoint = str(checkpoint_path)
    device = build_device(args.device)
    set_seed(args.seed)
    source = load_rgb_tensor(source_path, args.resolution).unsqueeze(0).to(device)
    reference = load_rgb_tensor(reference_path, args.resolution).unsqueeze(0).to(device)
    target_pose = load_pose_map(
        pose_path, size=(args.resolution, args.resolution)
    ).unsqueeze(0).to(device)
    if target_pose.shape[1] != 3:
        raise ValueError(f"DensePose must have 3 channels, got {target_pose.shape[1]}.")
    pixel_mask = load_mask_tensor(mask_path, args.resolution).unsqueeze(0).to(device)

    print(f"Device: {device}")
    print(f"Loading diffusion model: {checkpoint_path}")
    autoencoder, model, diffusion = build_models(args, device)
    refiner_model = None
    refiner_config = None
    if refiner_checkpoint_path is not None:
        from Refiner.predict_refiner import build_model_from_checkpoint, predict_batch
        from new_diffusion.utils import load_checkpoint

        print(f"Loading Refiner model: {refiner_checkpoint_path}")
        refiner_checkpoint = load_checkpoint(refiner_checkpoint_path, map_location=device)
        refiner_model, refiner_config = build_model_from_checkpoint(refiner_checkpoint, device)
        print("Loaded Refiner config: " + ", ".join(f"{key}={value}" for key, value in refiner_config.items()))

    with torch.inference_mode():
        source_latent = autoencoder.encode(source, sample_posterior=False).latent
        reference_latent = autoencoder.encode(reference, sample_posterior=False).latent
        latent_mask = F.interpolate(pixel_mask, size=reference_latent.shape[-2:], mode="nearest")
        cond = {
            "source_image": source,
            "source_latent": source_latent,
            "target_pose": target_pose,
        }
        sample_steps = args.steps if args.sampler == "ddim" else args.timesteps
        edited_latent = diffusion.diffusers_sample(
            model,
            tuple(reference_latent.shape),
            cond,
            sampler=args.sampler,
            steps=sample_steps,
            eta=0.0,
            source_guidance_scale=args.source_guidance_scale,
            pose_guidance_scale=args.pose_guidance_scale,
            device=device,
            show_progress=True,
            progress_desc=f"DensePose {args.sampler.upper()} edit",
            known_latent=reference_latent,
            edit_mask=latent_mask,
        )
        generated = autoencoder.decode(edited_latent).clamp(-1.0, 1.0)
        diffusion_result = generated * pixel_mask + reference * (1.0 - pixel_mask)

    result = diffusion_result.detach().cpu()
    if refiner_model is not None:
        print("Running Refiner post-processing...")
        diffusion_01 = diffusion_result.add(1.0).mul(0.5)
        refined_01 = predict_batch(refiner_model, diffusion_01, device, "no")
        reference_01 = reference.detach().cpu().add(1.0).mul(0.5)
        pixel_mask_cpu = pixel_mask.detach().cpu()
        refined_01 = refined_01 * pixel_mask_cpu + reference_01 * (1.0 - pixel_mask_cpu)
        result = refined_01.mul(2.0).sub(1.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_image = tensor_to_image(result[0])
    result_image.save(args.output)
    if not args.no_comparison:
        result_panels = []
        if refiner_model is not None:
            result_panels.append(("Diffusion result", tensor_to_image(diffusion_result[0])))
        result_panels.append(("Refined result" if args.refine else "Result", result_image))
        save_comparison(
            args.comparison_output,
            [
                ("Source appearance", tensor_to_image(source[0])),
                ("Reference target", tensor_to_image(reference[0])),
                ("White = edit", mask_to_image(pixel_mask[0])),
                ("3-channel DensePose", pose_to_image(target_pose[0])),
                *result_panels,
            ],
        )

    metadata = {
        "backend": "new_diffusion (3-channel DensePose)",
        "source": str(source_path),
        "reference": str(reference_path),
        "densepose": str(pose_path),
        "mask": str(mask_path),
        "checkpoint": str(checkpoint_path),
        "refine": args.refine,
        "refiner_checkpoint": str(refiner_checkpoint_path) if refiner_checkpoint_path is not None else None,
        "refiner_config": refiner_config,
        "resolution": args.resolution,
        "sampler": args.sampler,
        "steps": sample_steps,
        "source_guidance_scale": args.source_guidance_scale,
        "pose_guidance_scale": args.pose_guidance_scale,
        "seed": args.seed,
        "mask_semantics": "white=edit, black=preserve reference",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Result:     {args.output.resolve()}")
    if not args.no_comparison:
        print(f"Comparison: {args.comparison_output.resolve()}")
    print(f"Metadata:   {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
