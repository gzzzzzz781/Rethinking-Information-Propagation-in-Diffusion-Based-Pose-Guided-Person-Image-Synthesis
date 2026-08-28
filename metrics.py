from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torch.nn import functional as F
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from new_diffusion.utils import resize_pil_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class PairResolutionResult:
    pairs: list[tuple[Path, Path]]
    num_missing_predictions: int


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


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


def load_image_tensor(path: str | Path, size: int | tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        if isinstance(size, int):
            pil_size = (size, size)
        else:
            height, width = size
            pil_size = (width, height)
        img = resize_pil_image(img, pil_size)
    tensor = TF.to_tensor(img)
    return tensor


def match_image_pairs(pred_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    pred_files = {p.name: p for p in list_images(pred_dir)}
    gt_files = {p.name: p for p in list_images(gt_dir)}
    common = sorted(set(pred_files) & set(gt_files))
    if not common:
        raise ValueError("No matching filenames found between prediction and ground-truth directories.")
    return [(pred_files[name], gt_files[name]) for name in common]


def _candidate_prediction_names(target_rel: str, source_rel: str) -> list[str]:
    names = []
    target_name = Path(target_rel).name
    target_stem = Path(target_rel).stem
    source_stem = Path(source_rel).stem
    target_key = _image_key(target_rel)
    source_key = _image_key(source_rel)
    names.extend(
        [
            f"{source_key}_to_{target_key}.png",
            f"{source_key}_to_{target_key}_vis.png",
            f"{source_stem}_to_{target_stem}.png",
            f"{source_stem}_to_{target_stem}_vis.png",
            f"{source_stem}_2_{target_stem}.png",
            f"{source_stem}_2_{target_stem}_vis.png",
            f"{source_stem}-{target_stem}.png",
            f"{source_stem}-{target_stem}_vis.png",
            f"{source_stem}_{target_stem}.png",
            f"{source_stem}_{target_stem}_vis.png",
            f"{source_stem}_to_{target_stem}.jpg",
            f"{source_stem}_to_{target_stem}_vis.jpg",
            target_name,
            f"{target_stem}.png",
            f"{target_stem}_vis.png",
            f"{target_stem}.jpg",
            f"{target_stem}_vis.jpg",
            f"{target_stem}.jpeg",
        ]
    )
    return list(dict.fromkeys(names))


def _find_prediction_path(
    pred_files: dict[str, Path],
    pred_paths: list[Path],
    pred_pair_files: dict[tuple[str, str], Path],
    target_rel: str,
    source_rel: str,
) -> Path | None:
    for name in _candidate_prediction_names(target_rel, source_rel):
        if name in pred_files:
            return pred_files[name]

    target_norm = _normalize_text(_compact_key(target_rel))
    source_norm = _normalize_text(_compact_key(source_rel))
    for prefix in ("", "fashion"):
        pred_path = pred_pair_files.get((f"{prefix}{source_norm}", f"{prefix}{target_norm}"))
        if pred_path is not None:
            return pred_path

    target_stem = Path(target_rel).stem.lower()
    source_stem = Path(source_rel).stem.lower()
    target_key = _image_key(target_rel).lower()
    source_key = _image_key(source_rel).lower()
    target_compact = _compact_key(target_rel).lower()
    source_compact = _compact_key(source_rel).lower()
    for path in pred_paths:
        name = path.stem.lower()
        name_norm = _normalize_text(name)
        if source_key in name and target_key in name:
            return path
        source_pos = name.find(source_stem)
        target_pos = name.find(target_stem)
        if source_pos >= 0 and target_pos > source_pos:
            return path
        source_pos = name.find(source_compact)
        target_pos = name.find(target_compact)
        if source_pos >= 0 and target_pos > source_pos:
            return path
        source_pos = name_norm.find(source_norm)
        target_pos = name_norm.find(target_norm)
        if source_pos >= 0 and target_pos > source_pos:
            return path
    return None


def _image_key(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts[1:] if path.parts and path.parts[0] == "img" else path.parts
    if not parts:
        return path.stem
    return "_".join([*parts[:-1], Path(parts[-1]).stem])


def _compact_key(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts[1:] if path.parts and path.parts[0] == "img" else path.parts
    if not parts:
        return path.stem
    return "".join([*parts[:-1], Path(parts[-1]).stem])


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _prediction_pair_index(pred_paths: list[Path]) -> dict[tuple[str, str], Path]:
    index = {}
    for path in pred_paths:
        stem = path.stem
        if stem.endswith("_vis"):
            stem = stem[:-4]
        if "_2_" not in stem:
            continue
        source_part, target_part = stem.split("_2_", 1)
        key = (_normalize_text(source_part), _normalize_text(target_part))
        index.setdefault(key, path)
    return index


def build_pairs_from_test_pairs(
    pred_dir: str | Path,
    dataset_root: str | Path,
    pairs_file: str = "test_pairs.txt",
    gt_subdir: str = "img",
) -> list[tuple[Path, Path]]:
    return resolve_pairs_from_test_pairs(
        pred_dir=pred_dir,
        dataset_root=dataset_root,
        pairs_file=pairs_file,
        gt_subdir=gt_subdir,
    ).pairs


def resolve_pairs_from_test_pairs(
    pred_dir: str | Path,
    dataset_root: str | Path,
    pairs_file: str = "test_pairs.txt",
    gt_subdir: str = "img",
) -> PairResolutionResult:
    pred_dir = Path(pred_dir)
    dataset_root = Path(dataset_root)
    pred_paths = list_images(pred_dir)
    pred_files = {p.name: p for p in pred_paths}
    pred_pair_files = _prediction_pair_index(pred_paths)
    pairs_meta = parse_pairs_file(dataset_root, pairs_file=pairs_file)

    resolved_pairs = []
    missing = []
    for target_rel, source_rel_list in pairs_meta:
        gt_path = dataset_root / target_rel
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        for source_rel in source_rel_list:
            candidates = _candidate_prediction_names(target_rel, source_rel)
            pred_path = _find_prediction_path(pred_files, pred_paths, pred_pair_files, target_rel, source_rel)
            if pred_path is None:
                missing.append((f"{source_rel} -> {target_rel}", candidates[:3]))
                continue
            resolved_pairs.append((pred_path, gt_path))

    if not resolved_pairs:
        preview = "\n".join([f"{target} -> {cands}" for target, cands in missing[:5]])
        raise ValueError(
            "No prediction files could be matched from test_pairs.txt.\n"
            f"Examples of attempted names:\n{preview}"
        )
    return PairResolutionResult(
        pairs=resolved_pairs,
        num_missing_predictions=len(missing),
    )


def collect_real_image_paths_from_pairs(
    dataset_root: str | Path,
    pairs_file: str = "train_pairs.txt",
) -> list[Path]:
    dataset_root = Path(dataset_root)
    pairs_meta = parse_pairs_file(dataset_root, pairs_file=pairs_file)
    paths = []
    seen = set()
    for target_rel, source_rel_list in pairs_meta:
        for rel in [target_rel, *source_rel_list]:
            path = dataset_root / rel
            if not path.exists():
                raise FileNotFoundError(path)
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                paths.append(path)
    if not paths:
        raise ValueError(f"No real images found from {dataset_root / pairs_file}")
    return paths


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma), persistent=False)

    def _create_window(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        return kernel.view(1, 1, size, size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        window = self.window.to(device=x.device, dtype=x.dtype).expand(c, 1, -1, -1)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=c)
        mu_y = F.conv2d(y, window, padding=pad, groups=c)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

        c1 = 0.01**2
        c2 = 0.03**2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        return ssim_map.mean(dim=(1, 2, 3))


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, 1.0)
        return self.model(x)


class PoCoLDInceptionFeatureExtractor(nn.Module):
    """Inception feature extractor matching PoCoLD's FID preprocessing."""

    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, transform_input=False)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    inception.Conv2d_1a_3x3,
                    inception.Conv2d_2a_3x3,
                    inception.Conv2d_2b_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                ),
                nn.Sequential(
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                ),
            ]
        )
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, 1.0).clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for block in self.blocks:
            x = block(x)
        return x.reshape(x.shape[0], -1)


def compute_activation_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@dataclass
class MetricResults:
    fid: float | None
    ssim: float | None
    lpips: float | None
    psnr: float | None
    num_pairs: int
    num_pred_images: int
    num_gt_images: int
    num_missing_pred_images: int = 0
    ssim_256: float | None = None


def _batch_tensors(paths: Iterable[Path], size: int | tuple[int, int] | None, device: torch.device) -> torch.Tensor:
    tensors = [load_image_tensor(path, size=size) for path in paths]
    return torch.stack(tensors, dim=0).to(device)


def _resize_tensor_image(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _resize_tensor_batch(tensor: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(
        tensor,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def _quantize_rgb01_tensor_batch(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    quantized = [
        TF.to_tensor(TF.to_pil_image(image.detach().cpu().clamp(0.0, 1.0)))
        for image in tensor
    ]
    return torch.stack(quantized, dim=0).to(device)


def _pocold_numpy_images_from_tensors(
    tensor: torch.Tensor,
    eval_size: tuple[int, int] | None,
) -> list[np.ndarray]:
    images = []
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    for image_tensor in tensor:
        image = TF.to_pil_image(image_tensor)
        if eval_size is not None:
            height, width = eval_size
            image = resize_pil_image(image, (width, height))
        images.append(TF.to_tensor(image).permute(1, 2, 0).numpy())
    return images


def _batch_pair_tensors(
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    eval_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_tensors = []
    gt_tensors = []
    batch_size_hw = eval_size
    for pred_path, gt_path in pairs:
        pred = load_image_tensor(pred_path)
        gt = load_image_tensor(gt_path)
        if batch_size_hw is None:
            batch_size_hw = tuple(pred.shape[-2:])
        pred = _resize_tensor_image(pred, batch_size_hw)
        gt = _resize_tensor_image(gt, batch_size_hw)
        pred_tensors.append(pred)
        gt_tensors.append(gt)
    return torch.stack(pred_tensors, dim=0).to(device), torch.stack(gt_tensors, dim=0).to(device)


def compute_fid(
    pred_dir: str | Path,
    gt_dir: str | Path,
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    pred_paths = list_images(pred_dir)
    gt_paths = list_images(gt_dir)
    if not pred_paths or not gt_paths:
        raise ValueError("FID requires non-empty prediction and ground-truth directories.")

    extractor = InceptionFeatureExtractor().to(device)
    pred_feats = []
    gt_feats = []
    fid_load_size: int | tuple[int, int] | None = eval_size if eval_size is not None else 299

    for paths, target in ((pred_paths, pred_feats), (gt_paths, gt_feats)):
        for i in range(0, len(paths), batch_size):
            batch = _batch_tensors(paths[i:i + batch_size], size=fid_load_size, device=device)
            feats = extractor(batch).detach().cpu().numpy()
            target.append(feats)

    pred_feats_np = np.concatenate(pred_feats, axis=0)
    gt_feats_np = np.concatenate(gt_feats, axis=0)
    mu1, sigma1 = compute_activation_stats(pred_feats_np)
    mu2, sigma2 = compute_activation_stats(gt_feats_np)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_fid_from_pairs(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 16,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    extractor = InceptionFeatureExtractor().to(device)
    pred_feats = []
    gt_feats = []
    fid_load_size: int | tuple[int, int] | None = eval_size if eval_size is not None else 299

    pred_paths = [pred for pred, _ in pairs]
    gt_paths = [gt for _, gt in pairs]
    for paths, target in ((pred_paths, pred_feats), (gt_paths, gt_feats)):
        for i in range(0, len(paths), batch_size):
            batch = _batch_tensors(paths[i:i + batch_size], size=fid_load_size, device=device)
            feats = extractor(batch).detach().cpu().numpy()
            target.append(feats)

    pred_feats_np = np.concatenate(pred_feats, axis=0)
    gt_feats_np = np.concatenate(gt_feats, axis=0)
    mu1, sigma1 = compute_activation_stats(pred_feats_np)
    mu2, sigma2 = compute_activation_stats(gt_feats_np)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def _activation_stats_for_paths_pocold(
    paths: list[Path],
    extractor: nn.Module,
    batch_size: int,
    device: torch.device,
    eval_size: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("PoCoLD FID requires non-empty image paths.")
    batch_size = max(1, min(batch_size, len(paths)))

    features = []
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch = _batch_tensors(paths[start:end], size=eval_size, device=device)
        feats = extractor(batch).detach().cpu().numpy()
        features.append(feats)
    return compute_activation_stats(np.concatenate(features, axis=0))


def compute_fid_pocold_style(
    pred_dir: str | Path,
    fid_real_dir: str | Path,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    pred_paths = list_images(pred_dir)
    real_paths = list_images(fid_real_dir)
    extractor = PoCoLDInceptionFeatureExtractor().to(device)
    mu_real, sigma_real = _activation_stats_for_paths_pocold(real_paths, extractor, batch_size, device, eval_size)
    mu_pred, sigma_pred = _activation_stats_for_paths_pocold(pred_paths, extractor, batch_size, device, eval_size)
    return frechet_distance(mu_real, sigma_real, mu_pred, sigma_pred)


def compute_fid_pocold_style_from_real_paths(
    pred_paths: list[Path],
    real_paths: list[Path],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    extractor = PoCoLDInceptionFeatureExtractor().to(device)
    mu_real, sigma_real = _activation_stats_for_paths_pocold(real_paths, extractor, batch_size, device, eval_size)
    mu_pred, sigma_pred = _activation_stats_for_paths_pocold(pred_paths, extractor, batch_size, device, eval_size)
    return frechet_distance(mu_real, sigma_real, mu_pred, sigma_pred)


def build_fid_pocold_style_model(device: str | torch.device = "cpu") -> nn.Module:
    return PoCoLDInceptionFeatureExtractor().to(torch.device(device)).eval()


def compute_fid_real_stats_pocold_style(
    real_paths: list[Path],
    extractor: nn.Module,
    batch_size: int,
    device: str | torch.device,
    eval_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _activation_stats_for_paths_pocold(real_paths, extractor, batch_size, torch.device(device), eval_size)


@torch.no_grad()
def extract_fid_features_pocold_style_from_tensors(
    tensor: torch.Tensor,
    extractor: nn.Module,
    eval_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    tensor = _resize_tensor_batch(_quantize_rgb01_tensor_batch(tensor), eval_size)
    return extractor(tensor).detach()


def compute_ssim(
    pred_dir: str | Path,
    gt_dir: str | Path,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    pairs = match_image_pairs(pred_dir, gt_dir)
    ssim = SSIM().to(device)
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        scores.append(ssim(pred, gt).detach().cpu())
    return float(torch.cat(scores).mean().item())


def compute_ssim_from_pairs(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    ssim = SSIM().to(device)
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        scores.append(ssim(pred, gt).detach().cpu())
    return float(torch.cat(scores).mean().item())


def compute_psnr_from_pairs(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        mse = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(torch.clamp(1.0 / torch.clamp(mse, min=1e-10), min=1e-10))
        scores.append(psnr.detach().cpu())
    return float(torch.cat(scores).mean().item())


def build_lpips_model(net_type: str = "alex"):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires torchmetrics with image support or the lpips package. "
            "Try: pip install torchmetrics lpips"
        ) from exc
    try:
        return LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize LPIPS backbone weights. "
            "This usually means the pretrained weights are not cached locally and the environment cannot download them. "
            "Pre-download the torchvision backbone weights or run once with internet access."
        ) from exc


def match_pocold_deform_pairs(pred_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    gt_dir = Path(gt_dir)
    pairs = []
    for pred_path in list_images(pred_dir):
        image = pred_path.name
        image = image.split("_2_")[-1]
        image = image.split("_vis")[0] + ".png"
        gt_path = gt_dir / image
        if gt_path.is_file():
            pairs.append((pred_path, gt_path))
    if not pairs:
        raise ValueError("No PoCoLD-style pairs found. Expected prediction names like source_2_target.png.")
    return pairs


def compute_ssim_pocold_style_from_pairs(
    pairs: list[tuple[Path, Path]],
    eval_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    from skimage.metrics import structural_similarity

    ssim_scores = []
    ssim_256_scores = []
    for pred_path, gt_path in pairs:
        pred = load_image_tensor(pred_path, size=eval_size).permute(1, 2, 0).numpy()
        gt = load_image_tensor(gt_path, size=eval_size).permute(1, 2, 0).numpy()
        pred_255 = pred * 255.0
        gt_255 = gt * 255.0
        ssim_scores.append(
            structural_similarity(
                gt,
                pred,
                data_range=1,
                win_size=51,
                channel_axis=2,
            )
        )
        data_range = float(pred_255.max() - pred_255.min())
        if data_range <= 0:
            data_range = 255.0
        ssim_256_scores.append(
            structural_similarity(
                gt_255,
                pred_255,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                channel_axis=2,
                data_range=data_range,
            )
        )
    return float(np.mean(ssim_scores)), float(np.mean(ssim_256_scores))


def compute_ssim_pocold_style_from_tensors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eval_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    from skimage.metrics import structural_similarity

    pred_images = _pocold_numpy_images_from_tensors(pred, eval_size)
    gt_images = _pocold_numpy_images_from_tensors(gt, eval_size)
    ssim_scores = []
    ssim_256_scores = []
    for pred_np, gt_np in zip(pred_images, gt_images):
        pred_255 = pred_np * 255.0
        gt_255 = gt_np * 255.0
        ssim_scores.append(
            structural_similarity(
                gt_np,
                pred_np,
                data_range=1,
                win_size=51,
                channel_axis=2,
            )
        )
        data_range = float(pred_255.max() - pred_255.min())
        if data_range <= 0:
            data_range = 255.0
        ssim_256_scores.append(
            structural_similarity(
                gt_255,
                pred_255,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                channel_axis=2,
                data_range=data_range,
            )
        )
    return float(np.mean(ssim_scores)), float(np.mean(ssim_256_scores))


def compute_psnr_pocold_style_from_pairs(
    pairs: list[tuple[Path, Path]],
    eval_size: tuple[int, int] | None = None,
) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    scores = []
    for pred_path, gt_path in pairs:
        pred = load_image_tensor(pred_path, size=eval_size).permute(1, 2, 0).numpy()
        gt = load_image_tensor(gt_path, size=eval_size).permute(1, 2, 0).numpy()
        scores.append(peak_signal_noise_ratio(gt, pred, data_range=1))
    return float(np.mean(scores))


def compute_psnr_pocold_style_from_tensors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eval_size: tuple[int, int] | None = None,
) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    pred_images = _pocold_numpy_images_from_tensors(pred, eval_size)
    gt_images = _pocold_numpy_images_from_tensors(gt, eval_size)
    scores = []
    for pred_np, gt_np in zip(pred_images, gt_images):
        scores.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1))
    return float(np.mean(scores))


def compute_lpips(
    pred_dir: str | Path,
    gt_dir: str | Path,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    net_type: str = "alex",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    pairs = match_image_pairs(pred_dir, gt_dir)
    metric = build_lpips_model(net_type=net_type).to(device)
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        scores.append(metric(pred, gt).detach().view(1).cpu())
    return float(torch.cat(scores).mean().item())


def compute_lpips_from_pairs(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    net_type: str = "alex",
    eval_size: tuple[int, int] | None = None,
) -> float:
    device = torch.device(device)
    metric = build_lpips_model(net_type=net_type).to(device)
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        scores.append(metric(pred, gt).detach().view(1).cpu())
    return float(torch.cat(scores).mean().item())


def compute_lpips_pocold_style_from_pairs(
    pairs: list[tuple[Path, Path]],
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    eval_size: tuple[int, int] | None = None,
) -> float:
    try:
        import lpips
    except ImportError:
        return compute_lpips_from_pairs(
            pairs,
            batch_size=batch_size,
            device=device,
            net_type="alex",
            eval_size=eval_size,
        )

    device = torch.device(device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    if not pairs:
        raise ValueError("PoCoLD LPIPS has no usable pairs after batch splitting.")
    batch_size = max(1, min(batch_size, len(pairs)))

    scores = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        pred, gt = _batch_pair_tensors(chunk, device, eval_size=eval_size)
        scores.append(metric(pred, gt, normalize=True).detach().cpu().numpy())
    return float(np.concatenate(scores, axis=0).mean())


def build_lpips_pocold_style_model(device: str | torch.device = "cpu", net_type: str = "alex") -> nn.Module:
    device = torch.device(device)
    try:
        import lpips
    except ImportError:
        return build_lpips_model(net_type=net_type).to(device).eval()
    return lpips.LPIPS(net=net_type).to(device).eval()


@torch.no_grad()
def compute_lpips_pocold_style_from_tensors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metric: nn.Module,
    eval_size: tuple[int, int] | None = None,
) -> float:
    pred = _resize_tensor_batch(_quantize_rgb01_tensor_batch(pred), eval_size)
    gt = _resize_tensor_batch(_quantize_rgb01_tensor_batch(gt), eval_size)
    try:
        score = metric(pred, gt, normalize=True)
    except TypeError:
        score = metric(pred, gt)
    return float(score.detach().mean().item())


def compute_all_metrics(
    pred_dir: str | Path,
    gt_dir: str | Path,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    lpips_net: str = "alex",
    eval_size: tuple[int, int] | None = None,
) -> MetricResults:
    pred_images = list_images(pred_dir)
    gt_images = list_images(gt_dir)
    pairs = match_image_pairs(pred_dir, gt_dir)
    fid = compute_fid(pred_dir, gt_dir, batch_size=batch_size, device=device, eval_size=eval_size)
    ssim = compute_ssim(pred_dir, gt_dir, batch_size=batch_size, device=device, eval_size=eval_size)
    psnr = compute_psnr_from_pairs(pairs, batch_size=batch_size, device=device, eval_size=eval_size)
    lpips = compute_lpips(pred_dir, gt_dir, batch_size=batch_size, device=device, net_type=lpips_net, eval_size=eval_size)
    return MetricResults(
        fid=fid,
        ssim=ssim,
        lpips=lpips,
        psnr=psnr,
        num_pairs=len(pairs),
        num_pred_images=len(pred_images),
        num_gt_images=len(gt_images),
        num_missing_pred_images=0,
    )


def compute_all_metrics_from_test_pairs(
    pred_dir: str | Path,
    dataset_root: str | Path,
    pairs_file: str = "test_pairs.txt",
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    lpips_net: str = "alex",
    eval_size: tuple[int, int] | None = None,
) -> MetricResults:
    resolved = resolve_pairs_from_test_pairs(pred_dir=pred_dir, dataset_root=dataset_root, pairs_file=pairs_file)
    pairs = resolved.pairs
    pred_images = list_images(pred_dir)
    fid = compute_fid_from_pairs(pairs, batch_size=batch_size, device=device, eval_size=eval_size)
    ssim = compute_ssim_from_pairs(pairs, batch_size=batch_size, device=device, eval_size=eval_size)
    psnr = compute_psnr_from_pairs(pairs, batch_size=batch_size, device=device, eval_size=eval_size)
    lpips = compute_lpips_from_pairs(pairs, batch_size=batch_size, device=device, net_type=lpips_net, eval_size=eval_size)
    return MetricResults(
        fid=fid,
        ssim=ssim,
        lpips=lpips,
        psnr=psnr,
        num_pairs=len(pairs),
        num_pred_images=len(pred_images),
        num_gt_images=len(pairs),
        num_missing_pred_images=resolved.num_missing_predictions,
    )


def compute_all_metrics_pocold_style(
    pred_dir: str | Path,
    gt_dir: str | Path = "",
    fid_real_dir: str | Path = "",
    dataset_root: str | Path = "",
    pairs_file: str = "test_pairs.txt",
    fid_real_pairs_file: str = "train_pairs.txt",
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    lpips_net: str = "alex",
    eval_size: tuple[int, int] | None = None,
) -> MetricResults:
    if dataset_root:
        resolved = resolve_pairs_from_test_pairs(pred_dir=pred_dir, dataset_root=dataset_root, pairs_file=pairs_file)
        pairs = resolved.pairs
        real_paths = list_images(fid_real_dir) if fid_real_dir else collect_real_image_paths_from_pairs(
            dataset_root=dataset_root,
            pairs_file=fid_real_pairs_file,
        )
        missing_predictions = resolved.num_missing_predictions
    else:
        if not gt_dir or not fid_real_dir:
            raise ValueError("PoCoLD-style directory evaluation requires gt_dir and fid_real_dir.")
        pairs = match_pocold_deform_pairs(pred_dir, gt_dir)
        real_paths = list_images(fid_real_dir)
        missing_predictions = len(list_images(pred_dir)) - len(pairs)

    pred_images = list_images(pred_dir)
    matched_pred_paths = [pred for pred, _ in pairs]
    fid = compute_fid_pocold_style_from_real_paths(
        pred_paths=matched_pred_paths,
        real_paths=real_paths,
        batch_size=batch_size,
        device=device,
        eval_size=eval_size,
    )
    ssim, ssim_256 = compute_ssim_pocold_style_from_pairs(pairs, eval_size=eval_size)
    psnr = compute_psnr_pocold_style_from_pairs(pairs, eval_size=eval_size)
    lpips = compute_lpips_pocold_style_from_pairs(
        pairs,
        batch_size=batch_size,
        device=device,
        eval_size=eval_size,
    )
    return MetricResults(
        fid=fid,
        ssim=ssim,
        lpips=lpips,
        psnr=psnr,
        num_pairs=len(pairs),
        num_pred_images=len(pred_images),
        num_gt_images=len(real_paths),
        num_missing_pred_images=missing_predictions,
        ssim_256=ssim_256,
    )


def compute_metrics_mcld_style(
    pred_dir: str | Path,
    gt_dir: str | Path,
    training_path: str | Path,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> dict[int, MetricResults]:
    resolutions = [256, 512]
    results: dict[int, MetricResults] = {}
    for resolution in resolutions:
        pred_subdir = Path(pred_dir) / str(resolution)
        gt_subdir = Path(gt_dir) / str(resolution)
        fid_real_dir = Path(training_path)
        if not pred_subdir.exists():
            raise FileNotFoundError(pred_subdir)
        if not gt_subdir.exists():
            raise FileNotFoundError(gt_subdir)
        pred_images = list_images(pred_subdir)
        gt_images = list_images(gt_subdir)
        pairs = match_image_pairs(pred_subdir, gt_subdir)
        fid = compute_fid_pocold_style_from_real_paths(
            pred_paths=[pred for pred, _ in pairs],
            real_paths=list_images(fid_real_dir),
            batch_size=batch_size,
            device=device,
            eval_size=(resolution, 176 if resolution == 256 else 352),
        )
        ssim, ssim_256 = compute_ssim_pocold_style_from_pairs(pairs, eval_size=(resolution, 176 if resolution == 256 else 352))
        psnr = compute_psnr_pocold_style_from_pairs(pairs, eval_size=(resolution, 176 if resolution == 256 else 352))
        lpips = compute_lpips_pocold_style_from_pairs(
            pairs,
            batch_size=batch_size,
            device=device,
            eval_size=(resolution, 176 if resolution == 256 else 352),
        )
        results[resolution] = MetricResults(
            fid=fid,
            ssim=ssim,
            lpips=lpips,
            psnr=psnr,
            num_pairs=len(pairs),
            num_pred_images=len(pred_images),
            num_gt_images=len(gt_images),
            num_missing_pred_images=0,
            ssim_256=ssim_256,
        )
    return results
