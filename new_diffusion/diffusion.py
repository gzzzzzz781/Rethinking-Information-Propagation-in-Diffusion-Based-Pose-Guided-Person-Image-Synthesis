from __future__ import annotations

import math

import torch
from torch import nn
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, betas: torch.Tensor, objective: str = "v"):
        super().__init__()
        if objective not in {"eps", "v"}:
            raise ValueError(f"Unsupported diffusion objective: {objective}")
        betas = betas.float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.objective = objective
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def _cond_device(self, cond) -> torch.device:
        if isinstance(cond, dict):
            for value in cond.values():
                if isinstance(value, torch.Tensor):
                    return value.device
            raise ValueError("Condition dict does not contain any tensor values.")
        if isinstance(cond, torch.Tensor):
            return cond.device
        if hasattr(cond, "source_latent") and isinstance(cond.source_latent, torch.Tensor):
            return cond.source_latent.device
        if hasattr(cond, "source_image") and isinstance(cond.source_image, torch.Tensor):
            return cond.source_image.device
        raise ValueError("Unable to infer device from condition input.")

    def _apply_cond_drop(self, cond, source_mask: torch.Tensor | None = None, pose_mask: torch.Tensor | None = None):
        def apply_mask(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            return value if mask is None else value * mask

        if isinstance(cond, dict):
            out = dict(cond)
            if "source_latent" in out:
                out["source_latent"] = apply_mask(out["source_latent"], source_mask)
            if "source_image" in out:
                out["source_image"] = apply_mask(out["source_image"], source_mask)
            if "target_pose" in out:
                out["target_pose"] = apply_mask(out["target_pose"], pose_mask)
            return out
        if hasattr(cond, "source_latent") and hasattr(cond, "target_pose"):
            return type(cond)(
                apply_mask(cond.source_latent, source_mask),
                apply_mask(cond.target_pose, pose_mask),
            )
        if hasattr(cond, "source_image") and hasattr(cond, "target_pose"):
            return type(cond)(
                apply_mask(cond.source_image, source_mask),
                apply_mask(cond.target_pose, pose_mask),
            )
        if source_mask is None:
            return cond
        return cond * source_mask

    def _zero_cond(self, cond, zero_source: bool = True, zero_pose: bool = False):
        device = self._cond_device(cond)
        source_mask = torch.zeros(1, 1, 1, 1, device=device) if zero_source else None
        pose_mask = torch.zeros(1, 1, 1, 1, device=device) if zero_pose else None
        return self._apply_cond_drop(cond, source_mask=source_mask, pose_mask=pose_mask)

    def _resolve_guidance_scales(
        self,
        guidance_scale: float = 1.0,
        source_guidance_scale: float | None = None,
        pose_guidance_scale: float | None = None,
    ) -> tuple[float, float]:
        source_scale = guidance_scale if source_guidance_scale is None else source_guidance_scale
        pose_scale = guidance_scale if pose_guidance_scale is None else pose_guidance_scale
        return float(source_scale), float(pose_scale)

    def _guided_model_output(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond,
        guidance_scale: float = 1.0,
        source_guidance_scale: float | None = None,
        pose_guidance_scale: float | None = None,
    ) -> torch.Tensor:
        source_scale, pose_scale = self._resolve_guidance_scales(
            guidance_scale=guidance_scale,
            source_guidance_scale=source_guidance_scale,
            pose_guidance_scale=pose_guidance_scale,
        )
        if source_scale == 1.0 and pose_scale == 1.0:
            return model(x, t, cond)

        pred_none = model(x, t, self._zero_cond(cond, zero_source=True, zero_pose=True))
        pred_pose = model(x, t, self._zero_cond(cond, zero_source=True, zero_pose=False))
        pred_both = model(x, t, cond)

        return pred_none + pose_scale * (pred_pose - pred_none) + source_scale * (pred_both - pred_pose)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas_cumprod.gather(0, t).float()
        return alpha / (1.0 - alpha).clamp(min=1e-8)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_prediction(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.objective == "eps":
            pred_noise = model_output
            x0 = self.predict_start_from_noise(x_t, t, pred_noise)
            return pred_noise, x0
        pred_v = model_output
        x0 = self.predict_start_from_v(x_t, t, pred_v)
        pred_noise = (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * pred_v
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )
        return pred_noise, x0

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    def training_step(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        cond,
        loss_weight: torch.Tensor | None = None,
        source_drop_prob: float = 0.0,
        pose_drop_prob: float = 0.0,
        snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        b = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                b,
                x_start.shape[1],
                1,
                1,
                device=device,
                dtype=x_start.dtype,
            )
        x_t = self.q_sample(x_start, t, noise)

        if source_drop_prob > 0 or pose_drop_prob > 0:
            source_keep = None
            pose_keep = None
            if source_drop_prob > 0:
                source_keep = (torch.rand(b, device=device) >= source_drop_prob).float().view(b, 1, 1, 1)
            if pose_drop_prob > 0:
                pose_keep = (torch.rand(b, device=device) >= pose_drop_prob).float().view(b, 1, 1, 1)
            cond = self._apply_cond_drop(cond, source_mask=source_keep, pose_mask=pose_keep)

        model_output = model(x_t, t, cond)
        target = noise if self.objective == "eps" else self.predict_v(x_start, t, noise)
        loss_map = (model_output - target) ** 2
        if loss_weight is not None:
            loss_weight = loss_weight.to(device=device, dtype=loss_map.dtype)
            if loss_weight.ndim == loss_map.ndim - 1:
                loss_weight = loss_weight.unsqueeze(1)
            elif loss_weight.ndim != loss_map.ndim:
                raise ValueError(
                    f"loss_weight must have {loss_map.ndim - 1} or {loss_map.ndim} dims, got {loss_weight.ndim}"
                )
            loss_map = loss_map * loss_weight
            weight_sum = loss_weight.sum(dim=tuple(range(1, loss_weight.ndim)), keepdim=False).clamp(min=1e-8)
            loss = loss_map.sum(dim=tuple(range(1, loss_map.ndim))) / weight_sum
        else:
            loss = loss_map.mean(dim=tuple(range(1, loss_map.ndim)))
        if snr_gamma > 0:
            snr = self.snr(t).to(device=loss.device, dtype=loss.dtype)
            gamma = torch.full_like(snr, float(snr_gamma))
            weights = torch.minimum(snr, gamma)
            if self.objective == "eps":
                weights = weights / snr.clamp(min=1e-8)
            else:
                weights = weights / (snr + 1.0)
            loss = loss * weights
        _, pred_x0 = self.model_prediction(model_output, x_t, t)
        return {
            "loss": loss.mean(),
            "pred_x0": pred_x0,
            "t": t,
        }

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        cond,
        source_drop_prob: float = 0.0,
        pose_drop_prob: float = 0.0,
        snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
        loss_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.training_step(
            model,
            x_start,
            cond,
            loss_weight=loss_weight,
            source_drop_prob=source_drop_prob,
            pose_drop_prob=pose_drop_prob,
            snr_gamma=snr_gamma,
            noise_offset=noise_offset,
        )["loss"]

    def _diffusers_prediction_type(self) -> str:
        return "epsilon" if self.objective == "eps" else "v_prediction"

    def _build_diffusers_scheduler(self, sampler: str):
        try:
            from diffusers import DDIMScheduler, DDPMScheduler
        except ImportError as exc:
            raise ImportError(
                "Diffusers scheduler sampling requires the `diffusers` package. "
                "Install requirements.txt or run `pip install diffusers`."
            ) from exc

        kwargs = {
            "num_train_timesteps": self.num_timesteps,
            "trained_betas": self.betas.detach().float().cpu().numpy(),
            "prediction_type": self._diffusers_prediction_type(),
            "clip_sample": False,
            "steps_offset": 0,
        }
        if sampler == "ddim":
            return DDIMScheduler(set_alpha_to_one=True, timestep_spacing="leading", **kwargs)
        if sampler == "ddpm":
            return DDPMScheduler(timestep_spacing="leading", **kwargs)
        raise ValueError(f"Unsupported diffusers sampler={sampler}.")

    @torch.no_grad()
    def diffusers_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        cond,
        sampler: str = "ddim",
        steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        source_guidance_scale: float | None = None,
        pose_guidance_scale: float | None = None,
        device: torch.device | str | None = None,
        show_progress: bool = False,
        progress_desc: str = "sampling",
        known_latent: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self._cond_device(cond)
        if (known_latent is None) != (edit_mask is None):
            raise ValueError("known_latent and edit_mask must be provided together.")
        if known_latent is not None:
            known_latent = known_latent.to(device)
            edit_mask = edit_mask.to(device=device, dtype=known_latent.dtype)
            if tuple(known_latent.shape) != tuple(shape):
                raise ValueError(f"known_latent shape {tuple(known_latent.shape)} does not match {shape}.")
            if edit_mask.ndim != 4 or edit_mask.shape[0] != shape[0] or edit_mask.shape[-2:] != shape[-2:]:
                raise ValueError(f"edit_mask shape {tuple(edit_mask.shape)} is incompatible with {shape}.")
            if edit_mask.shape[1] not in {1, shape[1]}:
                raise ValueError("edit_mask must have one channel or match the latent channels.")
            edit_mask = edit_mask.clamp(0.0, 1.0)

        scheduler = self._build_diffusers_scheduler(sampler)
        scheduler.set_timesteps(steps, device=device)
        x = torch.randn(shape, device=device)
        x = x * getattr(scheduler, "init_noise_sigma", 1.0)
        known_noise = torch.randn_like(known_latent) if known_latent is not None else None

        if known_latent is not None:
            initial_t = scheduler.timesteps[0].reshape(1).repeat(shape[0])
            known_noisy = scheduler.add_noise(known_latent, known_noise, initial_t)
            x = x * edit_mask + known_noisy * (1.0 - edit_mask)

        scheduler_timesteps = scheduler.timesteps
        timesteps = tqdm(scheduler_timesteps, desc=progress_desc, leave=False) if show_progress else scheduler_timesteps
        for step_index, timestep in enumerate(timesteps):
            latent_model_input = scheduler.scale_model_input(x, timestep)
            t = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
            model_output = self._guided_model_output(
                model,
                latent_model_input,
                t,
                cond,
                guidance_scale=guidance_scale,
                source_guidance_scale=source_guidance_scale,
                pose_guidance_scale=pose_guidance_scale,
            )
            step_kwargs = {"return_dict": False}
            if sampler == "ddim":
                step_kwargs["eta"] = eta
            x = scheduler.step(model_output, timestep, x, **step_kwargs)[0]

            if known_latent is not None:
                # PoCoLD-style editing: only the mask is generated while the
                # known latent follows the matching forward-noise trajectory.
                if step_index + 1 < len(scheduler_timesteps):
                    next_t = scheduler_timesteps[step_index + 1].reshape(1).repeat(shape[0])
                    known_noisy = scheduler.add_noise(known_latent, known_noise, next_t)
                else:
                    known_noisy = known_latent
                x = x * edit_mask + known_noisy * (1.0 - edit_mask)
        return x

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        cond,
        guidance_scale: float = 1.0,
        source_guidance_scale: float | None = None,
        pose_guidance_scale: float | None = None,
        device: torch.device | str | None = None,
        show_progress: bool = False,
        progress_desc: str = "sampling",
        known_latent: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.diffusers_sample(
            model,
            shape,
            cond,
            sampler="ddpm",
            steps=self.num_timesteps,
            eta=0.0,
            guidance_scale=guidance_scale,
            source_guidance_scale=source_guidance_scale,
            pose_guidance_scale=pose_guidance_scale,
            device=device,
            show_progress=show_progress,
            progress_desc=progress_desc,
            known_latent=known_latent,
            edit_mask=edit_mask,
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        cond,
        steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        source_guidance_scale: float | None = None,
        pose_guidance_scale: float | None = None,
        device: torch.device | str | None = None,
        show_progress: bool = False,
        progress_desc: str = "sampling",
        known_latent: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.diffusers_sample(
            model,
            shape,
            cond,
            sampler="ddim",
            steps=steps,
            eta=eta,
            guidance_scale=guidance_scale,
            source_guidance_scale=source_guidance_scale,
            pose_guidance_scale=pose_guidance_scale,
            device=device,
            show_progress=show_progress,
            progress_desc=progress_desc,
            known_latent=known_latent,
            edit_mask=edit_mask,
        )
