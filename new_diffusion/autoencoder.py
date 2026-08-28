from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AutoencoderOutput:
    latent: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class DiffusersAutoencoder(nn.Module):
    """Stable-Diffusion VAE wrapper used as the only VAE path."""

    def __init__(
        self,
        name_or_path: str = "stabilityai/sd-vae-ft-mse",
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "DiffusersAutoencoder requires the `diffusers` package. "
                "Install requirements.txt or run `pip install diffusers`."
            ) from exc

        self.vae = AutoencoderKL.from_pretrained(name_or_path)
        self.scaling_factor = float(scaling_factor)
        self.latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
        block_out_channels = getattr(self.vae.config, "block_out_channels", (1, 1, 1, 1))
        self.downsample_factor = 2 ** (len(block_out_channels) - 1)

    def encode(self, image: torch.Tensor, sample_posterior: bool = True) -> AutoencoderOutput:
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.sample() if sample_posterior else posterior.mode()
        latent = latent * self.scaling_factor
        return AutoencoderOutput(latent=latent, mean=posterior.mean, logvar=posterior.logvar)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.scaling_factor).sample

    def forward(
        self,
        image: torch.Tensor,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(image, sample_posterior=sample_posterior)
        recon = self.decode(encoded.latent)
        return recon, encoded.latent, encoded.mean, encoded.logvar
