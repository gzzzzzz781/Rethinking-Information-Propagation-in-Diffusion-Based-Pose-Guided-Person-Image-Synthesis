from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return x / torch.sqrt(var + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.body = WithBiasLayerNorm(dim) if bias else BiasFreeLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.body(x)
        return x.permute(0, 3, 1, 2).contiguous().view(x.shape[0], -1, h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if shift_size < 0:
            raise ValueError("shift_size must be non-negative.")

        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _partition_windows(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        b, heads, head_dim, h, w = tensor.shape
        tensor = tensor.view(b, heads, head_dim, h // window_size, window_size, w // window_size, window_size)
        tensor = tensor.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        return tensor.view(-1, heads, window_size * window_size, head_dim)

    def _reverse_windows(self, tensor: torch.Tensor, batch: int, height: int, width: int, window_size: int) -> torch.Tensor:
        tensor = tensor.view(
            batch,
            height // window_size,
            width // window_size,
            self.num_heads,
            window_size,
            window_size,
            self.head_dim,
        )
        tensor = tensor.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return tensor.view(batch, self.num_heads * self.head_dim, height, width)

    def _partition_mask(self, mask: torch.Tensor, window_size: int) -> torch.Tensor:
        batch, height, width, channels = mask.shape
        mask = mask.view(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        mask = mask.permute(0, 1, 3, 2, 4, 5).contiguous()
        return mask.view(-1, window_size * window_size)

    def _build_attention_mask(
        self,
        height: int,
        width: int,
        window_size: int,
        shift_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        img_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = self._partition_mask(img_mask, window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        window_size = max(1, min(self.window_size, h, w))
        shift_size = min(self.shift_size, window_size // 2)

        qkv = self.qkv_dwconv(self.qkv(x))
        pad_h = (window_size - (h % window_size)) % window_size
        pad_w = (window_size - (w % window_size)) % window_size
        if pad_h > 0 or pad_w > 0:
            qkv = F.pad(qkv, (0, pad_w, 0, pad_h), mode="replicate")

        _, _, padded_h, padded_w = qkv.shape
        if shift_size > 0:
            qkv = torch.roll(qkv, shifts=(-shift_size, -shift_size), dims=(-2, -1))

        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, self.head_dim, padded_h, padded_w)
        k = k.view(b, self.num_heads, self.head_dim, padded_h, padded_w)
        v = v.view(b, self.num_heads, self.head_dim, padded_h, padded_w)

        q = self._partition_windows(q, window_size)
        k = self._partition_windows(k, window_size)
        v = self._partition_windows(v, window_size)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        if shift_size > 0:
            num_windows = (padded_h // window_size) * (padded_w // window_size)
            attn = attn.view(b, num_windows, self.num_heads, window_size * window_size, window_size * window_size)
            attn_mask = self._build_attention_mask(padded_h, padded_w, window_size, shift_size, attn.device)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, window_size * window_size, window_size * window_size)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = self._reverse_windows(out, b, padded_h, padded_w, window_size)
        if shift_size > 0:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(-2, -1))
        out = out[:, :, :h, :w]
        return self.project_out(out)


class WindowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(dim, num_heads, bias, window_size=window_size, shift_size=shift_size)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, inp_channels: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class Refiner(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, int, int, int] = (1, 2, 2, 4),
        num_refinement_blocks: int = 2,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2,
        bias: bool = False,
        window_size: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.residual_scale = float(residual_scale)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        def make_blocks(block_dim: int, block_heads: int, num_stage_blocks: int) -> nn.Sequential:
            blocks = []
            for block_idx in range(num_stage_blocks):
                shift_size = 0 if block_idx % 2 == 0 or self.window_size <= 1 else self.window_size // 2
                blocks.append(
                    WindowTransformerBlock(
                        block_dim,
                        block_heads,
                        ffn_expansion_factor,
                        bias,
                        window_size=self.window_size,
                        shift_size=shift_size,
                    )
                )
            return nn.Sequential(*blocks)

        self.encoder_level1 = nn.Sequential(
            *make_blocks(dim, heads[0], num_blocks[0])
        )
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *make_blocks(dim * 2, heads[1], num_blocks[1])
        )
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(
            *make_blocks(dim * 4, heads[2], num_blocks[2])
        )
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(
            *make_blocks(dim * 8, heads[3], num_blocks[3])
        )

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *make_blocks(dim * 4, heads[2], num_blocks[2])
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *make_blocks(dim * 2, heads[1], num_blocks[1])
        )

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(
            *make_blocks(dim * 2, heads[0], num_blocks[0])
        )
        self.refinement = nn.Sequential(
            *make_blocks(dim * 2, heads[0], num_refinement_blocks)
        )
      
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        residual = torch.tanh(self.output(out_dec_level1))

        return inp_img + self.residual_scale * residual


def build_refiner(
    *,
    model_dim: int = 48,
    num_blocks: tuple[int, int, int, int] = (1, 2, 2, 4),
    num_refinement_blocks: int = 2,
    heads: tuple[int, int, int, int] = (1, 2, 4, 8),
    ffn_expansion_factor: float = 2,
    model_bias: bool = False,
    window_size: int = 8,
    residual_scale: float = 0.1,
) -> Refiner:
    return Refiner(
        inp_channels=3,
        out_channels=3,
        dim=model_dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=model_bias,
        window_size=window_size,
        residual_scale=residual_scale,
    )


@torch.no_grad()
def run_sanity_check(
    *,
    batch_size: int = 2,
    height: int = 256,
    width: int = 256,
    model_dim: int = 48,
    window_size: int = 8,
    residual_scale: float = 0.1,
    device: str = "cuda",
) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")

    torch_device = torch.device(device)
    model = build_refiner(
        model_dim=model_dim,
        window_size=window_size,
        residual_scale=residual_scale,
    ).to(torch_device)
    model.eval()

    dummy = torch.rand(batch_size, 3, height, width, device=torch_device)
    output = model(dummy)
    num_params = sum(param.numel() for param in model.parameters())

    print("Refiner sanity check passed.")
    print(f"Device:       {torch_device}")
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Model dim:    {model_dim}")
    print(f"Window size:  {window_size}")
    print(f"Residual scale: {residual_scale}")
    print(f"Parameters:   {num_params:,}")
    print(f"Value range:  [{output.min().item():.4f}, {output.max().item():.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()
    run_sanity_check(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        model_dim=args.model_dim,
        window_size=args.window_size,
        residual_scale=args.residual_scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
