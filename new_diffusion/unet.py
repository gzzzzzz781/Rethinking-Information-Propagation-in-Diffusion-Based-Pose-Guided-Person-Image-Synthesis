from __future__ import annotations
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D
try:
    from diffusers.models.unets.unet_2d_blocks import get_down_block
except ModuleNotFoundError:
    from diffusers.models.unet_2d_blocks import get_down_block
try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
except ModuleNotFoundError:
    from diffusers.models.unet_2d_condition import UNet2DConditionOutput


class ChannelMapper(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 1,
        concat: bool = True,
        bias: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.concat = concat
        self.cond_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        if zero_init:
            self.cond_conv.weight.data.fill_(0.0)

    def forward(self, x, cond):
        cond = self.cond_conv(cond)
        if self.concat:
            return torch.cat([x, cond], dim=1)
        return x + cond


class DoubleAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_filters: int,
        heads: int = 8,
        dim_head: int = 64,
        bias=False,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.attn_scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(cross_attention_dim, eps=ln_eps)
        self.self_norm = nn.LayerNorm(query_dim, eps=ln_eps)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.self_to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.self_to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.self_to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.self_to_out = nn.Linear(inner_dim, query_dim)
        #self.value_gate = nn.Linear(inner_dim, inner_dim, bias=True)

        self.num_filters = num_filters
        self.extraction_filters = nn.Parameter(torch.randn(num_filters, inner_dim))
        self.distribution_filters = nn.Parameter(torch.randn(num_filters, inner_dim))
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.self_residual_scale = nn.Parameter(torch.ones(1))

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.contiguous().reshape(
            batch_size // head_size, head_size, seq_len, dim
        )
        tensor = (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size // head_size, seq_len, dim * head_size)
        )
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.contiguous().reshape(
            batch_size, seq_len, head_size, dim // head_size
        )
        tensor = (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * head_size, seq_len, dim // head_size)
        )
        return tensor

    def get_attention_scores(self, query, key, dim=-1):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.bmm(
            query,
            key.transpose(-1, -2),
        )
        attention_scores = attention_scores * self.attn_scale

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=dim)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def _spatial_to_sequence(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, _, height, width = hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, hidden_states.shape[1])
        )
        return hidden_states, height, width

    def _sequence_to_spatial(
        self, hidden_states: torch.Tensor, batch_size: int, height: int, width: int, channels: int
    ) -> torch.Tensor:
        return (
            hidden_states.reshape(batch_size, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _self_attention_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_size, _, height, width = hidden_states.shape
        hidden_states, _, _ = self._spatial_to_sequence(hidden_states)
        hidden_states = self.self_norm(hidden_states)

        query = self.self_to_q(hidden_states)
        key = self.self_to_k(hidden_states)
        value = self.self_to_v(hidden_states)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        extraction_filters = self.extraction_filters.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        distribution_filters = self.distribution_filters.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        extraction_filters = self.head_to_batch_dim(extraction_filters)
        distribution_filters = self.head_to_batch_dim(distribution_filters)

        spatial_attention_probs = self.get_attention_scores(
            extraction_filters, key, dim=-1
        )
        channel_attention_probs = self.get_attention_scores(
            distribution_filters, query, dim=1
        )
        attention_probs = torch.bmm(
            channel_attention_probs.transpose(-1, -2), spatial_attention_probs
        )
        attention_probs = self.get_attention_scores(query, key, dim=-1)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        hidden_states = self.self_to_out(hidden_states)
        hidden_states = self._sequence_to_spatial(
            hidden_states, batch_size, height, width, self.query_dim
        )
        return hidden_states + residual * self.self_residual_scale

    def forward(self, hidden_states, encoder_hidden_states):
        residual = hidden_states
        batch_size, _, height, width = hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, self.query_dim)
        )
        encoder_hidden_states = (
            encoder_hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, -1, self.cross_attention_dim)
        )
        encoder_hidden_states = self.norm(encoder_hidden_states)

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        #value_gate = torch.sigmoid(self.value_gate(query))
        #value = value * value_gate

        query = self.head_to_batch_dim(query)  # hw,c
        key = self.head_to_batch_dim(key)  # hw,c
        value = self.head_to_batch_dim(value)  # hw,c
 
        attention_probs = self.get_attention_scores(query, key, dim=-1)  # hw,hw

        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = self.batch_to_head_dim(hidden_states)
        hidden_states = self.to_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, self.query_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        hidden_states = hidden_states + residual * self.residual_scale
        hidden_states = self._self_attention_forward(hidden_states)

        return hidden_states


class DoubleAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        resnets2 = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                DoubleAttention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=out_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for DoubleAttnDownBlock2D.")

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DoubleAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        res_skip_channels_list = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            res_skip_channels_list.append(res_skip_channels)

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                DoubleAttention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=out_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.res_skip_channels = tuple(res_skip_channels_list)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for DoubleAttnUpBlock2D.")

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UNetMidBlock2DDoubleAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_filters: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_heads=8,
        cross_attention_dim=512,
        output_scale_factor=1.0,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                DoubleAttention(
                    query_dim=in_channels,
                    cross_attention_dim=cross_attention_dim,
                    num_filters=num_filters,
                    heads=attn_num_heads,
                    dim_head=in_channels // attn_num_heads,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                    ln_eps=resnet_eps,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for UNetMidBlock2DDoubleAttn.")

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNet2DDoubleAttentionConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        num_filters: Union[int, Tuple[int]] = 64,
        layers_per_block: int = 1,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_channels: Tuple[int] = (128, 256, 512, 512),
        attn_heads_nums: Union[int, Tuple[int]] = 8,
        upcast_attention: bool = True,
        upcast_softmax: bool = True,
        resnet_time_scale_shift: str = "default",
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(attn_heads_nums, int):
            attn_heads_nums = (attn_heads_nums,) * len(block_out_channels)
        if isinstance(num_filters, int):
            num_filters = (num_filters,) * len(block_out_channels)

        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DoubleAttnDownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_filters=num_filters[i],
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_heads=attn_heads_nums[i],
                cross_attention_dim=cross_attention_channels[i],
                add_downsample=not is_final_block,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2DDoubleAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            num_filters=num_filters[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_groups=norm_num_groups,
            attn_num_heads=attn_heads_nums[-1],
            cross_attention_dim=cross_attention_channels[-1],
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )

        self.num_upsamplers = 0

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attn_heads_nums = list(reversed(attn_heads_nums))
        reversed_cross_attention_channels = list(reversed(cross_attention_channels))
        reversed_num_filters = list(reversed(num_filters))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = DoubleAttnUpBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                num_filters=reversed_num_filters[i],
                add_upsample=add_upsample,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_heads=reversed_attn_heads_nums[i],
                cross_attention_dim=reversed_cross_attention_channels[i],
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2 ** self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for UNet2DDoubleAttentionConditionModel.")
        for i, downsample_block in enumerate(self.down_blocks):
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states[i],
            )
            down_block_res_samples += res_samples

        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states[-1],
        )

        reversed_encoder_hidden_states = encoder_hidden_states[::-1]

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            remaining_res_samples = list(down_block_res_samples)
            matched_res_samples = []
            target_size = sample.shape[-2:]
            for expected_channels in upsample_block.res_skip_channels:
                while remaining_res_samples and (
                    remaining_res_samples[-1].shape[-2:] != target_size
                    or remaining_res_samples[-1].shape[1] != expected_channels
                ):
                    remaining_res_samples.pop()
                if not remaining_res_samples:
                    raise RuntimeError(
                        f"No skip sample matches up block {i} target size {target_size} "
                        f"and channels {expected_channels}."
                    )
                matched_res_samples.append(remaining_res_samples.pop())
            res_samples = tuple(reversed(matched_res_samples))
            down_block_res_samples = tuple(remaining_res_samples)

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=reversed_encoder_hidden_states[i],
                upsample_size=upsample_size,
            )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


class MultiScaleReferenceImageEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        num_layers: int = 1,
        norm_eps: float = 1e-5,
        use_time_emb: Optional[bool] = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.down_blocks = nn.ModuleList([])

        if use_time_emb:
            timestep_input_dim = block_out_channels[0]
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        else:
            time_embed_dim = None
            self.time_proj = None
            self.time_embedding = None

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=num_layers,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=i > 0,
                resnet_eps=norm_eps,
                resnet_act_fn="silu",
                attention_head_dim=1,
                resnet_groups=32,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

    def forward(self, sample, timesteps=None):
        sample = self.conv_in(sample)
        samples = []
        if timesteps is not None and self.config.use_time_emb:
            timesteps = timesteps.expand(sample.shape[0])
            temb = self.time_proj(timesteps)
            temb = temb.to(dtype=self.dtype)
            temb = self.time_embedding(temb)
        else:
            temb = None
        for i, downsample_block in enumerate(self.down_blocks):
            sample, _ = downsample_block(hidden_states=sample, temb=temb)
            samples.append(sample)

        return samples


class PoseTransferUNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 4,
        pose_channels: int = 3,
        base_channels: int = 256,
        pose_latent_channels: int = 4,
        attn_heads: int = 4,
        num_filters: int = 64,
        layers_per_block: int = 3,
        block_out_channels: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        if block_out_channels is None:
            block_out_channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)

        self.reference_encoder = MultiScaleReferenceImageEncoder(
            in_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_layers=2,
        )
        self.pose_mapper = ChannelMapper(
            in_channels=pose_channels,
            out_channels=pose_latent_channels,
            concat=True,
        )
        self.unet = UNet2DDoubleAttentionConditionModel(
            in_channels=latent_channels + pose_latent_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            cross_attention_channels=block_out_channels,
            attn_heads_nums=attn_heads,
            num_filters=num_filters,
            layers_per_block=layers_per_block,
        )

    @staticmethod
    def _split_cond(cond) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(cond, dict):
            return cond["source_latent"], cond["target_pose"]
        return cond.source_latent, cond.target_pose

    def encode_reference(self, source_latent: torch.Tensor, latent_size: tuple[int, int]) -> list[torch.Tensor]:
        if source_latent.shape[-2:] != latent_size:
            source_latent = F.interpolate(source_latent, size=latent_size, mode="bilinear", align_corners=False)
        return self.reference_encoder(source_latent)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond=None,
        source_latent: Optional[torch.Tensor] = None,
        target_pose: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[list[torch.Tensor]] = None,
        return_dict: bool = False,
    ):
        if cond is not None:
            source_latent, target_pose = self._split_cond(cond)
        if target_pose is None:
            raise ValueError("target_pose is required.")
        if encoder_hidden_states is None:
            if source_latent is None:
                raise ValueError("source_latent is required when encoder_hidden_states is not provided.")
            encoder_hidden_states = self.encode_reference(source_latent, sample.shape[-2:])

        pose = F.interpolate(target_pose, size=sample.shape[-2:], mode="bilinear", align_corners=False)
        sample = self.pose_mapper(sample, pose)
        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                return_dict=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        if return_dict:
            return output
        return output.sample


UNet = UNet2DDoubleAttentionConditionModel
UNetDiffusers = UNet2DDoubleAttentionConditionModel
PoseTransferUNetDiffusers = PoseTransferUNet


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_model_macs(model: nn.Module, *args, **kwargs) -> int:
    macs = 0
    handles = []
    original_bmm = torch.bmm
    original_matmul = torch.matmul
    original_einsum = torch.einsum
    original_tensor_softmax = torch.Tensor.softmax
    original_sdpa = getattr(F, "scaled_dot_product_attention", None)

    def _prod(shape) -> int:
        result = 1
        for value in shape:
            result *= int(value)
        return result

    def _count_matmul(input: torch.Tensor, mat2: torch.Tensor) -> int:
        if input.ndim == 1 and mat2.ndim == 1:
            return int(input.shape[0])
        if input.ndim == 1 and mat2.ndim >= 2:
            batch_shape = torch.broadcast_shapes(tuple(input.shape[:-1]), tuple(mat2.shape[:-2]))
            return _prod(batch_shape) * int(input.shape[-1]) * int(mat2.shape[-1])
        if input.ndim >= 2 and mat2.ndim == 1:
            batch_shape = torch.broadcast_shapes(tuple(input.shape[:-2]), tuple(mat2.shape[:-1]))
            return _prod(batch_shape) * int(input.shape[-2]) * int(input.shape[-1])
        if input.ndim >= 2 and mat2.ndim >= 2:
            batch_shape = torch.broadcast_shapes(tuple(input.shape[:-2]), tuple(mat2.shape[:-2]))
            return _prod(batch_shape) * int(input.shape[-2]) * int(input.shape[-1]) * int(mat2.shape[-1])
        return 0

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, tuple) else output
        batch, out_channels, out_h, out_w = out.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups
        macs += batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal macs
        out = output[0] if isinstance(output, tuple) else output
        macs += out.numel() * module.in_features

    def counted_bmm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        nonlocal macs
        if input.ndim == 3 and mat2.ndim == 3:
            macs += input.shape[0] * input.shape[1] * input.shape[2] * mat2.shape[2]
        return original_bmm(input, mat2)

    def counted_matmul(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        nonlocal macs
        macs += _count_matmul(input, mat2)
        return original_matmul(input, mat2)

    def counted_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
        nonlocal macs
        out = original_einsum(equation, *operands)
        if len(operands) == 2 and "->" in equation:
            lhs, rhs = equation.replace(" ", "").split("->")
            inputs = lhs.split(",")
            contracted = set(inputs[0]) & set(inputs[1]) - set(rhs)
            dim_by_label = {}
            for labels, tensor in zip(inputs, operands):
                for label, dim in zip(labels, tensor.shape):
                    dim_by_label[label] = max(dim_by_label.get(label, 1), int(dim))
            macs += out.numel() * _prod(dim_by_label[label] for label in contracted)
        return out

    def counted_softmax(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal macs
        macs += input.numel() * 3
        return original_tensor_softmax(input, *args, **kwargs)

    def counted_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *sdpa_args, **sdpa_kwargs) -> torch.Tensor:
        nonlocal macs
        batch_heads = _prod(torch.broadcast_shapes(tuple(query.shape[:-2]), tuple(key.shape[:-2]), tuple(value.shape[:-2])))
        q_len, k_len = int(query.shape[-2]), int(key.shape[-2])
        q_dim, v_dim = int(query.shape[-1]), int(value.shape[-1])
        macs += batch_heads * q_len * k_len * (q_dim + v_dim)
        macs += batch_heads * q_len * k_len * 3
        return original_sdpa(query, key, value, *sdpa_args, **sdpa_kwargs)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    torch.bmm = counted_bmm
    torch.matmul = counted_matmul
    torch.einsum = counted_einsum
    torch.Tensor.softmax = counted_softmax
    if original_sdpa is not None:
        F.scaled_dot_product_attention = counted_sdpa
    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        torch.bmm = original_bmm
        torch.matmul = original_matmul
        torch.einsum = original_einsum
        torch.Tensor.softmax = original_tensor_softmax
        if original_sdpa is not None:
            F.scaled_dot_product_attention = original_sdpa
        if was_training:
            model.train()
        for handle in handles:
            handle.remove()
    return macs


count_conv_linear_macs = count_model_macs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseTransferUNet(base_channels=256).to(device).eval()
    latent = torch.randn(1, 4, 64, 64, device=device)
    timestep = torch.randint(0, 1000, (1,), device=device)
    cond = {
        "source_latent": torch.randn(1, 4, 64, 64, device=device),
        "target_pose": torch.randn(1, 3, 512, 512, device=device),
    }
    with torch.no_grad():
        output = model(latent, timestep, cond)
    total, trainable = count_parameters(model)
    macs = count_model_macs(model, latent, timestep, cond)
    print(f"output: {tuple(output.shape)}")
    print(f"parameters: {total / 1e6:.3f}M")
    print(f"trainable parameters: {trainable / 1e6:.3f}M")
    print(f"estimated conv/linear/attention compute: {macs / 1e9:.3f}G")


if __name__ == "__main__":
    main()
