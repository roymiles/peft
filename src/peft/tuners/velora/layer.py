# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


def _reshape_to_grouped_subtokens(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    group_dim = x.shape[-1] // num_groups
    return x.reshape(-1, num_groups, group_dim)


def _compress_activations(x: torch.Tensor, embed: torch.Tensor, num_groups: int) -> torch.Tensor:
    grouped = _reshape_to_grouped_subtokens(x, num_groups)
    return torch.einsum("tgd,d->tg", grouped, embed)


def _reconstruct_activations(
    compressed: torch.Tensor,
    embed: torch.Tensor,
    in_features: int,
    velora_scale: float,
) -> torch.Tensor:
    grouped = compressed.unsqueeze(-1) * embed.view(1, 1, -1)
    return grouped.reshape(-1, in_features) * velora_scale


def _normalize_projection(embed: torch.Tensor) -> torch.Tensor:
    embed = embed.float()
    norm = torch.linalg.vector_norm(embed)
    if norm == 0:
        embed = torch.ones_like(embed)
        norm = torch.linalg.vector_norm(embed)
    return embed / norm


def _compute_dtype_for_backward(reference: torch.Tensor) -> torch.dtype:
    dtype = reference.dtype
    if reference.device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class VeLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        embed: torch.Tensor,
        num_groups: int,
        velora_scale: float,
    ) -> torch.Tensor:
        output = F.linear(input, weight, bias)

        ctx.input_shape = tuple(input.shape)
        ctx.input_dtype = input.dtype
        ctx.in_features = input.shape[-1]
        ctx.num_groups = num_groups
        ctx.velora_scale = velora_scale
        ctx.bias_dtype = None if bias is None else bias.dtype

        compressed = _compress_activations(input, embed.to(dtype=input.dtype), num_groups)
        ctx.save_for_backward(compressed, weight, embed)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compressed, weight, embed = ctx.saved_tensors
        compute_dtype = _compute_dtype_for_backward(grad_output)

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(compute_dtype)
        weight_compute = weight.to(compute_dtype)

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output_2d @ weight_compute
            grad_input = grad_input.reshape(ctx.input_shape).to(dtype=ctx.input_dtype)

        if ctx.needs_input_grad[1]:
            reconstructed = _reconstruct_activations(
                compressed.to(compute_dtype),
                embed.to(compute_dtype),
                ctx.in_features,
                ctx.velora_scale,
            )
            grad_weight = grad_output_2d.transpose(0, 1) @ reconstructed
            grad_weight = grad_weight.to(dtype=weight.dtype)

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)
            grad_bias = grad_bias.to(dtype=ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None, None


class VeloraLayer(BaseTunerLayer):
    adapter_layer_names = ("velora_embed",)
    other_param_names = ("r", "num_groups", "init_type", "velora_scale", "velora_initialized")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.num_groups = {}
        self.init_type = {}
        self.velora_scale = {}
        self.velora_initialized = {}
        self.velora_embed = BufferDict({}, persistent=True)

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise NotImplementedError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _get_available_adapters(self) -> set[str]:
        return set(self.velora_embed.keys())

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        self._active_adapter = adapter_names

    def update_layer(self, adapter_name, r, num_groups, init_type, velora_scale):
        if self.in_features % num_groups != 0:
            raise ValueError(
                f"in_features ({self.in_features}) should be divisible by num_groups ({num_groups}) for VeLoRA"
            )

        self.r[adapter_name] = r
        self.num_groups[adapter_name] = num_groups
        self.init_type[adapter_name] = init_type
        self.velora_scale[adapter_name] = velora_scale

        group_dim = self.in_features // num_groups
        if init_type == "random":
            embed = torch.randn(group_dim)
            embed = _normalize_projection(embed)
            initialized = True
        else:
            embed = torch.zeros(group_dim)
            initialized = False

        self.velora_embed[adapter_name] = embed
        self.velora_initialized[adapter_name] = initialized

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _maybe_initialize_embed(self, x: torch.Tensor, adapter_name: str) -> None:
        if self.velora_initialized[adapter_name]:
            return
        if self.init_type[adapter_name] != "batch_average_once":
            return

        subtokens = _reshape_to_grouped_subtokens(x.detach(), self.num_groups[adapter_name]).reshape(
            -1, self.in_features // self.num_groups[adapter_name]
        )
        embed = _normalize_projection(subtokens.mean(dim=0))
        target = self.velora_embed[adapter_name]
        self.velora_embed[adapter_name] = embed.to(device=target.device, dtype=target.dtype)
        self.velora_initialized[adapter_name] = True


class VeloraLinear(nn.Linear, VeloraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 1,
        num_groups: int = 32,
        init_type: str = "batch_average_once",
        velora_scale: float = 1.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        VeloraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, num_groups, init_type, velora_scale)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        warnings.warn("VeLoRA does not merge weights into the base layer; `merge()` is a no-op.", UserWarning)

    def unmerge(self) -> None:
        warnings.warn("VeLoRA layers are never merged; `unmerge()` is a no-op.", UserWarning)

    def _get_active_adapter_name(self) -> Optional[str]:
        if not self.active_adapters:
            return None
        if len(self.active_adapters) > 1:
            raise ValueError("VeLoRA does not support multiple simultaneously active adapters on the same layer.")
        return self.active_adapters[0]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if args or kwargs:
            return self.base_layer(x, *args, **kwargs)

        active_adapter = self._get_active_adapter_name()
        if self.disable_adapters or active_adapter is None or active_adapter not in self.velora_embed:
            return self.base_layer(x)

        if not self.training or not torch.is_grad_enabled():
            return self.base_layer(x)

        self._maybe_initialize_embed(x, active_adapter)
        weight = transpose(self.get_base_layer().weight, self.fan_in_fan_out)
        bias = self.get_base_layer().bias
        return VeLoRAFunction.apply(
            x,
            weight,
            bias,
            self.velora_embed[active_adapter],
            self.num_groups[active_adapter],
            self.velora_scale[active_adapter],
        )

    def __repr__(self) -> str:
        return "velora." + super().__repr__()
