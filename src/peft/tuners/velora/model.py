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

from __future__ import annotations

import warnings

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_VELORA_TARGET_MODULES_MAPPING

from .layer import VeloraLayer, VeloraLinear


class VeloraModel(BaseTuner):
    prefix: str = "velora_"
    tuner_layer_cls = VeloraLayer

    target_module_mapping = TRANSFORMERS_MODELS_TO_VELORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        velora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "r": velora_config.r,
            "num_groups": velora_config.num_groups,
            "init_type": velora_config.init_type,
            "velora_scale": velora_config.velora_scale,
            "fan_in_fan_out": velora_config.fan_in_fan_out,
            "bias": hasattr(target, "bias") and target.bias is not None,
        }

        if isinstance(target, VeloraLinear):
            target.update_layer(
                adapter_name,
                velora_config.r,
                velora_config.num_groups,
                velora_config.init_type,
                velora_config.velora_scale,
            )
        else:
            new_module = self._create_new_module(velora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(velora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = velora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = velora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        return VeloraLinear(target, adapter_name, **kwargs)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for _, param in model.named_parameters():
            param.requires_grad = False

        train_all_bias = False
        for active_adapter in self.active_adapters:
            config = self.peft_config.get(active_adapter)
            if config is None or config.inference_mode:
                continue
            if getattr(config, "bias", "none") == "all":
                train_all_bias = True
                break

        if train_all_bias:
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True

        for module in model.modules():
            if not isinstance(module, VeloraLayer):
                continue

            active_trainable = []
            for active_adapter in module.active_adapters:
                config = self.peft_config.get(active_adapter)
                if config is not None and not config.inference_mode and active_adapter in module.velora_embed:
                    active_trainable.append(active_adapter)

            if not active_trainable:
                continue

            base_layer = module.get_base_layer()
            if hasattr(base_layer, "weight") and base_layer.weight is not None:
                base_layer.weight.requires_grad = True

            train_layer_bias = any(getattr(self.peft_config[adapter], "bias", "none").endswith("_only") for adapter in active_trainable)
            if train_layer_bias and getattr(base_layer, "bias", None) is not None:
                base_layer.bias.requires_grad = True
