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
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VeloraConfig(PeftConfig):
    """Configuration for the paper-faithful VeLoRA activation-compression wrapper."""

    r: int = field(
        default=1,
        metadata={"help": "Projection rank. Official VeLoRA uses rank-1 sub-token projections."},
    )
    num_groups: int = field(
        default=32,
        metadata={"help": "Number of feature groups used to split the input activation depth."},
    )
    velora_scale: float = field(
        default=1.0,
        metadata={"help": "Scale applied to the reconstructed activations in the backward pass."},
    )
    init_type: str = field(
        default="batch_average_once",
        metadata={"help": "Projection initialization strategy. Supported: 'batch_average_once', 'random'."},
    )
    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Deprecated alias for `velora_scale`."},
    )
    velora_dropout: float = field(
        default=0.0,
        metadata={"help": "Deprecated. Paper-faithful VeLoRA does not use adapter dropout."},
    )
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set True if target weights are stored as (fan_in, fan_out)."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'velora_only'."})
    modules_to_save: Optional[list[str]] = field(default=None)
    layers_to_transform: Optional[Union[list[int], int]] = field(default=None)
    layers_pattern: Optional[Union[list[str], str]] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VELORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        if self.alpha is not None:
            if self.velora_scale != 1.0 and self.velora_scale != float(self.alpha):
                raise ValueError("Specify only one of `alpha` or `velora_scale` for VeLoRA.")
            warnings.warn("`alpha` is deprecated for VeLoRA; use `velora_scale` instead.", FutureWarning)
            self.velora_scale = float(self.alpha)

        if self.r != 1:
            raise ValueError("Official VeLoRA uses rank-1 sub-token projections; only `r=1` is supported.")
        if self.num_groups <= 0:
            raise ValueError(f"`num_groups` should be positive, got {self.num_groups}.")
        if self.velora_scale <= 0:
            raise ValueError(f"`velora_scale` should be positive, got {self.velora_scale}.")
        if self.init_type not in {"batch_average_once", "random"}:
            raise ValueError(
                f"Unsupported VeLoRA init_type {self.init_type!r}. Supported values are 'batch_average_once' and 'random'."
            )
        if self.velora_dropout != 0.0:
            raise ValueError("Paper-faithful VeLoRA does not use adapter dropout; set `velora_dropout=0.0`.")
