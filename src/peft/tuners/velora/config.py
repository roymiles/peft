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
from typing import Optional

from peft.tuners.lora.config import LoraConfig


@dataclass
class VeloraConfig(LoraConfig):
    """Convenience wrapper for enabling VeLoRA as a LoRA variant."""

    r: int = field(
        default=8,
        metadata={"help": "LoRA rank."},
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
    bias: str = field(default="none", metadata={"help": "Bias type. Can be 'none', 'all' or 'lora_only'."})

    def __post_init__(self):
        self.use_velora = True
        self.velora_num_groups = self.num_groups
        self.velora_init_type = self.init_type

        if self.alpha is not None:
            if self.velora_scale != 1.0 and self.velora_scale != float(self.alpha):
                raise ValueError("Specify only one of `alpha` or `velora_scale` for VeLoRA.")
            warnings.warn("`alpha` is deprecated for VeLoRA; use `velora_scale` instead.", FutureWarning)
            self.velora_scale = float(self.alpha)

        if self.velora_dropout != 0.0:
            raise ValueError("Paper-faithful VeLoRA does not use adapter dropout; set `velora_dropout=0.0`.")

        super().__post_init__()
