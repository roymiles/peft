# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
import pytest
import torch
from torch import nn

from peft import VeloraConfig, get_peft_model
from peft.tuners.velora.layer import _compress_activations, _normalize_projection, _reconstruct_activations
from peft.utils import get_peft_model_state_dict


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(16, 32)
        self.lin1 = nn.Linear(32, 16)

    def forward(self, x):
        return self.lin1(torch.relu(self.lin0(x)))


class SingleLinear(nn.Module):
    def __init__(self, in_features=128, out_features=64, bias=False):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.lin(x)


class TinyCifarMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(3 * 32 * 32, 128)
        self.lin1 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.lin0(x))
        return self.lin1(x)


def _saved_tensor_bytes(loss_factory) -> int:
    saved_bytes = 0

    def pack_hook(tensor):
        nonlocal saved_bytes
        saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        loss = loss_factory()
    loss.backward()
    return saved_bytes


def test_velora_injection_and_forward_matches_base_layer():
    torch.manual_seed(0)
    base_model = MLP()
    velora_model = get_peft_model(
        copy.deepcopy(base_model),
        VeloraConfig(target_modules=["lin0"], r=1, velora_scale=1.0, init_type="random", num_groups=8),
    )

    x = torch.randn(2, 16)
    base_model.train()
    velora_model.train()

    base_out = base_model(x)
    velora_out = velora_model(x)

    assert torch.allclose(velora_out, base_out, atol=1e-6, rtol=1e-5)

    trainable_names = {name for name, param in velora_model.named_parameters() if param.requires_grad}
    assert "base_model.model.lin0.base_layer.weight" in trainable_names
    assert "base_model.model.lin0.base_layer.bias" not in trainable_names
    assert not velora_model.base_model.model.lin0.velora_embed["default"].requires_grad


def test_velora_requires_group_divisibility():
    model = MLP()
    config = VeloraConfig(target_modules=["lin0"], r=1, velora_scale=1.0, num_groups=7)

    with pytest.raises(ValueError, match="divisible by num_groups"):
        _ = get_peft_model(model, config)


def test_velora_batch_average_once_initializes_projection_once():
    torch.manual_seed(0)
    model = get_peft_model(
        MLP(),
        VeloraConfig(
            target_modules=["lin0"],
            r=1,
            velora_scale=1.0,
            init_type="batch_average_once",
            num_groups=8,
        ),
    )
    layer = model.base_model.model.lin0

    x0 = torch.randn(2, 16)
    model.train()
    _ = model(x0)

    expected = _normalize_projection(x0.reshape(-1, 8, 2).reshape(-1, 2).mean(dim=0)).to(layer.velora_embed["default"])
    assert layer.velora_initialized["default"] is True
    assert torch.allclose(layer.velora_embed["default"], expected, atol=1e-6, rtol=1e-5)

    stored_embed = layer.velora_embed["default"].clone()
    x1 = torch.randn(2, 16)
    _ = model(x1)

    assert torch.allclose(layer.velora_embed["default"], stored_embed, atol=1e-6, rtol=1e-5)


def test_velora_reduces_saved_activation_memory_vs_full_finetuning():
    torch.manual_seed(0)
    base_model = SingleLinear()
    velora_model = get_peft_model(
        copy.deepcopy(base_model),
        VeloraConfig(target_modules=["lin"], r=1, velora_scale=1.0, init_type="random", num_groups=32),
    )

    target = torch.randn(8, 4, 64)
    base_model.train()
    velora_model.train()

    def make_loss(model, x):
        output = model(x)
        return (output - target).pow(2).mean()

    x_base = torch.randn(8, 4, 128, requires_grad=True)
    x_velora = x_base.detach().clone().requires_grad_(True)

    base_saved_bytes = _saved_tensor_bytes(lambda: make_loss(base_model, x_base))
    velora_saved_bytes = _saved_tensor_bytes(lambda: make_loss(velora_model, x_velora))

    print(f"{velora_saved_bytes=} bytes.")
    print(f"{base_saved_bytes=} bytes.")
    assert velora_saved_bytes < base_saved_bytes


def test_velora_backward_matches_manual_reconstruction():
    torch.manual_seed(0)
    model = get_peft_model(
        SingleLinear(in_features=16, out_features=10, bias=False),
        VeloraConfig(target_modules=["lin"], r=1, velora_scale=0.5, init_type="random", num_groups=4),
    )
    layer = model.base_model.model.lin

    embed = _normalize_projection(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    layer.velora_embed["default"] = embed.to(layer.velora_embed["default"])
    layer.velora_initialized["default"] = True

    x = torch.randn(2, 3, 16, requires_grad=True)
    grad_output = torch.randn(2, 3, 10)

    model.train()
    output = model(x)
    output.backward(grad_output)

    weight = layer.base_layer.weight.detach()
    compressed = _compress_activations(x.detach(), embed.to(x.dtype), num_groups=4)
    reconstructed = _reconstruct_activations(compressed, embed.to(x.dtype), in_features=16, velora_scale=0.5)

    grad_output_2d = grad_output.reshape(-1, 10)
    expected_grad_weight = grad_output_2d.transpose(0, 1) @ reconstructed
    expected_grad_input = (grad_output_2d @ weight).reshape_as(x)

    assert torch.allclose(layer.base_layer.weight.grad, expected_grad_weight, atol=1e-6, rtol=1e-5)
    assert torch.allclose(x.grad, expected_grad_input, atol=1e-6, rtol=1e-5)


@pytest.mark.regression
def test_velora_cifar10_single_batch_learning():
    tv = pytest.importorskip("torchvision")

    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    try:
        dataset = tv.datasets.CIFAR10(root="/tmp/peft_cifar10", train=True, transform=transform, download=True)
    except Exception as exc:  # pragma: no cover - network/dataset mirror issues
        pytest.skip(f"CIFAR10 download unavailable: {exc}")

    xs, ys = zip(*[dataset[i] for i in range(32)])
    x = torch.stack(xs)
    y = torch.tensor(ys)

    torch.manual_seed(0)
    model = get_peft_model(
        TinyCifarMLP(),
        VeloraConfig(
            target_modules=["lin0", "lin1"],
            r=1,
            velora_scale=1.0,
            init_type="batch_average_once",
            num_groups=32,
        ),
    )

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    model.train()
    for _ in range(6):
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.7


def test_velora_save_pretrained_includes_base_weights():
    """save_pretrained must persist the trained base-layer weights, not just the embed buffers."""
    torch.manual_seed(0)
    model = get_peft_model(
        MLP(),
        VeloraConfig(target_modules=["lin0", "lin1"], r=1, velora_scale=1.0, init_type="random", num_groups=8),
    )

    # Mutate the base-layer weights so we can verify they were saved and restored.
    with torch.no_grad():
        model.base_model.model.lin0.base_layer.weight.fill_(0.42)
        model.base_model.model.lin1.base_layer.weight.fill_(-0.7)

    state_dict = get_peft_model_state_dict(model)

    # Both embed buffers and base weights must be present.
    assert any("velora_embed" in k for k in state_dict), "velora_embed missing from saved state dict"
    assert any("base_layer.weight" in k for k in state_dict), "base_layer.weight missing from saved state dict"

    # Verify the saved values match.
    for key, tensor in state_dict.items():
        if "lin0.base_layer.weight" in key:
            assert torch.all(tensor == 0.42), f"Unexpected value for {key}"
        if "lin1.base_layer.weight" in key:
            assert torch.all(tensor == -0.7), f"Unexpected value for {key}"