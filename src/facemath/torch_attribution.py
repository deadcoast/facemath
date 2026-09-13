from __future__ import annotations
from typing import Callable
import torch


def input_gradient(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, target_index: int) -> torch.Tensor:
    z = x.detach().clone().requires_grad_(True)
    out = model(z)
    scalar = out[..., target_index].sum()
    scalar.backward()
    return z.grad.detach()


def integrated_gradients(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, target_index: int, baseline: torch.Tensor | None = None, steps: int = 64) -> torch.Tensor:
    baseline = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype):
        z = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        out = model(z)[..., target_index].sum()
        grad = torch.autograd.grad(out, z)[0]
        total += grad
    avg = total / steps
    return (x - baseline) * avg
