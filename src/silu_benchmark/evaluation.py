"""Evaluation helpers."""

import torch

from .constants import DEVICE


def test(model, loader, device=DEVICE, max_batches=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total if total else 0.0
