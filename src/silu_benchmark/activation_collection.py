"""Memory-conscious activation collection."""

import numpy as np
import torch
import torch.nn as nn

from .constants import DEVICE
from .config import SILU_ACTIVATION_MODULES


class MemoryEfficientActivationCollector:
    def __init__(self, max_samples_per_layer=10000, target_layers=SILU_ACTIVATION_MODULES):
        self.max_samples_per_layer = max_samples_per_layer
        self.target_layers = set(target_layers)
        self.activations = {}
        self.hooks = []

    def collect(self, model, loader, device, layer_types=(nn.SiLU,)):
        self.activations.clear()

        def hook_factory(name):
            def hook(_module, _input, output):
                if name not in self.activations:
                    self.activations[name] = []
                act_data = output.detach().cpu().numpy().flatten()
                if len(act_data) > self.max_samples_per_layer:
                    indices = np.random.choice(
                        len(act_data), self.max_samples_per_layer, replace=False
                    )
                    act_data = act_data[indices]
                self.activations[name].append(act_data)
            return hook

        for name, module in model.named_modules():
            if name in self.target_layers and isinstance(module, layer_types):
                self.hooks.append(module.register_forward_hook(hook_factory(name)))

        model.eval()
        with torch.no_grad():
            for data, _ in loader:
                model(data.to(device))

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        for name in self.activations:
            if self.activations[name]:
                self.activations[name] = np.concatenate(self.activations[name])

        return self.activations


def collect_activations(model, loader, device=DEVICE, max_samples=10000):
    collector = MemoryEfficientActivationCollector(
        max_samples_per_layer=max_samples,
    )
    return collector.collect(model, loader, device, layer_types=(nn.SiLU,))
