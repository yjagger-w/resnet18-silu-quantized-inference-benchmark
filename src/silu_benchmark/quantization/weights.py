"""Per-channel weight quantization and bias correction."""

import torch
import torch.nn as nn


def quantize_weights_per_channel(model, bits=8, use_percentile=True):
    qmax = 2 ** (bits - 1) - 1
    quantile_value = 0.95 if bits <= 4 and use_percentile else 1.0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            weight_reshaped = weight.view(weight.size(0), -1)
            if quantile_value < 1.0:
                max_val = torch.quantile(torch.abs(weight_reshaped), quantile_value, dim=1)
            else:
                max_val = torch.max(torch.abs(weight_reshaped), dim=1)[0]
            max_val = torch.clamp(max_val, min=1e-6).view(-1, 1, 1, 1)
            scale = qmax / max_val
            q_weight = torch.round(weight * scale)
            q_weight = torch.clamp(q_weight, -qmax, qmax)
            module.weight.data = q_weight / scale

        elif isinstance(module, nn.Linear):
            weight = module.weight.data
            if quantile_value < 1.0:
                max_val = torch.quantile(torch.abs(weight), quantile_value, dim=1)
            else:
                max_val = torch.max(torch.abs(weight), dim=1)[0]
            max_val = torch.clamp(max_val, min=1e-6).view(-1, 1)
            scale = qmax / max_val
            q_weight = torch.round(weight * scale)
            q_weight = torch.clamp(q_weight, -qmax, qmax)
            module.weight.data = q_weight / scale

    return model


def _named_leaf_modules(model):
    return {name: module for name, module in model.named_modules() if name}


def apply_bias_correction(fp_model, q_model, loader, device, max_batches=1):
    fp_modules = _named_leaf_modules(fp_model)
    q_modules = _named_leaf_modules(q_model)
    corrections = {}
    handles = []

    def save_output(store, name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                dims = tuple(range(output.dim()))
                if output.dim() > 1:
                    dims = (0,) + tuple(range(2, output.dim()))
                store.setdefault(name, []).append(output.detach().mean(dim=dims).cpu())
        return hook

    fp_out = {}
    q_out = {}
    for name, module in fp_modules.items():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in q_modules:
            handles.append(module.register_forward_hook(save_output(fp_out, name)))
            handles.append(q_modules[name].register_forward_hook(save_output(q_out, name)))

    fp_model.eval()
    q_model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            fp_model(x)
            q_model(x)

    for handle in handles:
        handle.remove()

    for name, q_module in q_modules.items():
        if name not in fp_out or name not in q_out or q_module.bias is None:
            continue
        fp_mean = torch.stack(fp_out[name]).mean(dim=0).to(q_module.bias.device)
        q_mean = torch.stack(q_out[name]).mean(dim=0).to(q_module.bias.device)
        correction = fp_mean - q_mean
        q_module.bias.data.add_(correction.view_as(q_module.bias.data))
        corrections[name] = correction.detach().cpu()

    return corrections


quantize_weights_optimized = quantize_weights_per_channel
