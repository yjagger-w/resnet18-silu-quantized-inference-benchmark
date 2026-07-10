"""Activation calibration and quantizer insertion."""

import copy

import numpy as np
import torch.nn as nn

from .activation_collection import collect_activations
from .config import QuantizationConfig, SILU_ACTIVATION_MODULES
from .constants import DEVICE
from .quantization.activation import ActivationQuantizerNCNN, ActivationQuantizerSiLUAware
from .quantization.thresholds import compute_ncnn_threshold, compute_silu_aware_thresholds
from .quantization.weights import apply_bias_correction


def calibrate_ncnn_style(model, calib_loader, device=DEVICE, config=None):
    if config is None:
        config = QuantizationConfig(strategy="ncnn")

    acts_history = []
    for _ in range(config.num_calib_passes):
        acts_history.append(
            collect_activations(model, calib_loader, device, config.max_activation_samples)
        )

    thresholds = {}
    layer_names = [name for name in SILU_ACTIVATION_MODULES if acts_history and name in acts_history[0]]
    for layer_name in layer_names:
        all_activations = [acts[layer_name] for acts in acts_history if layer_name in acts]
        if all_activations:
            thresholds[layer_name] = compute_ncnn_threshold(
                np.concatenate(all_activations), config
            )
    return thresholds


def calibrate_silu_aware(model, calib_loader, device=DEVICE, config=None):
    if config is None:
        config = QuantizationConfig(strategy="silu_aware")
    acts = collect_activations(model, calib_loader, device, config.max_activation_samples)
    return {
        layer_name: compute_silu_aware_thresholds(layer_activations, config)
        for layer_name, layer_activations in acts.items()
        if layer_name in SILU_ACTIVATION_MODULES
    }


def _set_module(model, name, module):
    parent = model
    path = name.split(".")
    for part in path[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path[-1], module)


def insert_ncnn_quantizers(model, thresholds, bits=8):
    quantized_layers = []
    modules = dict(model.named_modules())
    for name in SILU_ACTIVATION_MODULES:
        module = modules.get(name)
        if isinstance(module, nn.SiLU) and name in thresholds:
            _set_module(
                model,
                name,
                nn.Sequential(module, ActivationQuantizerNCNN(thresholds[name], bits, use_ste=False)),
            )
            quantized_layers.append(name)
    return quantized_layers


def insert_silu_aware_quantizers(model, thresholds, bits=8):
    quantized_layers = []
    modules = dict(model.named_modules())
    for name in SILU_ACTIVATION_MODULES:
        module = modules.get(name)
        if isinstance(module, nn.SiLU) and name in thresholds:
            layer_thresholds = thresholds[name]
            _set_module(
                model,
                name,
                nn.Sequential(
                    module,
                    ActivationQuantizerSiLUAware(
                        layer_thresholds["vmax"], layer_thresholds["vsplit"], bits, use_ste=False
                    ),
                ),
            )
            quantized_layers.append(name)
    return quantized_layers


def build_quantized_pair(model, calib_loader, device=DEVICE, bits=8, bias_correction=True):
    ncnn_model = copy.deepcopy(model).to(device)
    silu_model = copy.deepcopy(model).to(device)

    ncnn_thresholds = calibrate_ncnn_style(ncnn_model, calib_loader, device, QuantizationConfig(bits=bits, strategy="ncnn"))
    silu_thresholds = calibrate_silu_aware(silu_model, calib_loader, device, QuantizationConfig(bits=bits, strategy="silu_aware"))

    ncnn_layers = insert_ncnn_quantizers(ncnn_model, ncnn_thresholds, bits=bits)
    silu_layers = insert_silu_aware_quantizers(silu_model, silu_thresholds, bits=bits)

    ncnn_corrections = {}
    silu_corrections = {}
    if bias_correction:
        silu_corrections = apply_bias_correction(model, silu_model, calib_loader, device, max_batches=1)

    return {
        "ncnn": (ncnn_model, ncnn_thresholds, ncnn_layers, ncnn_corrections),
        "silu_aware": (silu_model, silu_thresholds, silu_layers, silu_corrections),
    }


calibrate_dual_scale = calibrate_silu_aware
insert_dual_scale_quantizers = insert_silu_aware_quantizers
