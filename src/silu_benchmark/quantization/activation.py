"""Activation quantizer modules."""

import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):
    def __init__(self, threshold, bits=8, use_ste=False, symmetric=True):
        super().__init__()
        self.bits = bits
        self.threshold = float(threshold)
        self.use_ste = use_ste
        self.symmetric = symmetric
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
        self.scale = self.qmax / max(self.threshold, 1e-6)

    def quantize_dequantize(self, x):
        if self.symmetric:
            clipped = torch.clamp(x, -self.threshold, self.threshold)
            quantized = torch.round(clipped * self.scale)
            quantized = torch.clamp(quantized, -self.qmax, self.qmax)
            return quantized / self.scale

        clipped = torch.clamp(x, 0, self.threshold)
        quantized = torch.round(clipped * self.scale)
        quantized = torch.clamp(quantized, 0, self.qmax)
        return quantized / self.scale

    def forward(self, x):
        if self.use_ste and self.training:
            with torch.no_grad():
                dequantized = self.quantize_dequantize(x)
            return x + (dequantized - x).detach()
        return self.quantize_dequantize(x)


class ActivationQuantizerNCNN(BaseQuantizer):
    def __init__(self, vmax, bits=8, use_ste=False):
        super().__init__(vmax, bits, use_ste, symmetric=True)


class ActivationQuantizerSiLUAware(nn.Module):
    def __init__(self, vmax, vsplit, bits=8, use_ste=False):
        super().__init__()
        self.bits = bits
        self.vmax = float(vmax)
        self.vsplit = float(vsplit)
        self.use_ste = use_ste
        self.qmax = 2 ** bits - 1
        self.pos_scale = max(self.vmax / self.qmax, 1e-6)
        self.neg_scale = max(abs(self.vsplit) / self.qmax, 1e-6)

    def quantize_dequantize(self, x):
        out = torch.zeros_like(x)
        pos_mask = x >= 0
        neg_mask = x < 0

        if pos_mask.any():
            pos = torch.clamp(x[pos_mask], 0, self.vmax)
            out[pos_mask] = torch.round(pos / self.pos_scale) * self.pos_scale

        if neg_mask.any():
            neg = torch.clamp(x[neg_mask], self.vsplit, 0)
            out[neg_mask] = torch.round(neg / self.neg_scale) * self.neg_scale

        return out

    def forward(self, x):
        if self.use_ste and self.training:
            with torch.no_grad():
                dequantized = self.quantize_dequantize(x)
            return x + (dequantized - x).detach()
        return self.quantize_dequantize(x)
