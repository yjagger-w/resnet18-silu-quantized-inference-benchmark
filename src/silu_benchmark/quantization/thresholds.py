"""Threshold search for NCNN and SiLU-aware activation quantization."""

import numpy as np

from silu_benchmark.config import QuantizationConfig


def ncnn_kld_threshold_optimized(data, num_bins=2048, target_bins=128, percentile=99.99):
    if len(data) == 0:
        return 1.0

    data = np.abs(data)

    if len(data) > 1000000:
        sample_size = min(1000000, len(data))
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]

    if len(data) > 1000:
        q_high = np.percentile(data, percentile)
        data = data[data <= q_high]

    if len(data) == 0:
        return 1.0

    max_val = np.max(data)
    if max_val < 1e-6:
        return 1e-6

    if len(data) < 1000:
        num_bins = 512

    hist, _ = np.histogram(data, bins=num_bins, range=(0, max_val))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return max_val

    distribution = hist / hist_sum
    best_threshold_idx = target_bins
    min_kl = float("inf")

    for i in range(num_bins // 2, num_bins):
        p = distribution[:i].copy()
        p[-1] += np.sum(distribution[i:])
        p_sum = p.sum()
        if p_sum > 0:
            p = p / p_sum

        num_per_bin = i / target_bins
        q = np.zeros_like(p)
        for j in range(target_bins):
            start = int(j * num_per_bin)
            end = int((j + 1) * num_per_bin)
            if j == target_bins - 1:
                end = i
            if end > start:
                bin_sum = p[start:end].sum()
                q[start:end] = bin_sum / (end - start)

        eps = 1e-12
        p_safe = p + eps
        q_safe = q + eps
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        kl = np.sum(p_safe * np.log(p_safe / q_safe))

        if kl < min_kl:
            min_kl = kl
            best_threshold_idx = i

    bin_width = max_val / num_bins
    return (best_threshold_idx + 0.5) * bin_width


def _fake_quant_silu(data, vmax, vsplit, bits):
    qmax = 2 ** bits - 1
    vmax = max(float(vmax), 1e-6)
    vsplit = min(float(vsplit), -1e-6)
    out = np.zeros_like(data, dtype=np.float64)

    pos_mask = data >= 0
    if np.any(pos_mask):
        pos = np.clip(data[pos_mask], 0.0, vmax)
        pos_scale = vmax / qmax
        out[pos_mask] = np.round(pos / pos_scale) * pos_scale

    neg_mask = data < 0
    if np.any(neg_mask):
        neg = np.clip(data[neg_mask], vsplit, 0.0)
        neg_scale = abs(vsplit) / qmax
        out[neg_mask] = np.round(neg / neg_scale) * neg_scale

    return out


def mse_split_threshold(data, vmax, bits=8, num_candidates=128):
    neg_data = data[data < 0]
    if len(neg_data) == 0:
        return 0.0

    neg_min = float(np.min(neg_data))
    if neg_min >= -1e-6:
        return -1e-6

    candidates = np.linspace(neg_min, -1e-6, num_candidates)
    best_split = neg_min
    best_mse = float("inf")
    eval_data = data
    if len(eval_data) > 200000:
        indices = np.random.choice(len(eval_data), 200000, replace=False)
        eval_data = eval_data[indices]

    for candidate in candidates:
        recon = _fake_quant_silu(eval_data, vmax, candidate, bits)
        mse = np.mean((eval_data - recon) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_split = float(candidate)

    return best_split


def compute_ncnn_threshold(data, config=None):
    if config is None:
        config = QuantizationConfig(strategy="ncnn")
    return ncnn_kld_threshold_optimized(
        data, target_bins=config.get_target_bins(), percentile=config.percentile
    )


def compute_silu_aware_thresholds(data, config=None):
    if config is None:
        config = QuantizationConfig(strategy="silu_aware")
    vmax = ncnn_kld_threshold_optimized(
        data[data > 0] if np.any(data > 0) else data,
        target_bins=config.get_target_bins(),
        percentile=config.percentile,
    )
    vsplit = mse_split_threshold(data, vmax, bits=config.bits)
    return {"vmax": float(vmax), "vsplit": float(vsplit)}
