"""Real-checkpoint/full-model numerical closure for the v0.6 manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from export_onnx import export_onnx, load_checkpoint
from silu_benchmark.backends import discover_silu_patterns, inspect_onnx_model, load_site_spec_manifest, rewrite_silu_piecewise_model, save_rewrite_result, write_inspection_report
from silu_benchmark.calibration_manifest import sha256_file
from silu_benchmark.data import cifar10_transform
from silu_benchmark.models import ResNet18
from silu_benchmark.quantization import piecewise_quantize_dequantize


class CallsitePiecewiseSiLU(nn.Module):
    def __init__(self, module_path, state, site_specs):
        super().__init__()
        self.module_path = module_path
        self.state = state
        self.site_specs = site_specs

    def forward(self, values):
        ordinal = self.state.get(self.module_path, 0)
        self.state[self.module_path] = ordinal + 1
        site_id = f"{self.module_path}.call_{ordinal}"
        if site_id not in self.site_specs:
            raise RuntimeError(f"missing manifest spec for PyTorch site {site_id}")
        return piecewise_quantize_dequantize(torch.nn.functional.silu(values), self.site_specs[site_id]).to(values.dtype)


def _set_module(model, path, module):
    parent = model
    for part in path.split(".")[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path.split(".")[-1], module)


def install_callsite_reference(model, site_specs):
    state = {}
    module_paths = sorted({site_id.rsplit(".call_", 1)[0] for site_id in site_specs})
    for path in module_paths:
        _set_module(model, path, CallsitePiecewiseSiLU(path, state, site_specs))
    model.register_forward_pre_hook(lambda _module, _inputs: state.clear())


def metrics(left, right):
    error = np.abs(left - right)
    index = tuple(int(item) for item in np.unravel_index(np.argmax(error), error.shape))
    return {"max_abs_error": float(error[index]), "mean_abs_error": float(error.mean()), "worst_index": list(index), "left_value": float(left[index]), "right_value": float(right[index])}


def main():
    parser = argparse.ArgumentParser(description="Validate real piecewise ResNet18 ONNX rewrite.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet18_cifar10.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/calibration/resnet18_silu_piecewise_v06.json"))
    parser.add_argument("--baseline-output", type=Path, default=Path("artifacts/onnx/resnet18_silu_fp32_phase31.onnx"))
    parser.add_argument("--rewritten-output", type=Path, default=Path("artifacts/onnx/resnet18_silu_piecewise_reference_phase31.onnx"))
    parser.add_argument("--report", type=Path, default=Path("results/silu_piecewise_phase31_validation.json"))
    parser.add_argument("--validation-index", type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(20260827)
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=cifar10_transform())
    image, _label = testset[args.validation_index]
    values = image.unsqueeze(0)
    model = ResNet18().eval()
    load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    export_onnx(model, args.baseline_output, values, opset=18)
    baseline_graph = onnx.load(str(args.baseline_output))
    sites = discover_silu_patterns(baseline_graph)
    site_specs = load_site_spec_manifest(args.manifest)
    if set(site_specs) != {site.site_id for site in sites}:
        raise ValueError("manifest and exported ONNX site IDs do not match exactly")
    baseline_session = ort.InferenceSession(str(args.baseline_output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        baseline_torch = model(values).numpy()
    baseline_ort = baseline_session.run(None, {"images": values.numpy()})[0]
    result = rewrite_silu_piecewise_model(baseline_graph, site_specs)
    save_rewrite_result(result, args.rewritten_output)
    reference_model = copy.deepcopy(model).eval()
    install_callsite_reference(reference_model, site_specs)
    with torch.no_grad():
        piecewise_torch = reference_model(values).numpy()
    piecewise_ort = ort.InferenceSession(str(args.rewritten_output), providers=["CPUExecutionProvider"]).run(None, {"images": values.numpy()})[0]
    report = inspect_onnx_model(args.rewritten_output)
    report["phase31_validation"] = {
        "checkpoint": {"logical_id": args.checkpoint.as_posix(), "sha256": sha256_file(args.checkpoint)},
        "manifest": {"logical_id": args.manifest.as_posix(), "sha256": sha256_file(args.manifest)},
        "validation_input": {"dataset": "CIFAR-10/test", "index": args.validation_index, "shape": list(values.shape)},
        "baseline_pytorch_vs_ort": metrics(baseline_torch, baseline_ort),
        "piecewise_pytorch_vs_ort": metrics(piecewise_torch, piecewise_ort),
    }
    write_inspection_report(report, args.report)
    print(json.dumps(report["phase31_validation"], indent=2))


if __name__ == "__main__":
    main()
