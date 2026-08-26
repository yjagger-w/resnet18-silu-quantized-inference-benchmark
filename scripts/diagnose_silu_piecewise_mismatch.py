"""Localize real PyTorch/ORT piecewise mismatches without changing the graph."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from export_onnx import export_onnx, load_checkpoint
from silu_benchmark.backends import discover_silu_patterns, load_site_spec_manifest, rewrite_silu_piecewise_model, save_rewrite_result
from silu_benchmark.calibration_manifest import sha256_file
from silu_benchmark.data import cifar10_transform
from silu_benchmark.models import ResNet18
from silu_benchmark.quantization import piecewise_dequantize, piecewise_quantize


def _set_module(model, path, module):
    parent = model
    for part in path.split(".")[:-1]: parent = getattr(parent, part)
    setattr(parent, path.split(".")[-1], module)


def _debug_model(model, names):
    debug = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    known = {item.name: item for item in [*debug.graph.input, *debug.graph.output, *debug.graph.value_info]}
    for name in names:
        if name not in known: raise ValueError(f"debug tensor lacks inferred type: {name}")
        debug.graph.output.append(copy.deepcopy(known[name]))
    onnx.checker.check_model(debug)
    return debug


def _hook_baseline(model, sites, values):
    records, counts = {}, {}
    def reset(_m, _i): counts.clear()
    def hook(path):
        def capture(_m, inputs, output):
            ordinal = counts.get(path, 0); counts[path] = ordinal + 1
            records[f"{path}.call_{ordinal}"] = {"pre": inputs[0].detach().cpu().numpy(), "silu": output.detach().cpu().numpy()}
        return capture
    model.register_forward_pre_hook(reset)
    hooks = [module.register_forward_hook(hook(path)) for path, module in model.named_modules() if isinstance(module, nn.SiLU)]
    with torch.no_grad(): logits = model(values).numpy()
    for item in hooks: item.remove()
    if set(records) != {site.site_id for site in sites}: raise RuntimeError("PyTorch baseline hook site ordering mismatch")
    return logits, records


class _DebugPiecewise(nn.Module):
    def __init__(self, path, counts, specs, records):
        super().__init__(); self.path, self.counts, self.specs, self.records = path, counts, specs, records
    def forward(self, values):
        ordinal = self.counts.get(self.path, 0); self.counts[self.path] = ordinal + 1
        site_id = f"{self.path}.call_{ordinal}"; spec = self.specs[site_id]
        # This is the exported ONNX SiLU expression, deliberately not nn.SiLU.
        silu = values * torch.sigmoid(values)
        codes = piecewise_quantize(silu, spec)
        dequantized = piecewise_dequantize(codes, spec).to(values.dtype)
        self.records[site_id] = {"pre": values.detach().cpu().numpy(), "silu": silu.detach().cpu().numpy(), "codes": codes.detach().cpu().numpy().astype(np.uint8), "dequantized": dequantized.detach().cpu().numpy()}
        return dequantized


def _stats(values):
    return {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "sha256": __import__("hashlib").sha256(values.tobytes()).hexdigest()}


def _margin(values, spec):
    clipped = np.clip(np.asarray(values, dtype=np.float64), spec.vmin, spec.vmax)
    lower = clipped < spec.vsplit
    affine = np.where(lower, clipped / spec.lower_scale + spec.lower_zero_point, clipped / spec.upper_scale + spec.upper_zero_point)
    half_distance = np.abs(affine - (np.floor(affine) + 0.5))
    scales = np.where(lower, spec.lower_scale, spec.upper_scale)
    return {"min_rounding_code_margin": float(half_distance.min()), "min_rounding_input_margin": float((half_distance * scales).min()), "min_vsplit_margin": float(np.abs(clipped - spec.vsplit).min()), "lower_count": int(lower.sum()), "upper_count": int((~lower).sum())}


def _affine_detail(value, spec):
    clipped = float(np.clip(value, spec.vmin, spec.vmax)); lower = clipped < spec.vsplit
    scale, zero_point = (spec.lower_scale, spec.lower_zero_point) if lower else (spec.upper_scale, spec.upper_zero_point)
    affine = clipped / scale + zero_point
    boundary = np.floor(affine) + 0.5
    return {"segment": "lower" if lower else "upper", "unclipped_affine_code": float(affine), "nearest_half_integer": float(boundary), "rounding_code_margin": float(abs(affine - boundary)), "rounding_input_margin": float(abs(affine - boundary) * scale), "vsplit_margin": float(abs(clipped - spec.vsplit)), "clip_margin": float(min(clipped - spec.vmin, spec.vmax - clipped))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose a real v0.6 PyTorch/ORT mismatch.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet18_cifar10.pth")); parser.add_argument("--data-root", type=Path, default=Path("data")); parser.add_argument("--manifest", type=Path, default=Path("configs/calibration/resnet18_silu_piecewise_v06.json")); parser.add_argument("--report", type=Path, default=Path("results/silu_piecewise_phase32_diagnostic.json")); parser.add_argument("--validation-index", type=int, default=123)
    args = parser.parse_args(); torch.manual_seed(20260827)
    image, _ = torchvision.datasets.CIFAR10(args.data_root, train=False, download=False, transform=cifar10_transform())[args.validation_index]
    values = image.unsqueeze(0); model = ResNet18().eval(); load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    baseline_path = ROOT / "artifacts/onnx/resnet18_silu_fp32_phase32_debug.onnx"; export_onnx(model, baseline_path, values, 18)
    baseline = onnx.load(str(baseline_path)); sites = discover_silu_patterns(baseline); specs = load_site_spec_manifest(args.manifest)
    rewritten = rewrite_silu_piecewise_model(baseline, specs); rewritten_path = ROOT / "artifacts/onnx/resnet18_silu_piecewise_phase32_debug.onnx"; save_rewrite_result(rewritten, rewritten_path)
    torch_logits, torch_base = _hook_baseline(copy.deepcopy(model).eval(), sites, values)
    base_debug = _debug_model(baseline, [name for site in sites for name in (site.input_tensor, site.output_tensor)])
    base_out = ort.InferenceSession(base_debug.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"images": values.numpy()})
    base_names = [item.name for item in base_debug.graph.output]; ort_base = {name: array for name, array in zip(base_names[1:], base_out[1:])}
    records, counts = {}, {}; reference = copy.deepcopy(model).eval()
    for path in sorted({site.module_path for site in sites}): _set_module(reference, path, _DebugPiecewise(path, counts, specs, records))
    reference.register_forward_pre_hook(lambda _m, _i: counts.clear())
    with torch.no_grad(): torch_piecewise = reference(values).numpy()
    code_names = [rewritten.inserted_outputs[site.site_id]["quantized_codes_tensor"] for site in sites]; deq_names = [rewritten.inserted_outputs[site.site_id]["dequantized_output_tensor"] for site in sites]
    rw_silu_names = ["silu_piecewise_" + site.site_id.replace(".", "_") + "_silu_output" for site in sites]
    rw_debug = _debug_model(rewritten.model, [*code_names, *deq_names, *[site.input_tensor for site in sites], *rw_silu_names])
    rw_out = ort.InferenceSession(rw_debug.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"images": values.numpy()})
    rw_names = [item.name for item in rw_debug.graph.output]; ort_rw = {name: array for name, array in zip(rw_names[1:], rw_out[1:])}
    site_rows=[]; first=None; total=0
    for site in sites:
        spec = specs[site.site_id]
        py = records[site.site_id]
        ort_pre = ort_rw[site.input_tensor]
        ort_silu = ort_rw["silu_piecewise_" + site.site_id.replace(".", "_") + "_silu_output"]
        codes = ort_rw[rewritten.inserted_outputs[site.site_id]["quantized_codes_tensor"]]
        deq = ort_rw[rewritten.inserted_outputs[site.site_id]["dequantized_output_tensor"]]
        mismatch=np.argwhere(py["codes"] != codes); count=int(mismatch.shape[0]); total += count
        row = {
            "site_id": site.site_id, "module_path": site.module_path,
            "invocation_index": site.call_index,
            "spec": {"vmin": spec.vmin, "vsplit": spec.vsplit, "vmax": spec.vmax},
            "pytorch_pre": _stats(py["pre"]),
            "baseline_pre_max_error": float(np.max(np.abs(torch_base[site.site_id]["pre"] - ort_base[site.input_tensor]))),
            "baseline_silu_max_error": float(np.max(np.abs(torch_base[site.site_id]["silu"] - ort_base[site.output_tensor]))),
            "piecewise_pre_max_error": float(np.max(np.abs(py["pre"] - ort_pre))),
            "piecewise_silu_max_error": float(np.max(np.abs(py["silu"] - ort_silu))),
            "nn_silu_vs_expression_max_error": float(np.max(np.abs(torch_base[site.site_id]["silu"] - (torch_base[site.site_id]["pre"] * (1.0 / (1.0 + np.exp(-torch_base[site.site_id]["pre"]))))))),
            "code_mismatch_count": count,
            "dequantized_max_error": float(np.max(np.abs(py["dequantized"] - deq))),
            "dequantized_mean_error": float(np.mean(np.abs(py["dequantized"] - deq))),
            "margin": _margin(py["silu"], spec),
        }
        if count:
            idx=tuple(int(x) for x in mismatch[0]); row["first_code_mismatch"]={"index":list(idx),"pytorch_pre":float(py["pre"][idx]),"ort_pre":float(ort_pre[idx]),"pre_abs_diff":float(abs(py["pre"][idx]-ort_pre[idx])),"pytorch_silu":float(py["silu"][idx]),"ort_silu":float(ort_silu[idx]),"pytorch_affine":_affine_detail(py["silu"][idx],spec),"ort_affine":_affine_detail(ort_silu[idx],spec),"pytorch_code":int(py["codes"][idx]),"ort_code":int(codes[idx]),"pytorch_dequantized":float(py["dequantized"][idx]),"ort_dequantized":float(deq[idx])}
            if first is None: first=row
        site_rows.append(row)
    final=np.abs(torch_piecewise-rw_out[0]); report={"checkpoint_sha256":sha256_file(args.checkpoint),"manifest_sha256":sha256_file(args.manifest),"input":{"dataset":"CIFAR-10/test","index":args.validation_index,"shape":list(values.shape),"seed":20260827},"site_count":len(sites),"total_code_mismatch_count":total,"first_divergence":first,"sites":site_rows,"final":{"max_abs_error":float(final.max()),"mean_abs_error":float(final.mean()),"worst_index":[int(x) for x in np.unravel_index(np.argmax(final),final.shape)]}}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"total_code_mismatch_count":total,"first_divergence":first,"final":report["final"]},indent=2))

if __name__ == "__main__": main()
