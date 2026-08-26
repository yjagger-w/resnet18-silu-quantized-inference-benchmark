"""Call-site-aware calibration and the v0.6 piecewise manifest contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

from .quantization import PiecewiseQuantizationSpec


SCHEMA_VERSION = "silu-piecewise-calibration-manifest/v1"
ALGORITHM = "silu-aware-piecewise-asymmetric-ptq"
CONTRACT = "v0.6 PiecewiseQuantizationSpec"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_silu_callsite_activations(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples_per_site: int = 10000,
) -> OrderedDict[str, np.ndarray]:
    """Collect distinct SiLU outputs keyed by module path and call ordinal."""
    import torch
    import torch.nn as nn

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.SiLU)]
    if not modules:
        raise ValueError("model has no nn.SiLU modules")
    samples: OrderedDict[str, list[np.ndarray]] = OrderedDict()
    try:
        per_batch_limit = max(1, max_samples_per_site // len(loader))
    except TypeError as exc:
        raise ValueError("calibration loader must define a finite length") from exc
    expected_order: list[str] | None = None
    active_order: list[str] = []
    counts: dict[str, int] = {}

    def reset(_module, _inputs):
        active_order.clear()
        counts.clear()

    def hook_for(name: str):
        def record(_module, _inputs, output):
            ordinal = counts.get(name, 0)
            counts[name] = ordinal + 1
            site_id = f"{name}.call_{ordinal}"
            active_order.append(site_id)
            values = output.detach().cpu().numpy().reshape(-1)
            if values.size > per_batch_limit:
                # Deterministic thinning avoids hidden RNG dependence.
                values = values[np.linspace(0, values.size - 1, per_batch_limit, dtype=np.int64)]
            samples.setdefault(site_id, []).append(values)
        return record

    pre_hook = model.register_forward_pre_hook(reset)
    hooks = [module.register_forward_hook(hook_for(name)) for name, module in modules]
    model.eval()
    try:
        with torch.no_grad():
            for images, _labels in loader:
                model(images.to(device))
                if expected_order is None:
                    expected_order = list(active_order)
                elif active_order != expected_order:
                    raise RuntimeError(
                        f"SiLU call-site ordering changed: expected={expected_order}, observed={active_order}"
                    )
    finally:
        pre_hook.remove()
        for hook in hooks:
            hook.remove()
    if not expected_order:
        raise RuntimeError("calibration loader produced no model forwards")
    return OrderedDict((site_id, np.concatenate(samples[site_id])) for site_id in expected_order)


def build_manifest(
    *,
    site_activations: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    bits: int = 8,
) -> dict:
    """Build an ordered, fully auditable positive-Vsplit manifest payload."""
    from .quantization.thresholds import compute_silu_aware_thresholds

    sites = []
    for site_id, activations in site_activations.items():
        if ".call_" not in site_id:
            raise ValueError(f"invalid call-site identity: {site_id}")
        module_path, ordinal = site_id.rsplit(".call_", 1)
        thresholds = compute_silu_aware_thresholds(np.asarray(activations), config=None)
        spec = PiecewiseQuantizationSpec(
            thresholds["vmin"], thresholds["vsplit"], thresholds["vmax"], bits
        )
        sites.append(
            {
                "site_id": site_id,
                "module_path": module_path,
                "invocation_index": int(ordinal),
                "vmin": spec.vmin,
                "vsplit": spec.vsplit,
                "vmax": spec.vmax,
                "bits": spec.bits,
                "lower_scale": spec.lower_scale,
                "lower_zero_point": spec.lower_zero_point,
                "upper_scale": spec.upper_scale,
                "upper_zero_point": spec.upper_zero_point,
                "code_ranges": {"lower": [0, 127], "upper": [128, 255], "split_owner": "upper"},
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "quantization_contract": CONTRACT,
        **dict(metadata),
        "actual_discovered_site_count": len(sites),
        "sites": sites,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(payload: Mapping[str, object], expected_site_ids: Sequence[str] | None = None) -> dict[str, PiecewiseQuantizationSpec]:
    """Validate and convert a manifest; legacy threshold JSON is rejected."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a v0.6 calibration manifest; legacy threshold JSON is unsupported")
    entries = payload.get("sites")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a non-empty ordered sites list")
    result: dict[str, PiecewiseQuantizationSpec] = {}
    identities = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest site entry must be an object")
        try:
            site_id = str(entry["site_id"])
            identity = (str(entry["module_path"]), int(entry["invocation_index"]))
            spec = PiecewiseQuantizationSpec(float(entry["vmin"]), float(entry["vsplit"]), float(entry["vmax"]), int(entry["bits"]))
        except KeyError as exc:
            raise ValueError(f"manifest site is missing {exc.args[0]}") from exc
        if site_id in result or identity in identities:
            raise ValueError(f"duplicate manifest site: {site_id}")
        if site_id != f"{identity[0]}.call_{identity[1]}":
            raise ValueError(f"site_id does not match logical identity: {site_id}")
        for key, actual in (("lower_scale", spec.lower_scale), ("upper_scale", spec.upper_scale), ("lower_zero_point", spec.lower_zero_point), ("upper_zero_point", spec.upper_zero_point)):
            if key not in entry or not np.isclose(float(entry[key]), actual, rtol=0.0, atol=1e-12):
                raise ValueError(f"derived {key} does not match canonical spec for {site_id}")
        result[site_id] = spec
        identities.add(identity)
    if payload.get("actual_discovered_site_count") != len(entries):
        raise ValueError("actual_discovered_site_count does not match sites")
    if expected_site_ids is not None and set(expected_site_ids) != set(result):
        raise ValueError("manifest site IDs do not exactly match discovered ONNX sites")
    return result


def load_manifest(path: Path, expected_site_ids: Sequence[str] | None = None) -> tuple[dict, dict[str, PiecewiseQuantizationSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, validate_manifest(payload, expected_site_ids)


def write_manifest_atomic(payload: Mapping[str, object], path: Path) -> None:
    validate_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
