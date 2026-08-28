from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silu_benchmark.quantization import PiecewiseQuantizationSpec, piecewise_quantize


DEFAULT_MANIFEST = ROOT / "configs/calibration/resnet18_silu_piecewise_v06.json"
DEFAULT_JSON = ROOT / "cpp/tests/data/act_call_0_golden.json"
DEFAULT_HEADER = ROOT / "cpp/tests/data/act_call_0_golden.h"
SITE_ID = "act.call_0"
SEED = 20260828
INPUT_SCALE = 0.0625
INPUT_ZERO_POINT = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the small v1.0 C++ SiLU kernel golden fixture."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--header-output", type=Path, default=DEFAULT_HEADER)
    return parser.parse_args()


def stable_silu_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.empty_like(values)
    positive = values >= np.float32(0.0)
    output[positive] = values[positive] / (
        np.float32(1.0) + np.exp(-values[positive])
    )
    exp_value = np.exp(values[~positive])
    output[~positive] = (
        values[~positive] * exp_value / (np.float32(1.0) + exp_value)
    )
    return output


def selected_site(manifest: dict) -> dict:
    matches = [entry for entry in manifest["sites"] if entry["site_id"] == SITE_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {SITE_ID!r} manifest entry")
    return matches[0]


def format_array(values: list, formatter, indent: str = "    ") -> str:
    tokens = [formatter(value) for value in values]
    lines = []
    for start in range(0, len(tokens), 12):
        lines.append(indent + ", ".join(tokens[start : start + 12]))
    return ",\n".join(lines)


def make_header(fixture: dict) -> str:
    output = fixture["output_piecewise_params"]
    inputs = fixture["quantized_input_codes"]
    expected = fixture["expected_output_codes"]
    probes = fixture["post_silu_reference_values"]
    probe_expected = fixture["post_silu_expected_codes"]
    return f'''#ifndef SILU_BENCHMARK_TESTS_DATA_ACT_CALL_0_GOLDEN_H_
#define SILU_BENCHMARK_TESTS_DATA_ACT_CALL_0_GOLDEN_H_

#include <array>
#include <cstdint>

namespace silu_benchmark::golden {{

inline constexpr double kInputScale = {fixture["input_uniform_params"]["scale"]!r};
inline constexpr std::int32_t kInputZeroPoint = {fixture["input_uniform_params"]["zero_point"]};
inline constexpr double kVmin = {output["vmin"]!r};
inline constexpr double kVsplit = {output["vsplit"]!r};
inline constexpr double kVmax = {output["vmax"]!r};
inline constexpr double kLowerScale = {output["lower_scale"]!r};
inline constexpr double kUpperScale = {output["upper_scale"]!r};
inline constexpr std::int32_t kLowerZeroPoint = {output["lower_zero_point"]};
inline constexpr std::int32_t kUpperZeroPoint = {output["upper_zero_point"]};

inline constexpr std::array<std::uint8_t, {len(inputs)}> kInputCodes = {{{{
{format_array(inputs, lambda value: str(value))}
}}}};

inline constexpr std::array<std::uint8_t, {len(expected)}> kExpectedOutputCodes = {{{{
{format_array(expected, lambda value: str(value))}
}}}};

inline constexpr std::array<double, {len(probes)}> kPostSiluReferenceValues = {{{{
{format_array(probes, lambda value: repr(value), indent="    ")}
}}}};

inline constexpr std::array<std::uint8_t, {len(probe_expected)}> kPostSiluExpectedCodes = {{{{
{format_array(probe_expected, lambda value: str(value))}
}}}};

}}  // namespace silu_benchmark::golden

#endif  // SILU_BENCHMARK_TESTS_DATA_ACT_CALL_0_GOLDEN_H_
'''


def build_fixture(manifest_path: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    site = selected_site(manifest)
    spec = PiecewiseQuantizationSpec(
        float(site["vmin"]), float(site["vsplit"]), float(site["vmax"]), int(site["bits"])
    )

    rng = np.random.default_rng(SEED)
    special_codes = np.asarray([0, 1, 126, 127, 128, 129, 254, 255], dtype=np.uint8)
    random_codes = rng.integers(0, 256, size=64, dtype=np.uint8)
    all_codes = np.arange(256, dtype=np.uint8)
    input_codes = np.concatenate([special_codes, random_codes, all_codes])
    dequantized = (
        input_codes.astype(np.int32) - INPUT_ZERO_POINT
    ).astype(np.float32) * np.float32(INPUT_SCALE)
    post_silu = stable_silu_float32(dequantized)
    expected = np.asarray(piecewise_quantize(post_silu, spec), dtype=np.uint8)

    lower_tie_even = (10.5 - spec.lower_zero_point) * spec.lower_scale
    lower_tie_odd = (11.5 - spec.lower_zero_point) * spec.lower_scale
    upper_tie_even = (200.5 - spec.upper_zero_point) * spec.upper_scale
    upper_tie_odd = (201.5 - spec.upper_zero_point) * spec.upper_scale
    random_post = rng.uniform(spec.vmin - 0.5, spec.vmax + 0.5, size=32)
    probes = np.concatenate(
        [
            np.asarray(
                [
                    spec.vmin - 1.0,
                    spec.vmin,
                    np.nextafter(spec.vmin, np.inf),
                    -0.0,
                    0.0,
                    lower_tie_even,
                    lower_tie_odd,
                    np.nextafter(spec.vsplit, -np.inf),
                    spec.vsplit,
                    np.nextafter(spec.vsplit, np.inf),
                    upper_tie_even,
                    upper_tie_odd,
                    np.nextafter(spec.vmax, -np.inf),
                    spec.vmax,
                    spec.vmax + 1.0,
                ],
                dtype=np.float64,
            ),
            random_post,
        ]
    )
    probe_expected = np.asarray(piecewise_quantize(probes, spec), dtype=np.uint8)

    return {
        "schema_version": "quantized-silu-kernel-golden/v1",
        "provenance": {
            "canonical_source": "src/silu_benchmark/quantization/activation.py::piecewise_quantize",
            "parameter_source": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "parameter_source_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "site_id": SITE_ID,
            "generator": "scripts/generate_cpp_silu_golden.py",
            "generator_seed": SEED,
            "numeric_path": "uint8 affine dequantize to float32; stable float32 SiLU; canonical float64 piecewise_quantize",
            "scope_note": "Input affine parameters exercise the standalone API and are not claimed as calibrated model parameters.",
        },
        "input_uniform_params": {
            "dtype": "uint8",
            "scale": INPUT_SCALE,
            "zero_point": INPUT_ZERO_POINT,
            "qmin": 0,
            "qmax": 255,
        },
        "output_piecewise_params": {
            "dtype": "uint8",
            "vmin": spec.vmin,
            "vsplit": spec.vsplit,
            "vmax": spec.vmax,
            "bits": spec.bits,
            "lower_scale": spec.lower_scale,
            "lower_zero_point": spec.lower_zero_point,
            "upper_scale": spec.upper_scale,
            "upper_zero_point": spec.upper_zero_point,
            "lower_codes": list(spec.lower_codes),
            "upper_codes": list(spec.upper_codes),
            "split_owner": "upper",
            "rounding": "nearest, ties to even",
        },
        "coverage": [
            "qmin/qmax and adjacent codes",
            "all 256 input codes",
            "64 deterministic random input codes",
            "Vmin/Vsplit/Vmax and adjacent float64 values",
            "lower and upper half-step ties with even and odd lower integers",
            "negative, zero, positive, and saturation probes",
            "32 deterministic random post-SiLU values",
        ],
        "quantized_input_codes": input_codes.astype(int).tolist(),
        "expected_output_codes": expected.astype(int).tolist(),
        "post_silu_reference_values": probes.tolist(),
        "post_silu_expected_codes": probe_expected.astype(int).tolist(),
    }


def main() -> int:
    args = parse_args()
    fixture = build_fixture(args.manifest.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.header_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.header_output.write_text(make_header(fixture), encoding="utf-8")
    print(f"wrote {args.json_output}")
    print(f"wrote {args.header_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
