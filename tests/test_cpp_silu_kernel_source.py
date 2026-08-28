import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp"
FIXTURE_JSON = CPP / "tests/data/act_call_0_golden.json"
FIXTURE_HEADER = CPP / "tests/data/act_call_0_golden.h"


class CppSiluKernelSourceTests(unittest.TestCase):
    def test_kernel_library_has_no_framework_runtime_headers(self):
        forbidden = ("torch/", "onnxruntime", "openvino", "qnn", "onnx/")
        sources = (
            CPP / "include/silu_benchmark/quantized_silu_kernel.h",
            CPP / "src/quantized_silu_kernel.cpp",
        )
        for source in sources:
            text = source.read_text(encoding="utf-8").lower()
            for token in forbidden:
                with self.subTest(source=source.name, token=token):
                    self.assertNotIn(token, text)

    def test_ort_wrapper_is_separate_and_reuses_kernel_library(self):
        wrapper = (CPP / "src/ort_quantized_silu_custom_op.cpp").read_text(
            encoding="utf-8"
        )
        cmake = (CPP / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("#include <onnxruntime_cxx_api.h>", wrapper)
        self.assertIn("QuantizedSiluScalarUnchecked", wrapper)
        self.assertIn("add_library(silu_ort_custom_op SHARED", cmake)
        self.assertIn("silu_quantized_kernel", cmake)

    def test_golden_fixture_is_small_complete_and_provenanced(self):
        fixture = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schema_version"], "quantized-silu-kernel-golden/v1")
        self.assertEqual(fixture["provenance"]["site_id"], "act.call_0")
        self.assertIn("piecewise_quantize", fixture["provenance"]["canonical_source"])
        self.assertEqual(len(fixture["quantized_input_codes"]), len(fixture["expected_output_codes"]))
        self.assertEqual(set(fixture["quantized_input_codes"]), set(range(256)))
        self.assertLess(FIXTURE_JSON.stat().st_size, 100_000)

    def test_fixture_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            generated_json = Path(directory) / "fixture.json"
            generated_header = Path(directory) / "fixture.h"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/generate_cpp_silu_golden.py"),
                    "--json-output",
                    str(generated_json),
                    "--header-output",
                    str(generated_header),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(generated_json.read_bytes(), FIXTURE_JSON.read_bytes())
            self.assertEqual(generated_header.read_bytes(), FIXTURE_HEADER.read_bytes())

    def test_generated_paths_are_ignored_but_fixtures_are_tracked_candidates(self):
        checks = {
            "build/cpp-silu-kernel/example.obj": True,
            "results/benchmarks/v1.0_silu_kernel/example.json": True,
            "cpp/tests/data/act_call_0_golden.json": False,
        }
        for path, expected_ignored in checks.items():
            result = subprocess.run(
                ["git", "check-ignore", "-q", path], cwd=ROOT, timeout=10
            )
            self.assertEqual(result.returncode == 0, expected_ignored, path)


if __name__ == "__main__":
    unittest.main()
