import json
import unittest
from pathlib import Path

from silu_benchmark.calibration_manifest import validate_manifest


class OrtNativeManifestTests(unittest.TestCase):
    def test_real_ort_manifest_provenance_when_available(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "calibration" / "resnet18_silu_piecewise_v06_ort_cpu.json"
        if not path.exists():
            self.skipTest("real ORT-native manifest prerequisite missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["calibration_backend"], "onnxruntime")
        self.assertEqual(payload["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(len(validate_manifest(payload)), 17)
        self.assertEqual(len(payload["onnx_site_order"]), 17)


if __name__ == "__main__":
    unittest.main()
