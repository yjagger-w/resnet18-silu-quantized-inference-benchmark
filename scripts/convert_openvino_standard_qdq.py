"""Convert only the official Standard-QDQ ONNX to an immutable OpenVINO IR bundle."""

from run_openvino_standard_qdq_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(conversion_only=True))
