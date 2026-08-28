#!/usr/bin/env python
"""Report v1.2 ORT C++ custom-op build prerequisites without installing anything."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from silu_benchmark.ort_custom_op_backend import detect_prerequisites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-root", type=Path)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()
    result = detect_prerequisites(args.ort_root)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.supported or not args.require else 2


if __name__ == "__main__":
    raise SystemExit(main())
