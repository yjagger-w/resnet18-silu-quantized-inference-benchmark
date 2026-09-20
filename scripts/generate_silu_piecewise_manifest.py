"""Generate the real v0.6 call-site calibration manifest from CIFAR-10."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from export_onnx import load_checkpoint
from silu_benchmark.backends import discover_silu_patterns
from silu_benchmark.calibration_manifest import (
    build_manifest,
    collect_silu_callsite_activations,
    sha256_file,
    write_manifest_atomic,
)
from silu_benchmark.models import ResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a positive-Vsplit 17-site SiLU calibration manifest.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet18_cifar10.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--onnx-baseline", type=Path, default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-calibration-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, default=Path("configs/calibration/resnet18_silu_piecewise_v06.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {args.checkpoint.resolve()}")
    if not args.onnx_baseline.is_file():
        raise FileNotFoundError(f"baseline ONNX missing: {args.onnx_baseline.resolve()}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=transform)
    sample_count = args.batch_size * args.num_calibration_batches
    if sample_count > len(trainset):
        raise ValueError(f"requested {sample_count} calibration samples from {len(trainset)} training samples")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(trainset), generator=generator)[:sample_count].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, indices), batch_size=args.batch_size, shuffle=False)
    model = ResNet18().eval()
    load_checkpoint(model, args.checkpoint, torch.device("cpu"))
    onnx_sites = discover_silu_patterns(onnx.load(str(args.onnx_baseline)))
    activations = collect_silu_callsite_activations(model, loader, torch.device("cpu"))
    if list(activations) != [site.site_id for site in onnx_sites]:
        raise RuntimeError("PyTorch call-site order does not match discovered ONNX SiLU-site identities")
    manifest = build_manifest(
        site_activations=activations,
        metadata={
            "model_architecture": "ResNet18-SiLU-CIFAR10",
            "checkpoint": {"logical_id": args.checkpoint.as_posix(), "sha256": sha256_file(args.checkpoint)},
            "calibration_dataset": {"logical_id": "CIFAR-10/train", "data_root": args.data_root.as_posix(), "preprocessing": "ToTensor; Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])"},
            "calibration": {"batch_size": args.batch_size, "num_batches": args.num_calibration_batches, "sample_count": sample_count, "seed": args.seed, "sample_indices": indices},
            "kld_vmax_search": {"num_bins": 2048, "target_bins": 256, "percentile": 99.99},
            "positive_vsplit_mse_search": {"candidate_fraction_range": [0.05, 0.8], "num_candidates": 128},
        },
    )
    write_manifest_atomic(manifest, args.output)
    print(f"Manifest: {args.output}")
    print(f"Call sites: {len(manifest['sites'])}")
    print(f"Vsplit range: {min(site['vsplit'] for site in manifest['sites']):.9g} .. {max(site['vsplit'] for site in manifest['sites']):.9g}")


if __name__ == "__main__":
    main()
