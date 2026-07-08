"""Experiment and quantization configuration."""

from dataclasses import dataclass

SILU_ACTIVATION_MODULES = (
    "act",
    "layer1.0.act",
    "layer1.1.act",
    "layer2.0.act",
    "layer2.1.act",
    "layer3.0.act",
    "layer3.1.act",
    "layer4.0.act",
    "layer4.1.act",
)


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int = 8
    strategy: str = "ncnn"
    percentile: float = 99.99
    num_calib_passes: int = 3
    max_activation_samples: int = 10000

    def __post_init__(self):
        if self.bits <= 4:
            object.__setattr__(self, "percentile", 99.7)
            object.__setattr__(self, "num_calib_passes", 8)
            object.__setattr__(self, "max_activation_samples", 8000)
        elif self.bits <= 8:
            object.__setattr__(self, "percentile", 99.99)
            object.__setattr__(self, "num_calib_passes", 3)
            object.__setattr__(self, "max_activation_samples", 10000)

    def get_target_bins(self):
        if self.strategy == "silu_aware":
            return 2 ** self.bits
        return 2 ** (self.bits - 1)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int = 128
    test_batch_size: int = 128
    calib_batches: int = 20
    data_root: str = "./data"
    weights_path: str = "checkpoints/resnet18_cifar10.pth"

    @property
    def calibration_samples(self):
        return self.batch_size * self.calib_batches
