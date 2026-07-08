"""CIFAR-10 data loaders."""

import torch
import torchvision
import torchvision.transforms as transforms

from .config import ExperimentConfig


def cifar10_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def build_cifar10_loaders(config=ExperimentConfig(), download=True):
    transform = cifar10_transform()
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=download, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=False, download=download, transform=transform
    )
    calib_indices = list(range(min(config.calibration_samples, len(trainset))))
    calib_subset = torch.utils.data.Subset(trainset, calib_indices)
    calib_loader = torch.utils.data.DataLoader(
        calib_subset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size, shuffle=False
    )
    return trainset, testset, calib_subset, calib_loader, test_loader
