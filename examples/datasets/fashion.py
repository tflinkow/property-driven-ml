import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

from property_driven_ml.training.mode import Mode

from examples.models import MnistNetSmall


def create_fashion_mnist_datasets(
    batch_size: int,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.nn.Module,
    Tuple[Tuple[float, ...], Tuple[float, ...]],
    Mode,
]:
    mean, std = (0.2860,), (0.3530,)

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_train = datasets.FashionMNIST(
        "data", train=True, download=True, transform=transform_train
    )
    dataset_test = datasets.FashionMNIST(
        "data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = MnistNetSmall()

    return train_loader, test_loader, model, (mean, std), Mode.MultiClassClassification
