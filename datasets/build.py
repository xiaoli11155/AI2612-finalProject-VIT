from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader


DATASET_META = {
    "cifar10": {"num_classes": 10, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"num_classes": 100, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}


def _build_transforms(image_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    train_ops = []
    if image_size != 32:
        train_ops.append(transforms.Resize((image_size, image_size)))
    train_ops.extend(
        [
            transforms.RandomCrop(image_size, padding=max(4, image_size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_transform = transforms.Compose(train_ops)

    eval_ops = []
    if image_size != 32:
        eval_ops.append(transforms.Resize((image_size, image_size)))
    eval_ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    eval_transform = transforms.Compose(eval_ops)
    return train_transform, eval_transform


def _split_train_val(dataset, val_split: float, seed: int):
    if val_split <= 0:
        return dataset, None
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)
    return train_subset, val_subset


class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _build_tiny_imagenet_val_dataset(val_dir: str, transform):
    val_path = Path(val_dir)
    class_dirs = [p for p in val_path.iterdir() if p.is_dir() and p.name != "images"]
    if class_dirs:
        return datasets.ImageFolder(root=val_dir, transform=transform)

    ann_path = val_path / "val_annotations.txt"
    images_dir = val_path / "images"
    if not ann_path.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet val split not found in expected format under: {val_dir}"
        )

    samples = []
    class_names = set()
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            image_path = images_dir / image_name
            if image_path.exists():
                samples.append((str(image_path), class_name))
                class_names.add(class_name)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_names))}
    indexed_samples = [(path, class_to_idx[class_name]) for path, class_name in samples]
    return TinyImageNetValDataset(indexed_samples, transform=transform)


def build_dataloaders(cfg: Dict, seed: int):
    dataset_name = cfg["name"].lower()
    data_dir = cfg.get("data_dir", "./data")
    image_size = int(cfg.get("image_size", 32))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    val_split = float(cfg.get("val_split", 0.1))

    if dataset_name in DATASET_META:
        meta = DATASET_META[dataset_name]
        train_transform, eval_transform = _build_transforms(image_size, meta["mean"], meta["std"])

        ds_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
        full_train = ds_cls(root=data_dir, train=True, download=True, transform=train_transform)
        train_set, val_set = _split_train_val(full_train, val_split=val_split, seed=seed)

        if isinstance(val_set, Subset):
            base = ds_cls(root=data_dir, train=True, download=True, transform=eval_transform)
            val_set = Subset(base, val_set.indices)
        test_set = ds_cls(root=data_dir, train=False, download=True, transform=eval_transform)
        num_classes = meta["num_classes"]
    elif dataset_name == "tiny_imagenet":
        train_transform, eval_transform = _build_transforms(
            image_size,
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2770, 0.2691, 0.2821),
        )
        train_dir = f"{data_dir}/train"
        val_dir = f"{data_dir}/val"
        full_train = datasets.ImageFolder(root=train_dir, transform=train_transform)
        train_set, val_set = _split_train_val(full_train, val_split=val_split, seed=seed)
        if isinstance(val_set, Subset):
            base = datasets.ImageFolder(root=train_dir, transform=eval_transform)
            val_set = Subset(base, val_set.indices)
        test_set = _build_tiny_imagenet_val_dataset(val_dir=val_dir, transform=eval_transform)
        num_classes = len(full_train.classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        ),
        "val": DataLoader(
            val_set if val_set is not None else test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, num_classes
