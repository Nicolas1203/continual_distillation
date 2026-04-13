from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils.paths import DATA_ROOT


DEFAULT_NUM_CLASSES = 20
DOMAINNET_NUM_CLASSES = 345
DIGITS_NUM_CLASSES = 10
DEFAULT_IMAGE_SIZE = 224
DEFAULT_NUM_WORKERS = 16

CIFAR20_COARSE_TO_FINE_LABELS = {
    0: [4, 30, 55, 72, 95],
    1: [1, 32, 67, 73, 91],
    2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61],
    4: [0, 51, 53, 57, 83],
    5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94],
    7: [6, 7, 14, 18, 24],
    8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76],
    10: [23, 33, 49, 60, 71],
    11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75],
    13: [26, 45, 77, 79, 99],
    14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93],
    16: [36, 50, 65, 74, 80],
    17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],
    19: [41, 69, 81, 85, 89],
}

FINE_TO_COARSE = [0 for _ in range(100)]
for coarse_label, fine_labels in CIFAR20_COARSE_TO_FINE_LABELS.items():
    for fine_label in fine_labels:
        FINE_TO_COARSE[fine_label] = coarse_label

DOMAINNET_DOMAIN_NAMES = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
]
DOMAINNET_PAIR_NAMES = {
    "pair_0_1": "clipart + infograph",
    "pair_0_2": "clipart + painting",
    "pair_0_3": "clipart + quickdraw",
    "pair_0_4": "clipart + real",
    "pair_0_5": "clipart + sketch",
}

DIGITS_DOMAIN_NAMES = {
    0: "mnist",
    1: "svhn",
    2: "mnist-m",
    3: "usps",
    4: "kmnist",
    5: "fashion-mnist",
}

DATASET_ALIASES = {
    "mixed": "mixed_cifar",
    "mixed-cifar": "mixed_cifar",
    "mixed_cifar": "mixed_cifar",
    "mixedd": "mixed_domainnet",
    "mixed-domainnet": "mixed_domainnet",
    "mixed_domainnet": "mixed_domainnet",
}


def _identity(value):
    return value


def normalize_dataset_name(dataset_name: str) -> str:
    """Map legacy dataset aliases to the maintained canonical names."""
    normalized = str(dataset_name).strip().lower()
    return DATASET_ALIASES.get(normalized, normalized)


def _repeat_grayscale_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1)
    return tensor


def _normalize_selected_domains(selected_domains, default_domain: int = 0) -> list[int]:
    if selected_domains is None:
        return [default_domain]
    if isinstance(selected_domains, int):
        return [int(selected_domains)]
    return [int(domain_id) for domain_id in selected_domains]


def get_domainnet_domain_names(domainnet_root=None) -> list[str]:
    root = Path(domainnet_root) if domainnet_root is not None else DATA_ROOT / "domainnet"
    if root.exists():
        folder_names = sorted(path.name for path in root.iterdir() if path.is_dir())
        if folder_names:
            return folder_names
    return list(DOMAINNET_DOMAIN_NAMES)


class SplitCIFAR20(datasets.CIFAR100):
    """View CIFAR-100 as five domain-specific splits of the CIFAR-20 coarse labels."""

    def __init__(self, root, train, transform, download=False, selected_domains=None):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.selected_domains = _normalize_selected_domains(selected_domains)
        self.fine_targets = torch.tensor(self.targets, dtype=torch.long)
        self.coarse_targets = torch.tensor(
            [FINE_TO_COARSE[fine_label] for fine_label in self.targets],
            dtype=torch.long,
        )
        self.coarse_to_fine = {
            coarse_label: list(fine_labels)
            for coarse_label, fine_labels in CIFAR20_COARSE_TO_FINE_LABELS.items()
        }

        selected_fine_labels: list[int] = []
        for domain_id in self.selected_domains:
            if domain_id < 0 or domain_id >= 5:
                raise ValueError(f"CIFAR20 domain ids must be in [0, 4], got {domain_id}.")
            for coarse_label in range(DEFAULT_NUM_CLASSES):
                selected_fine_labels.append(self.coarse_to_fine[coarse_label][domain_id])

        selected_tensor = torch.tensor(selected_fine_labels, dtype=torch.long)
        self.indexes = torch.nonzero(
            torch.isin(self.fine_targets, selected_tensor),
            as_tuple=False,
        ).flatten()

    def __getitem__(self, index):
        real_index = int(self.indexes[index].item())
        image = Image.fromarray(self.data[real_index])
        fine_target = int(self.fine_targets[real_index].item())
        coarse_target = int(self.coarse_targets[real_index].item())
        domain_id = self.coarse_to_fine[coarse_target].index(fine_target)

        if self.transform is not None:
            image = self.transform(image)
        return image, coarse_target, domain_id, int(index)

    def __len__(self):
        return int(self.indexes.numel())


class MNISTM(Dataset):
    """Minimal MNIST-M implementation used for the digits benchmark."""

    def __init__(self, root, train=True, transform=None, download=False, background_source="cifar10"):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.train = train
        self.background_source = background_source

        rng = np.random.RandomState(42 if train else 123)
        if background_source.lower() == "cifar10":
            background_dataset = datasets.CIFAR10(root=root, train=True, download=download)
            self.background_images = [np.array(image) for image, _ in background_dataset]
        elif background_source.lower() == "svhn":
            background_dataset = datasets.SVHN(root=root, split="train", download=download)
            self.background_images = [np.transpose(image, (1, 2, 0)) for image, _ in background_dataset]
        else:
            self.background_images = None

        if self.background_images is not None:
            self.background_indices = rng.randint(0, len(self.background_images), size=len(self.mnist))
        else:
            self.background_indices = None

    def __getitem__(self, index):
        image, target = self.mnist[index]
        digit = np.array(image, dtype=np.uint8)
        digit_norm = digit.astype(np.float32) / 255.0

        if self.background_images is not None:
            background = self.background_images[int(self.background_indices[index])]
            if background.dtype != np.uint8:
                background = (np.clip(background, 0.0, 1.0) * 255).astype(np.uint8)
            background_image = Image.fromarray(background).resize((28, 28), Image.BILINEAR)
            background_array = np.array(background_image, dtype=np.uint8)
        else:
            # Preserve the historical deterministic fallback used by this codebase.
            rng = np.random.RandomState(42 if self.train else 123)
            background_array = rng.randint(0, 255, (28, 28, 3), dtype=np.uint8)

        composite = (
            digit_norm[..., None] * 255.0
            + (1.0 - digit_norm[..., None]) * background_array.astype(np.float32)
        ).astype(np.uint8)
        composite_image = Image.fromarray(composite)
        if self.transform is not None:
            composite_image = self.transform(composite_image)
        return composite_image, int(target)

    def __len__(self):
        return len(self.mnist)


def get_digits_dataset(dataset_name, data_root, train=True, download=True):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return datasets.MNIST(root=data_root, train=train, download=download, transform=None)
    if dataset_name == "svhn":
        split = "train" if train else "test"
        return datasets.SVHN(root=data_root, split=split, download=download, transform=None)
    if dataset_name == "usps":
        return datasets.USPS(root=data_root, train=train, download=download, transform=None)
    if dataset_name == "kmnist":
        return datasets.KMNIST(root=data_root, train=train, download=download, transform=None)
    if dataset_name == "mnist-m":
        return MNISTM(root=data_root, train=train, download=download, transform=None)
    if dataset_name == "fashion-mnist":
        return datasets.FashionMNIST(root=data_root, train=train, download=download, transform=None)
    raise ValueError(f"Unsupported digits dataset: {dataset_name}")


class SplitDigits(Dataset):
    def __init__(self, root, train, transform, download=False, selected_domains=None):
        self.transform = transform
        self.selected_domains = _normalize_selected_domains(selected_domains)
        self.datasets_by_domain = {
            domain_id: get_digits_dataset(domain_name, root, train=train, download=download)
            for domain_id, domain_name in DIGITS_DOMAIN_NAMES.items()
        }

        self.intervals: list[int] = []
        for domain_id in self.selected_domains:
            if domain_id not in self.datasets_by_domain:
                raise ValueError(f"Digits domain ids must be in [0, 5], got {domain_id}.")
            self.intervals.append(len(self.datasets_by_domain[domain_id]))
        self.cum_intervals = np.cumsum(self.intervals)

    def __getitem__(self, index):
        dataset_idx = int(np.searchsorted(self.cum_intervals, index, side="right"))
        previous_total = 0 if dataset_idx == 0 else int(self.cum_intervals[dataset_idx - 1])
        real_index = int(index - previous_total)
        domain_id = int(self.selected_domains[dataset_idx])
        image, target = self.datasets_by_domain[domain_id][real_index]

        if self.transform is not None:
            image = self.transform(image)
        return image, int(target), domain_id

    def __len__(self):
        return int(self.cum_intervals[-1]) if len(self.cum_intervals) > 0 else 0


class MixedDatasetWrapper(Dataset):
    def __init__(self, inner, default_domain_id=0):
        self.inner = inner
        self.default_domain_id = default_domain_id

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        if isinstance(item, (list, tuple)):
            if len(item) == 2:
                image, target = item
                return image, target, self.default_domain_id, idx
            if len(item) == 3:
                image, target, domain_id = item
                return image, target, domain_id, idx
            if len(item) == 4:
                return item
        return item


class IndexedDigitsDataset(Dataset):
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        if isinstance(item, (list, tuple)) and len(item) == 3:
            image, target, domain_id = item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            image, target = item
            domain_id = -1
        else:
            raise RuntimeError(f"Unexpected item format from SplitDigits: {item!r}")
        return image, int(target), int(domain_id), int(idx)


class DomainNetCombinedDataset(Dataset):
    def __init__(self, items, transform_fn):
        self.items = items
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, domain_id = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform_fn is not None:
            image = self.transform_fn(image)
        return image, int(label), int(domain_id), int(idx)


class AuxDatasetWrapper(Dataset):
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        image, _ = self.inner[idx]
        return image, -1, -1, idx


def _stack_or_tensor(values):
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    return torch.tensor(values)


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = _stack_or_tensor([item[1] for item in batch])
    domain_ids = _stack_or_tensor([item[2] for item in batch])
    indices = [item[3] for item in batch]
    return images, targets, domain_ids, indices


def get_num_classes(args) -> int:
    dataset = normalize_dataset_name(args["dataset"])
    if dataset in {"domainnet", "mixed_domainnet"}:
        return DOMAINNET_NUM_CLASSES
    if dataset == "digits":
        return DIGITS_NUM_CLASSES
    return DEFAULT_NUM_CLASSES


def _balance_domainnet_entries(entries, reference_domain_id):
    entries_by_domain = {}
    for entry in entries:
        domain_id = int(entry[2])
        entries_by_domain.setdefault(domain_id, []).append(entry)

    if len(entries_by_domain) < 2:
        return entries

    reference_entries = entries_by_domain.get(int(reference_domain_id))
    if not reference_entries:
        return entries

    target_size = len(reference_entries)
    rng = np.random.RandomState(0)

    balanced_entries = list(reference_entries)
    for domain_id in sorted(entries_by_domain):
        if domain_id == int(reference_domain_id):
            continue
        domain_entries = entries_by_domain[domain_id]
        replace = len(domain_entries) < target_size
        selected_indices = rng.choice(len(domain_entries), size=target_size, replace=replace)
        balanced_entries.extend(domain_entries[int(index)] for index in selected_indices.tolist())

    rng.shuffle(balanced_entries)
    return balanced_entries


def _resolve_pin_memory(args) -> bool:
    device = args.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return device == "cuda"


def _build_concat_dataset_and_loader(primary_dataset, aux_dataset, batch_size: int, train: bool, pin_memory: bool):
    combined_dataset = ConcatDataset([primary_dataset, MixedDatasetWrapper(aux_dataset)])
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    return combined_dataset, dataloader


def create_dataset_and_loader(args, selected_domains, train=True):
    batch_size = args["batch_size"]
    pin_memory = _resolve_pin_memory(args)
    dataset_name = normalize_dataset_name(args["dataset"])

    if dataset_name == "mixed_domainnet":
        primary_dataset, _ = create_domainnet_dataset_and_loader(
            selected_domains=[0],
            batch_size=batch_size,
            image_size=DEFAULT_IMAGE_SIZE,
            train=train,
            pin_memory=pin_memory,
        )
        aux_dataset = get_aux_dataset(
            args.get("aux_dataset", "mnist"),
            data_root=str(DATA_ROOT),
            train=train,
            image_size=DEFAULT_IMAGE_SIZE,
        )
        if aux_dataset is None:
            return create_domainnet_dataset_and_loader(
                selected_domains=[0],
                batch_size=batch_size,
                image_size=DEFAULT_IMAGE_SIZE,
                train=train,
                pin_memory=pin_memory,
            )
        return _build_concat_dataset_and_loader(primary_dataset, aux_dataset, batch_size, train, pin_memory)

    if dataset_name == "domainnet":
        return create_domainnet_dataset_and_loader(
            selected_domains=selected_domains,
            batch_size=batch_size,
            image_size=DEFAULT_IMAGE_SIZE,
            train=train,
            pin_memory=pin_memory,
        )

    if dataset_name == "mixed_cifar":
        primary_dataset, _ = create_cifar20_dataset_and_loader(
            selected_domains=[0],
            batch_size=batch_size,
            image_size=DEFAULT_IMAGE_SIZE,
            train=train,
            pin_memory=pin_memory,
        )
        aux_dataset = get_aux_dataset(
            args.get("aux_dataset", "mnist"),
            data_root=str(DATA_ROOT),
            train=train,
            image_size=DEFAULT_IMAGE_SIZE,
        )
        if aux_dataset is None:
            return create_cifar20_dataset_and_loader(
                selected_domains=selected_domains,
                batch_size=batch_size,
                image_size=DEFAULT_IMAGE_SIZE,
                train=train,
                pin_memory=pin_memory,
            )
        return _build_concat_dataset_and_loader(primary_dataset, aux_dataset, batch_size, train, pin_memory)

    if dataset_name == "cifar20":
        return create_cifar20_dataset_and_loader(
            selected_domains=selected_domains,
            batch_size=batch_size,
            image_size=DEFAULT_IMAGE_SIZE,
            train=train,
            pin_memory=pin_memory,
        )

    if dataset_name == "digits":
        return create_digits_dataset_and_loader(
            selected_domains=selected_domains,
            batch_size=batch_size,
            image_size=DEFAULT_IMAGE_SIZE,
            train=train,
            pin_memory=pin_memory,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _build_cifar20_transform(image_size: int, train: bool):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(_identity),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def create_cifar20_dataset_and_loader(
    selected_domains,
    batch_size=64,
    image_size=224,
    train=True,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=True,
):
    dataset = SplitCIFAR20(
        root=str(DATA_ROOT),
        train=train,
        transform=_build_cifar20_transform(image_size=image_size, train=train),
        download=True,
        selected_domains=selected_domains,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, dataloader


def _resolve_domainnet_pairs(selected_domains, available_domains: list[str], domainnet_root: str):
    if selected_domains is None:
        selected_domains = [0]
    if isinstance(selected_domains, int):
        selected_domains = [selected_domains]

    selected_pairs = []
    if isinstance(selected_domains, (list, tuple)):
        for selected_domain in selected_domains:
            if isinstance(selected_domain, int):
                if not available_domains:
                    raise RuntimeError(f"No DomainNet folders found under {domainnet_root}")
                selected_pairs.append((int(selected_domain), available_domains[int(selected_domain)]))
            else:
                domain_name = str(selected_domain)
                selected_pairs.append((available_domains.index(domain_name), domain_name))
    else:
        domain_name = str(selected_domains)
        selected_pairs.append((available_domains.index(domain_name), domain_name))
    return selected_pairs


def _read_domainnet_entries(data_root: str, domainnet_root: str, selected_pairs, train: bool):
    split_name = "train" if train else "test"
    entries = []

    for domain_id, domain_name in selected_pairs:
        split_txt = os.path.join(domainnet_root, f"{domain_name}_{split_name}.txt")
        if os.path.isfile(split_txt):
            with open(split_txt, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    image_path = os.path.join(data_root, parts[0])
                    if not os.path.isfile(image_path):
                        image_path = os.path.join(domainnet_root, parts[0])
                    entries.append((image_path, int(parts[1]), int(domain_id)))
            continue

        domain_folder = os.path.join(domainnet_root, domain_name)
        if not os.path.isdir(domain_folder):
            raise RuntimeError(f"Domain '{domain_name}' missing split file and folder.")

        image_folder = datasets.ImageFolder(root=domain_folder, transform=None)
        for image_path, label in image_folder.samples:
            entries.append((image_path, int(label), int(domain_id)))

    return entries


def create_domainnet_dataset_and_loader(
    selected_domains,
    batch_size=64,
    image_size=224,
    train=True,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=True,
):
    data_root = str(DATA_ROOT)
    domainnet_root = os.path.join(data_root, "domainnet")
    available_domains = get_domainnet_domain_names(domainnet_root)
    selected_pairs = _resolve_domainnet_pairs(selected_domains, available_domains, domainnet_root)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(_identity),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    entries = _read_domainnet_entries(data_root, domainnet_root, selected_pairs, train=train)
    if train and len(selected_pairs) > 1:
        reference_domain_id = selected_pairs[0][0]
        entries = _balance_domainnet_entries(entries, reference_domain_id=reference_domain_id)

    dataset = DomainNetCombinedDataset(entries, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, dataloader


def create_digits_dataset_and_loader(
    selected_domains,
    batch_size=64,
    image_size=224,
    train=True,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=True,
):
    transform = get_digits_transforms(image_size=image_size, is_training=train)
    dataset = SplitDigits(
        root=str(DATA_ROOT),
        train=train,
        transform=transform,
        download=True,
        selected_domains=selected_domains,
    )
    wrapped_dataset = IndexedDigitsDataset(dataset)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return wrapped_dataset, dataloader


def _build_mnist_aux_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(_repeat_grayscale_channels),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_cub_aux_transform():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_aux_dataset(name, data_root, train=True, image_size=224, sample_frac=None, sample_seed=42):
    name = (name or "").lower()
    if not name:
        return None

    if name == "mnist":
        dataset = datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
            transform=_build_mnist_aux_transform(image_size),
        )
    elif name == "cub":
        cub_root = os.path.join(data_root, "cub")
        if not os.path.isdir(cub_root):
            raise RuntimeError(f"CUB dataset folder not found at {cub_root}")
        dataset = datasets.ImageFolder(root=cub_root, transform=_build_cub_aux_transform())
    else:
        raise ValueError(f"Unsupported aux dataset: {name}")

    if sample_frac is not None and float(sample_frac) < 1.0:
        total = len(dataset)
        subset_size = max(1, int(total * float(sample_frac)))
        rng = np.random.RandomState(int(sample_seed))
        subset_indices = rng.choice(total, size=subset_size, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    return AuxDatasetWrapper(dataset)


def get_digits_transforms(image_size=224, is_training=True):
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(_repeat_grayscale_channels),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(_repeat_grayscale_channels),
        ]
    )
