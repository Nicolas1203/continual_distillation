from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.data_utils import normalize_dataset_name
from src.utils.paths import CHECKPOINT_ROOT


BEST_MODEL_FILENAME = "best_model.pth"
FOUNDATION_MODEL_PATTERN = "best_foundation_model*.pth"
PREFERRED_FOUNDATION_MODEL = "best_foundation_model_vit_huge_patch14_clip_224.laion2b.pth"

DIGITS_TEACHER_ALIASES = {
    "0_1": "mnist_svhn",
    "0_2": "mnist_mnist-m",
    "0_3": "mnist_usps",
    "0_4": "mnist_kmnist",
    "0_5": "mnist_fashion-mnist",
}


def _strip_teacher_prefix(teacher_id: str) -> str:
    teacher_id = str(teacher_id)
    if teacher_id.startswith("teacher:"):
        return teacher_id.split("teacher:", 1)[1]
    return teacher_id


def _extract_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _resolve_cifar_teacher_path(teacher_id: str) -> Path:
    normalized = _strip_teacher_prefix(teacher_id)
    directory_name = normalized if normalized.startswith("domain_") else f"domain_{normalized}"
    return CHECKPOINT_ROOT / "cifar20" / directory_name / BEST_MODEL_FILENAME


def _resolve_domainnet_teacher_path(teacher_id: str, use_foundation: bool) -> Path:
    normalized = _strip_teacher_prefix(teacher_id)
    pair_name = normalized if normalized.startswith("pair_") else f"pair_{normalized}"
    root = CHECKPOINT_ROOT / ("domainnet_foundation" if use_foundation else "domainnet") / pair_name

    if not root.exists():
        return root / BEST_MODEL_FILENAME

    if not use_foundation:
        return root / BEST_MODEL_FILENAME

    preferred_path = root / PREFERRED_FOUNDATION_MODEL
    if preferred_path.exists():
        return preferred_path

    fallback_candidates = sorted(root.glob(FOUNDATION_MODEL_PATTERN))
    if fallback_candidates:
        return fallback_candidates[0]

    return root / BEST_MODEL_FILENAME


def _resolve_digits_teacher_path(teacher_id: str) -> Path:
    normalized = _strip_teacher_prefix(teacher_id)
    alias = DIGITS_TEACHER_ALIASES.get(normalized, normalized)
    return CHECKPOINT_ROOT / "digits" / alias / BEST_MODEL_FILENAME


def resolve_teacher_checkpoint_path(dataset: str, teacher_id: str, use_foundation: bool = False) -> Path:
    """Resolve the checkpoint path for the requested dataset and teacher id."""
    dataset = normalize_dataset_name(dataset)
    if dataset in {"cifar20", "mixed_cifar"}:
        return _resolve_cifar_teacher_path(teacher_id)
    if dataset in {"domainnet", "mixed_domainnet"}:
        return _resolve_domainnet_teacher_path(teacher_id, use_foundation=use_foundation)
    if dataset == "digits":
        return _resolve_digits_teacher_path(teacher_id)
    raise ValueError(f"Unsupported dataset for teacher checkpoints: {dataset}")


def load_teacher_checkpoint(teacher_id: str, args, use_foundation: bool = False):
    """Load a teacher checkpoint and normalize it to the expected dictionary structure."""
    checkpoint_path = resolve_teacher_checkpoint_path(
        args["dataset"],
        teacher_id,
        use_foundation=use_foundation,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = _extract_state_dict(checkpoint)
        return checkpoint

    return {
        "model_state_dict": _extract_state_dict(checkpoint),
        "checkpoint_path": str(checkpoint_path),
    }
