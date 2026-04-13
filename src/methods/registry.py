from __future__ import annotations

from src.methods.base import BaseMethod
from src.methods.dkd import DKDMethod
from src.methods.kl_divergence import KLDivergenceMethod
from src.methods.ls import LSMethod
from src.methods.mds import MDSMethod
from src.methods.se2d import SE2DMethod
from src.methods.self_distillation import SelfDistillationMethod


METHOD_ALIASES: dict[str, str] = {
    "baseline": "kl_divergence",
    "kl": "kl_divergence",
    "kl_divergence": "kl_divergence",
    "dkd": "dkd",
    "medium": "mds",
    "mds": "mds",
    "stand": "ls",
    "ls": "ls",
    "checkpoint": "self_distillation",
    "self-distillation": "self_distillation",
    "self_distillation": "self_distillation",
    "checkpoint_ours": "se2d",
    "checkpoint-ours": "se2d",
    "se2d": "se2d",
}


METHOD_CLASSES: dict[str, type[BaseMethod]] = {
    "kl_divergence": KLDivergenceMethod,
    "dkd": DKDMethod,
    "mds": MDSMethod,
    "ls": LSMethod,
    "self_distillation": SelfDistillationMethod,
    "se2d": SE2DMethod,
}


def normalize_method_name(name: str) -> str:
    """Resolve legacy aliases to the canonical method name."""
    normalized = METHOD_ALIASES.get(str(name).lower())
    if normalized is None:
        available = ", ".join(sorted(METHOD_CLASSES))
        raise ValueError(f"Unknown method '{name}'. Available methods: {available}")
    return normalized


def available_methods() -> list[str]:
    return sorted(METHOD_CLASSES)


def create_method(name: str, args, optimizer, config_path=None) -> BaseMethod:
    """Instantiate a method implementation from its canonical or aliased name."""
    normalized = normalize_method_name(name)
    return METHOD_CLASSES[normalized](args=args, optimizer=optimizer, config_path=config_path)
