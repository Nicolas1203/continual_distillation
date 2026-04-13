from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.paths import CONFIG_ROOT


class BaseMethod:
    """Base class for all training methods."""

    method_name: str = ""

    def __init__(self, args: dict[str, Any], optimizer, config_path: str | Path | None = None) -> None:
        self.optimizer = optimizer
        self.args = dict(args)
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config(self.config_path)
        self.args.update(self.config)

    def _resolve_config_path(self, config_path: str | Path | None) -> Path | None:
        if config_path:
            return Path(config_path)
        if not self.method_name:
            return None
        return CONFIG_ROOT / "methods" / f"{self.method_name}.json"

    @staticmethod
    def _load_config(config_path: Path | None) -> dict[str, Any]:
        if config_path is None or not config_path.exists():
            return {}

        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def train(
        self,
        *,
        student,
        domains_teacher: str,
        domains_data: list[int],
        device: str,
        epochs: int,
        dataloader,
        num_classes: int,
        **kwargs: Any,
    ):
        raise NotImplementedError("Subclasses must implement train().")
