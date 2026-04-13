"""Factory utilities for instantiating vision student and teacher models."""

from __future__ import annotations

from typing import Any

from .base_models import BaseStudent, BaseTeacher
from .vision_models import VisionModelFactory


class ModelFactory:
    """Create model instances for the maintained vision-only codepath."""

    def __init__(self) -> None:
        self.vision_factory = VisionModelFactory()

    def create_student(self, model_name: str, num_classes: int, pretrained: bool = True) -> BaseStudent:
        return self.vision_factory.create_student(model_name, num_classes, pretrained)

    def create_teacher(self, model_name: str, num_classes: int, pretrained: bool = True) -> BaseTeacher:
        return self.vision_factory.create_teacher(model_name, num_classes, pretrained)

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        return self.vision_factory.get_model_info(model_name)

    def get_available_models(self) -> dict[str, list]:
        return {"vision": self.vision_factory.get_available_models()}
