"""
Vision model implementations for incremental distillation.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional, List
from .base_models import BaseStudent, BaseTeacher


class VisionStudent(BaseStudent):
    """Vision student model for incremental distillation."""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__(model_name, num_classes, pretrained)
        
        # Load the base model from timm with 1000 classes to match teachers
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes  # Always use 1000 classes to match teachers
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.feature_dim = self.backbone.num_features
        elif hasattr(self.backbone, 'head'):
            self.feature_dim = self.backbone.head.in_features
        else:
            # Default for most vision models
            self.feature_dim = 2048
        
        # Keep the original pre-trained head intact (1000 classes)
        if hasattr(self.backbone, 'head'):
            self.original_head = self.backbone.head  # Pre-trained head with 1000 classes
        elif hasattr(self.backbone, 'classifier'):
            self.original_head = self.backbone.classifier  # Pre-trained classifier with 1000 classes
        else:
            self.original_head = None
        
        # No need for custom distillation head - use the pre-trained head directly
        self.distillation_head = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the student model."""
        # Use the full model with pre-trained head (1000 classes)
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        return self.backbone(x)
    
    def get_original_head_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get output from the original head if available."""
        if self.original_head is not None:
            features = self.backbone(x)
            return self.original_head(features)
        return self.forward(x)
    
    def replace_head(self, new_head: nn.Module) -> None:
        """Replace the distillation head with a new one."""
        self.distillation_head = new_head
    
    def get_feature_dimension(self) -> int:
        """Get the feature dimension of the model."""
        return self.feature_dim


class VisionTeacher(BaseTeacher):
    """Vision teacher model for incremental distillation."""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__(model_name, num_classes, pretrained)
        
        # Load the teacher model from timm with 1000 classes to preserve pre-trained head
        if num_classes != 1000:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes  # Always use 1000 classes to preserve pre-trained head
            )
        else:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained
            )
        
        # Get feature dimension
        if hasattr(self.model, 'num_features'):
            self.feature_dim = self.model.num_features
        elif hasattr(self.model, 'head'):
            self.feature_dim = self.model.head.in_features
        else:
            self.feature_dim = 2048
        
        # Keep the original pre-trained head intact (1000 classes)
        if hasattr(self.model, 'head'):
            self.original_head = self.model.head  # Pre-trained head with 1000 classes
        elif hasattr(self.model, 'classifier'):
            self.original_head = self.model.classifier  # Pre-trained classifier with 1000 classes
        else:
            self.original_head = None
        
        # Freeze teacher parameters by default
        self.freeze_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the teacher model."""
        # Use the full model with pre-trained head (1000 classes)
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        return self.model(x)
    
    def get_feature_dimension(self) -> int:
        """Get the feature dimension of the model."""
        return self.feature_dim


class VisionModelFactory:
    """Factory for creating vision models."""
    
    @staticmethod
    def create_student(model_name: str, num_classes: int, pretrained: bool = True) -> VisionStudent:
        """Create a vision student model."""
        return VisionStudent(model_name, num_classes, pretrained)
    
    @staticmethod
    def create_teacher(model_name: str, num_classes: int, pretrained: bool = True) -> VisionTeacher:
        """Create a vision teacher model."""
        return VisionTeacher(model_name, num_classes, pretrained)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available vision models."""
        return timm.list_models(pretrained=True)
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            model = timm.create_model(model_name, pretrained=False)
            info = {
                'name': model_name,
                'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            # Get feature dimension
            if hasattr(model, 'num_features'):
                info['feature_dim'] = model.num_features
            elif hasattr(model, 'head'):
                info['feature_dim'] = model.head.in_features
            else:
                info['feature_dim'] = 2048  # Default
            
            return info
        except Exception as e:
            return {'name': model_name, 'error': str(e)}
