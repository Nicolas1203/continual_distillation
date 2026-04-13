"""
Base model classes for student and teacher models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseStudent(nn.Module, ABC):
    """Base class for student models in incremental distillation."""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.activations = {}
        self.hooks = []
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the student model."""
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        pass
    
    def register_activation_hooks(self, layer_names: List[str]) -> None:
        """Register hooks to capture activations from specified layers."""
        self.clear_hooks()
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def clear_hooks(self) -> None:
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class BaseTeacher(nn.Module, ABC):
    """Base class for teacher models in incremental distillation."""
    
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.activations = {}
        self.hooks = []
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the teacher model."""
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        pass
    
    def register_activation_hooks(self, layer_names: List[str]) -> None:
        """Register hooks to capture activations from specified layers."""
        self.clear_hooks()
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def clear_hooks(self) -> None:
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()
    
    def freeze_parameters(self) -> None:
        """Freeze all parameters of the teacher model."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters of the teacher model."""
        for param in self.parameters():
            param.requires_grad = True
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def get_parameters_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
