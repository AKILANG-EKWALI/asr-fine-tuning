"""
src/model/__init__.py
Expose les fonctions publiques du sous-package model.
"""
from .build_model import build_model_with_lora
from .lora_utils import find_all_linear_names

__all__ = ["build_model_with_lora", "find_all_linear_names"]
