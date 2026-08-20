"""
src/training/__init__.py
Expose les fonctions publiques du sous-package training.
"""
from .metrics import compute_metrics, normalize_text

__all__ = ["compute_metrics", "normalize_text"]
