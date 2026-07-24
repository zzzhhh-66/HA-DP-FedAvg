"""Shared loss construction for fair comparisons across training methods."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_positive_class_weight(y, mode: str = "balanced") -> float | None:
    """Compute one global positive-class weight without using test labels."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class weighting mode: {mode}")
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        return 1.0
    return negatives / positives


def binary_criterion(pos_weight: float | None, device: torch.device) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    weight = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)
