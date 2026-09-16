"""Cross-view self-distillation used by trajectory-only HAR-CGCD.

This module intentionally depends only on PyTorch.  It keeps the small piece
of clustering mathematics needed by the motion-primitive pipeline independent
from the historical SimGCD/Happy image utility module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewDistillationLoss(nn.Module):
    """Cross entropy between sharpened opposite-view probability targets."""

    def __init__(
        self,
        warmup_teacher_temp_epochs: int,
        epochs: int,
        n_views: int = 2,
        warmup_teacher_temperature: float = 0.07,
        teacher_temperature: float = 0.04,
        student_temperature: float = 0.10,
    ) -> None:
        super().__init__()
        warmup = int(warmup_teacher_temp_epochs)
        epoch_count = int(epochs)
        views = int(n_views)
        if warmup < 0 or epoch_count < 1 or warmup > epoch_count:
            raise ValueError("Teacher-temperature warmup must lie within the epoch range.")
        if views < 2:
            raise ValueError("Cross-view distillation requires at least two views.")
        temperatures = (
            float(warmup_teacher_temperature),
            float(teacher_temperature),
            float(student_temperature),
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
            raise ValueError("Distillation temperatures must be positive and finite.")

        if warmup:
            warmup_values = torch.linspace(
                temperatures[0], temperatures[1], steps=warmup, dtype=torch.float64
            ).tolist()
        else:
            warmup_values = []
        self.teacher_temperature_schedule = tuple(
            float(value)
            for value in warmup_values
            + [temperatures[1]] * (epoch_count - warmup)
        )
        self.student_temperature = temperatures[2]
        self.n_views = views

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        epoch_index: int,
    ) -> torch.Tensor:
        if student_output.ndim != 2 or teacher_output.ndim != 2:
            raise ValueError("Distillation logits must be rank-2 tensors.")
        if student_output.shape != teacher_output.shape:
            raise ValueError("Student and teacher logits must have identical shapes.")
        if len(student_output) % self.n_views:
            raise ValueError("The logit batch must divide evenly across views.")
        epoch = int(epoch_index)
        if epoch < 0 or epoch >= len(self.teacher_temperature_schedule):
            raise IndexError(f"epoch_index {epoch} is outside the registered schedule.")

        students = (student_output / self.student_temperature).chunk(self.n_views)
        teacher_temperature = self.teacher_temperature_schedule[epoch]
        teachers = F.softmax(teacher_output / teacher_temperature, dim=-1).detach().chunk(
            self.n_views
        )
        terms = [
            torch.sum(-teacher * F.log_softmax(student, dim=-1), dim=-1).mean()
            for teacher_index, teacher in enumerate(teachers)
            for student_index, student in enumerate(students)
            if student_index != teacher_index
        ]
        if not terms:
            raise RuntimeError("Cross-view distillation produced no opposite-view pairs.")
        return torch.stack(terms).mean()


__all__ = ["CrossViewDistillationLoss"]
