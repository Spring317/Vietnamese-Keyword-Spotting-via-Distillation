"""
Training package for Vietnamese ASR distillation.
"""

from .distillation import (
    DistillationLoss,
    KnowledgeDistillationTrainer,
    create_distillation_trainer,
)

__all__ = [
    'DistillationLoss',
    'KnowledgeDistillationTrainer',
    'create_distillation_trainer',
]