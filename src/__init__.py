"""
Vietnamese ASR Knowledge Distillation Pipeline

A comprehensive framework for training efficient Vietnamese ASR models
using knowledge distillation from PhoWhisper-base teacher model.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Vietnamese ASR Knowledge Distillation Pipeline"

from . import models
from . import data
from . import training
from . import evaluation
from . import utils

__all__ = [
    'models',
    'data', 
    'training',
    'evaluation',
    'utils',
]