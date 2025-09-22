"""
Model package for Vietnamese ASR knowledge distillation.
"""

from .teacher import PhoWhisperTeacher
from .resnet18_student import ResNet18ASR, create_resnet18_asr
from .mobilenetv3_student import MobileNetV3ASR, create_mobilenetv3_asr
from .quartznet_student import QuartzNetASR, QuartzNetSmallASR, create_quartznet_asr

__all__ = [
    'PhoWhisperTeacher',
    'ResNet18ASR',
    'create_resnet18_asr',
    'MobileNetV3ASR',
    'create_mobilenetv3_asr',
    'QuartzNetASR',
    'QuartzNetSmallASR',
    'create_quartznet_asr',
]