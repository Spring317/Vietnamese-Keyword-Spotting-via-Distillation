#!/usr/bin/env python3
"""
Demo script to show how to use the Vietnamese ASR distillation pipeline.
"""

import sys
from pathlib import Path
import torch
from transformers import WhisperProcessor

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models import PhoWhisperTeacher, create_resnet18_asr
from data import AudioPreprocessor


def demo_teacher_model():
    """Demonstrate teacher model usage."""
    print("=== PhoWhisper Teacher Model Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load teacher model
    teacher = PhoWhisperTeacher(device=str(device))
    print("Teacher model loaded successfully!")
    
    print(f"Vocabulary size: {teacher.get_vocab_size()}")
    print(f"Hidden size: {teacher.get_hidden_size()}")


def demo_student_model():
    """Demonstrate student model creation."""
    print("\n=== Student Models Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ResNet18 student
    resnet_student = create_resnet18_asr(
        num_mel_bins=80,
        vocab_size=51865,
        hidden_size=512,
    )
    
    print(f"ResNet18 parameters: {resnet_student.get_num_parameters():,}")
    print(f"ResNet18 size: {resnet_student.get_model_size_mb():.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(2, 80, 100)  # batch_size=2, n_mels=80, time=100
    with torch.no_grad():
        output = resnet_student(dummy_input)
    
    print(f"Output shape: {output['logits'].shape}")


def demo_audio_preprocessing():
    """Demonstrate audio preprocessing."""
    print("\n=== Audio Preprocessing Demo ===")
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=80,
        normalize=True,
    )
    
    print("Audio preprocessor created")
    print(f"Sample rate: {preprocessor.sample_rate}")
    print(f"Number of mel bins: {preprocessor.n_mels}")
    
    # Create dummy audio
    dummy_audio = torch.randn(1, 16000)  # 1 second of audio
    features = preprocessor.extract_mel_features(dummy_audio)
    
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Mel features shape: {features.shape}")


def main():
    """Run all demos."""
    print("Vietnamese ASR Knowledge Distillation Pipeline Demo")
    print("=" * 60)
    
    try:
        demo_teacher_model()
        demo_student_model() 
        demo_audio_preprocessing()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("You can now:")
        print("1. Prepare your Vietnamese speech data (see data/DATA_PREPARATION.md)")
        print("2. Configure training settings (see configs/)")
        print("3. Start training: python scripts/train.py --config configs/resnet18_config.yaml --student_model resnet18")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure you have installed all requirements:")
        print("pip install -r requirements.txt")


if __name__ == '__main__':
    main()