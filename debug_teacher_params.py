#!/usr/bin/env python3
"""
Debug script to analyze PhoWhisper teacher model parameter count.
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from models.kws_models import create_kws_teacher

def analyze_teacher_parameters():
    """Analyze the teacher model parameter breakdown."""
    print("=" * 60)
    print("PHOWHISPER TEACHER PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Create teacher model
    print("Creating PhoWhisper teacher model...")
    teacher_model = create_kws_teacher(
        num_classes=10,
        hidden_size=512,
        dropout=0.1,
        freeze_encoder=True
    )
    
    print("\n1. TOTAL PARAMETER COUNT:")
    total_params = sum(p.numel() for p in teacher_model.parameters())
    trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")
    print(f"   Frozen percentage: {(frozen_params/total_params)*100:.1f}%")
    
    print("\n2. COMPONENT BREAKDOWN:")
    
    # Whisper encoder
    encoder_total = sum(p.numel() for p in teacher_model.whisper_encoder.parameters())
    encoder_trainable = sum(p.numel() for p in teacher_model.whisper_encoder.parameters() if p.requires_grad)
    print(f"   Whisper Encoder:")
    print(f"     Total: {encoder_total:,}")
    print(f"     Trainable: {encoder_trainable:,}")
    print(f"     Frozen: {encoder_total - encoder_trainable:,}")
    
    # Adaptation layers
    adaptation_total = sum(p.numel() for p in teacher_model.adaptation.parameters())
    adaptation_trainable = sum(p.numel() for p in teacher_model.adaptation.parameters() if p.requires_grad)
    print(f"   Adaptation Layers:")
    print(f"     Total: {adaptation_total:,}")
    print(f"     Trainable: {adaptation_trainable:,}")
    
    # Classifier
    classifier_total = sum(p.numel() for p in teacher_model.classifier.parameters())
    classifier_trainable = sum(p.numel() for p in teacher_model.classifier.parameters() if p.requires_grad)
    print(f"   Classifier:")
    print(f"     Total: {classifier_total:,}")
    print(f"     Trainable: {classifier_trainable:,}")
    
    # Feature projector
    projector_total = sum(p.numel() for p in teacher_model.feature_projector.parameters())
    projector_trainable = sum(p.numel() for p in teacher_model.feature_projector.parameters() if p.requires_grad)
    print(f"   Feature Projector:")
    print(f"     Total: {projector_total:,}")
    print(f"     Trainable: {projector_trainable:,}")
    
    print("\n3. WHISPER BASE EXPECTED PARAMETERS:")
    print("   Standard Whisper-base should have ~74M parameters")
    print("   But our encoder shows:", f"{encoder_total:,}")
    
    if encoder_total < 50_000_000:  # Less than 50M
        print("\n⚠️  WARNING: Encoder parameter count seems low!")
        print("   Possible issues:")
        print("   1. Model loading failed, using smaller fallback")
        print("   2. Only encoder loaded (not full model)")
        print("   3. Model architecture is different than expected")
    
    print("\n4. WHY TOTAL IS 21M:")
    print("   The script likely counts all parameters INCLUDING frozen ones.")
    print("   Whisper-base encoder alone should be ~74M parameters.")
    print("   If showing 21M, there might be an issue with model loading.")
    
    print("\n5. ARCHITECTURE DETAILS:")
    print(f"   Model config: {teacher_model.whisper_encoder.config}")
    print(f"   Encoder layers: {teacher_model.whisper_encoder.config.encoder_layers}")
    print(f"   Hidden size: {teacher_model.whisper_encoder.config.d_model}")
    print(f"   Attention heads: {teacher_model.whisper_encoder.config.encoder_attention_heads}")

if __name__ == "__main__":
    analyze_teacher_parameters()