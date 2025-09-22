#!/usr/bin/env python3
"""
Test script to validate new model implementations
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.models.kws_models import (
        create_kws_teacher, 
        create_kws_student, 
        create_tiny_kws_student,
        PhoWhisperKWSTeacher,
        ResNet18KWSStudent,
        MobileNetV3KWSStudent
    )
    
    print("✅ Successfully imported all new model classes")
    
    # Test model creation
    print("\n🏗️  Testing model creation...")
    
    # Teacher model (PhoWhisper-based)
    try:
        teacher = create_kws_teacher(num_classes=10, hidden_size=512, dropout=0.1)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"✅ PhoWhisper Teacher: {teacher_params:,} parameters")
    except Exception as e:
        print(f"❌ PhoWhisper Teacher failed: {e}")
    
    # Standard student (MobileNetV3)
    try:
        student = create_kws_student(num_classes=10, dropout=0.2)
        student_params = sum(p.numel() for p in student.parameters())
        print(f"✅ MobileNetV3 Student: {student_params:,} parameters")
    except Exception as e:
        print(f"❌ MobileNetV3 Student failed: {e}")
    
    # Tiny student (ResNet18)
    try:
        tiny_student = create_tiny_kws_student(num_classes=10, dropout=0.3)
        tiny_params = sum(p.numel() for p in tiny_student.parameters())
        print(f"✅ ResNet18 Tiny Student: {tiny_params:,} parameters")
    except Exception as e:
        print(f"❌ ResNet18 Tiny Student failed: {e}")
        
    print("\n🧪 Testing forward pass with dummy data...")
    
    import torch
    
    # Create dummy input (batch_size=2, mel_features=80, time_steps=100)
    dummy_input = torch.randn(2, 80, 100)
    dummy_labels = torch.randint(0, 10, (2,))
    
    # Test each model
    models_to_test = [
        ("PhoWhisper Teacher", teacher if 'teacher' in locals() else None),
        ("MobileNetV3 Student", student if 'student' in locals() else None),
        ("ResNet18 Tiny", tiny_student if 'tiny_student' in locals() else None)
    ]
    
    for name, model in models_to_test:
        if model is not None:
            try:
                with torch.no_grad():
                    outputs = model(dummy_input, dummy_labels)
                    logits_shape = outputs['logits'].shape
                    has_loss = 'loss' in outputs
                    print(f"✅ {name}: logits shape {logits_shape}, loss={'✓' if has_loss else '✗'}")
            except Exception as e:
                print(f"❌ {name} forward pass failed: {e}")
    
    print("\n🎉 Model validation complete!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")