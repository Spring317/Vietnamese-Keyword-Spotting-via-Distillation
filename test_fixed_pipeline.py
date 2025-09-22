#!/usr/bin/env python3
"""
Test script to validate the fixed knowledge distillation pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data.vivos_dataset import create_vivos_dataloader
from src.models import PhoWhisperTeacher, create_resnet18_asr
from src.training.distillation import create_distillation_trainer

def test_batch_processing():
    """Test that batches are processed without size mismatches"""
    print("Testing batch processing...")
    
    # Create a small dataloader
    try:
        train_loader = create_vivos_dataloader(
            data_dir="/home/quydx/distile_asr_phoWhisper/data/vivos",
            split="train", 
            batch_size=2,
            shuffle=False,
            max_samples=10
        )
        print(f"✓ DataLoader created successfully")
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded: input_features shape = {batch['input_features'].shape}")
        print(f"✓ Batch texts: {len(batch['texts'])} samples")
        
        # Test models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        # Load teacher
        teacher = PhoWhisperTeacher()
        teacher.to(device)
        print(f"✓ Teacher model loaded")
        
        # Load student 
        student = create_resnet18_asr(
            num_mel_bins=80,
            vocab_size=51865,
            hidden_size=512
        )
        student.to(device)
        print(f"✓ Student model loaded")
        
        # Create trainer
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        trainer = create_distillation_trainer(
            teacher_model=teacher,
            student_model=student,
            optimizer=optimizer,
            device=device
        )
        print(f"✓ Trainer created")
        
        # Test forward pass
        input_features = batch['input_features'].to(device)
        print(f"✓ Input features moved to device: {input_features.shape}")
        
        # Teacher forward
        with torch.no_grad():
            teacher_outputs = teacher(input_features, return_features=True)
            print(f"✓ Teacher forward pass completed")
            print(f"  Teacher logits shape: {teacher_outputs.get('logits', 'Not found')}")
        
        # Student forward
        student_outputs = student(input_features, return_features=True)
        print(f"✓ Student forward pass completed")
        print(f"  Student logits shape: {student_outputs.get('logits', 'Not found')}")
        
        # Test training step
        loss_dict = trainer.train_step(batch, return_features=True)
        print(f"✓ Training step completed")
        print(f"  Loss components: {list(loss_dict.keys())}")
        print(f"  Total loss: {loss_dict.get('total_loss', 'Not found')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vivos_loading():
    """Test VIVOS dataset loading"""
    print("\nTesting VIVOS dataset loading...")
    
    try:
        # Test train split
        train_loader = create_vivos_dataloader(
            data_dir="/home/quydx/distile_asr_phoWhisper/data/vivos",
            split="train",
            batch_size=4,
            max_samples=20
        )
        
        train_batch = next(iter(train_loader))
        print(f"✓ Train split: {train_batch['input_features'].shape}")
        print(f"  Sample text: {train_batch['texts'][0][:50]}...")
        
        # Test test split  
        test_loader = create_vivos_dataloader(
            data_dir="/home/quydx/distile_asr_phoWhisper/data/vivos",
            split="test",
            batch_size=4,
            max_samples=20
        )
        
        test_batch = next(iter(test_loader))
        print(f"✓ Test split: {test_batch['input_features'].shape}")
        print(f"  Sample text: {test_batch['texts'][0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Vietnamese ASR Knowledge Distillation Pipeline")
    print("=" * 60)
    
    success1 = test_vivos_loading()
    success2 = test_batch_processing()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The pipeline is working correctly.")
        print("✅ Batch size mismatch issue has been resolved.")
        print("✅ VIVOS dataset loading is working properly.")
        print("✅ Knowledge distillation training is functional.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\n💡 You can now run the full training with:")
    print("   python scripts/train.py --config configs/vivos_resnet18.yaml --student_model resnet18 --output_dir ./outputs_vivos")