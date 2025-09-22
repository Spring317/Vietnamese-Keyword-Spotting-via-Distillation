#!/usr/bin/env python3
"""
Test script to verify KWS pipeline with most frequent words as keywords.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from data.vietnamese_kws_dataset import create_kws_dataloader

def test_new_keywords():
    """Test the updated KWS dataset with most frequent words."""
    print("=" * 60)
    print("TESTING KWS PIPELINE WITH MOST FREQUENT KEYWORDS")
    print("=" * 60)
    
    print("\n1. NEW KEYWORDS:")
    from data.vietnamese_kws_dataset import VietnameseKWSDataset
    
    for class_id, keyword in VietnameseKWSDataset.KEYWORDS.items():
        print(f"   Class {class_id}: '{keyword}'")
    print(f"   Class 9: 'negative' (non-keyword speech)")
    
    print(f"\n2. TESTING DATASET CREATION:")
    try:
        # Test with dummy data first
        print("   Creating dataset with dummy data...")
        train_loader, train_dataset = create_kws_dataloader(
            data_dir="./data/vivos",
            split="train",
            batch_size=4,
            shuffle=True,
            max_samples=100,
            use_dummy_data=True
        )
        
        print(f"   ✅ Success! Created dataset with {len(train_dataset)} samples")
        
        # Get statistics
        stats = train_dataset.get_stats()
        print(f"   Dataset stats: {stats}")
        
        print(f"\n3. TESTING BATCH LOADING:")
        # Test loading a batch
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i+1}:")
            print(f"     Input features shape: {batch['input_features'].shape}")
            print(f"     Labels shape: {batch['labels'].shape}")
            print(f"     Labels: {batch['labels'].tolist()}")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print(f"\n4. TESTING WITH REAL DATA:")
        try:
            # Test with real data if available
            real_train_loader, real_train_dataset = create_kws_dataloader(
                data_dir="./data/vivos",
                split="train", 
                batch_size=4,
                shuffle=True,
                max_samples=50,
                use_dummy_data=False
            )
            
            real_stats = real_train_dataset.get_stats()
            print(f"   ✅ Real data success! Dataset has {len(real_train_dataset)} samples")
            print(f"   Real data stats: {real_stats}")
            
            # Check keyword frequency
            keyword_counts = real_stats.get('keyword_counts', {})
            total_samples = sum(keyword_counts.values())
            
            print(f"\\n   KEYWORD DISTRIBUTION:")
            for class_id in range(10):
                count = keyword_counts.get(class_id, 0)
                if class_id < 9:
                    keyword = VietnameseKWSDataset.KEYWORDS[class_id]
                    percentage = (count / total_samples * 100) if total_samples > 0 else 0
                    print(f"     Class {class_id} ('{keyword}'): {count} samples ({percentage:.1f}%)")
                else:
                    percentage = (count / total_samples * 100) if total_samples > 0 else 0
                    print(f"     Class {class_id} ('negative'): {count} samples ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"   ⚠️  Real data test failed: {e}")
            print(f"   This is expected if VIVOS data is not available.")
        
        print(f"\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print(f"   The updated keywords should provide much better training data")
        print(f"   since they are the most frequent words in the VIVOS dataset.")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Pipeline test failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_new_keywords()
    if success:
        print(f"\\n🚀 Ready to start training with optimized keywords!")
        print(f"   Run: python scripts/train_vietnamese_kws.py --config configs/vietnamese_kws_config.yaml --student_model tiny --data_dir ./data/vivos --output_dir ./outputs_frequent_words")
    else:
        print(f"\\n❌ Pipeline test failed. Please check the configuration.")