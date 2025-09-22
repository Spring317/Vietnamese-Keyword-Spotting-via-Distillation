# Vietnamese Keyword Spotting (KWS) System

## 🎯 Problem Pivot: From ASR to KWS

We've successfully pivoted from Vietnamese ASR (which was overfitting) to a **Vietnamese Keyword Spotting** system with 10 common commands. This is much more practical and manageable!

## 📋 System Overview

### Target Keywords (11 Classes Total):
1. **"chào"** (Hello/Hi) - Class 0
2. **"dừng"** (Stop) - Class 1  
3. **"đi"** (Go) - Class 2
4. **"lại đây"** (Come here) - Class 3
5. **"mở"** (Open) - Class 4
6. **"đóng"** (Close) - Class 5
7. **"bật"** (Turn on) - Class 6
8. **"tắt"** (Turn off) - Class 7
9. **"tìm"** (Find/Search) - Class 8
10. **"gọi"** (Call) - Class 9
11. **Negative** (Non-keyword speech) - Class 10

## 🏗️ Architecture

### Data Pipeline:
- **Input**: 2-second audio segments from VIVOS dataset
- **Processing**: Extract segments containing keywords + negative samples
- **Format**: Mel spectrograms (80 features) → 11-class classification
- **Balance**: 2:1 ratio of negative to positive samples

### Models:
1. **Teacher Model**: 
   - CNN-based audio classifier (~50K parameters)
   - 256 hidden dimensions
   - Provides soft targets for distillation

2. **Student Models**:
   - **Standard**: ~15K parameters (3x compression)
   - **Tiny**: ~5K parameters (10x compression)
   - Depthwise separable convolutions for efficiency

### Knowledge Distillation:
- **Temperature**: 4.0 (good for 11-class problem)
- **Loss Weights**: 70% soft + 30% hard + feature matching
- **Much more stable** than ASR distillation

## 📁 File Structure

```
├── configs/
│   ├── vietnamese_kws_config.yaml      # KWS training config
│   └── vietnamese_kws_keywords.md      # Keyword definitions
├── src/
│   ├── data/
│   │   └── vietnamese_kws_dataset.py   # KWS dataset implementation
│   └── models/
│       └── kws_models.py               # Teacher/student KWS models
└── scripts/
    ├── train_vietnamese_kws.py         # Main KWS training script  
    └── test_kws_pipeline.py            # Pipeline verification
```

## 🚀 Quick Start

### 1. Test the Pipeline:
```bash
cd /home/quydx/distile_asr_phoWhisper
python scripts/test_kws_pipeline.py
```

### 2. Train with Dummy Data:
```bash
python scripts/train_vietnamese_kws.py \
  --config configs/vietnamese_kws_config.yaml \
  --student_model standard \
  --use_dummy_data
```

### 3. Train with Real VIVOS Data:
```bash
python scripts/train_vietnamese_kws.py \
  --config configs/vietnamese_kws_config.yaml \
  --student_model standard \
  --data_dir ./data/vivos
```

### 4. Train Tiny Model for Edge Deployment:
```bash
python scripts/train_vietnamese_kws.py \
  --config configs/vietnamese_kws_config.yaml \
  --student_model tiny \
  --data_dir ./data/vivos
```

## ✅ Advantages Over ASR

1. **No Overfitting**: Only 11 classes vs 50K+ vocabulary
2. **Practical**: Real voice command applications
3. **Efficient**: Models are 10-100x smaller
4. **Stable Training**: Classification is much easier than sequence generation
5. **Better Metrics**: Clear accuracy/precision/recall per keyword
6. **Real-world Ready**: Perfect for voice interfaces

## 📊 Expected Results

- **Accuracy**: 85-95% on Vietnamese commands
- **Model Size**: 5K-50K parameters (vs 80M+ for ASR)
- **Training Time**: Minutes instead of hours
- **Inference**: Real-time on CPU/mobile devices
- **Applications**: Voice assistants, smart home, mobile apps

## 🎯 Next Steps

1. Run the test pipeline to verify everything works
2. Train on dummy data first to validate training loop
3. Train on real VIVOS data for actual performance
4. Evaluate per-keyword performance
5. Deploy tiny model for edge inference

This Vietnamese KWS system is much more practical and achievable than the full ASR approach!