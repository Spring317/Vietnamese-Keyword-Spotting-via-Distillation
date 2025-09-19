# Vietnamese ASR Knowledge Distillation Pipeline

This project implements knowledge distillation for Vietnamese Automatic Speech Recognition (ASR) using PhoWhisper-base as the teacher model and ResNet18, MobileNetV3, and QuartzNet as student models.

# Vietnamese Keyword Spotting with Knowledge Distillation

A knowledge distillation framework for Vietnamese Keyword Spotting (KWS) using PhoWhisper teacher and lightweight student models (ResNet18, MobileNetV3).

##  Overview

This repository implements a Vietnamese Keyword Spotting system that uses knowledge distillation to train lightweight student models from a powerful PhoWhisper teacher model. The system can detect the 9 most frequent Vietnamese words plus negative samples with high accuracy.

### Key Features

- **Teacher Model**: PhoWhisper-base encoder (~21M parameters)
- **Student Models**: 
  - ResNet18 (tiny): ~11.5M parameters
  - MobileNetV3 (standard): ~5M parameters
- **Knowledge Distillation**: Temperature-scaled softmax with feature matching
- **Dataset**: VIVOS Vietnamese speech corpus
- **Keywords**: 9 most frequent Vietnamese words for optimal training data

##  Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ storage for models and data

### Dependencies
Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- `torch>=2.8.0`
- `torchaudio>=2.8.0`
- `transformers>=4.53.0`
- `librosa>=0.11.0`
- `datasets>=4.1.0`
- `tensorboard>=2.20.0`

##  Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd distile_asr_phoWhisper
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Place your VIVOS dataset in the following structure:
```
data/vivos/
├── train/
│   └── prompts.txt
└── test/
    └── prompts.txt
```

### 4. Train with Knowledge Distillation
```bash
# Train ResNet18 student (tiny)
python scripts/train_vietnamese_kws.py \
    --config configs/vietnamese_kws_config.yaml \
    --student_model tiny \
    --data_dir ./data/vivos \
    --output_dir ./outputs_resnet18

# Train MobileNetV3 student (standard)  
python scripts/train_vietnamese_kws.py \
    --config configs/vietnamese_kws_config.yaml \
    --student_model standard \
    --data_dir ./data/vivos \
    --output_dir ./outputs_mobilenet
```

### 5. Monitor Training
```bash
# View logs
tail -f outputs_resnet18/logs/kws_training.log

# TensorBoard (in another terminal)
tensorboard --logdir outputs_resnet18/tensorboard
```

## 🏗️ Architecture

### Teacher Model: PhoWhisper-based KWS
```python
PhoWhisperKWSTeacher:
├── Whisper Encoder (21M params, frozen)
├── Adaptation Layers (512→256→10)
├── Feature Projector (for distillation)
└── Classifier (10 classes)
```

### Student Models

#### ResNet18 (Tiny)
```python
ResNet18KWSStudent:
├── Input Adapter (1→3 channels) 
├── ResNet18 Backbone (11M params)
├── Global Average Pooling
├── Classifier (512→256→10)
└── Feature Projector (for distillation)
```

#### MobileNetV3 (Standard)
```python
MobileNetV3KWSStudent:
├── Input Adapter (audio preprocessing)
├── MobileNetV3 Backbone (5M params)
├── Global Average Pooling  
├── Classifier (960→256→10)
└── Feature Projector (for distillation)
```

##  Target Keywords

The system detects the 9 most frequent Vietnamese words:

| Class | Keyword | Meaning | Frequency |
|-------|---------|---------|-----------|
| 0 | có | have/exist | 2,243 (1.38%) |
| 1 | là | is/be | 1,854 (1.14%) |
| 2 | không | no/not | 1,838 (1.13%) |
| 3 | một | one/a | 1,777 (1.10%) |
| 4 | của | of/belonging to | 1,698 (1.05%) |
| 5 | và | and | 1,583 (0.98%) |
| 6 | người | person/people | 1,393 (0.86%) |
| 7 | những | the/those | 1,366 (0.84%) |
| 8 | tôi | I/me | 1,291 (0.80%) |
| 9 | negative | non-keyword speech | - |

##  Configuration

### Key Configuration File: `configs/vietnamese_kws_config.yaml`

```yaml
# Training settings
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.002
  early_stopping_patience: 15

# Knowledge distillation
distillation:
  temperature: 6.0      # Soft target smoothing
  alpha: 0.8           # Teacher knowledge weight
  beta: 0.2            # Hard target weight
  feature_loss_weight: 0.3

# Models
models:
  teacher:
    hidden_size: 512
    freeze_encoder: true
  
  tiny_student:        # ResNet18
    dropout: 0.1
    pretrained: true
    
  student:             # MobileNetV3
    dropout: 0.2
    pretrained: false
```

##  Project Structure

```
distile_asr_phoWhisper/
├── configs/
│   └── vietnamese_kws_config.yaml    # Training configuration
├── src/
│   ├── data/
│   │   └── vietnamese_kws_dataset.py # Dataset implementation
│   └── models/
│       └── kws_models.py             # Model architectures
├── scripts/
│   └── train_vietnamese_kws.py       # Training script
├── data/
│   └── vivos/                        # VIVOS dataset
├── outputs_*/                        # Training outputs
│   ├── logs/                         # Training logs
│   ├── checkpoints/                  # Model checkpoints
│   └── tensorboard/                  # TensorBoard logs
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

##  Expected Performance

| Model | Parameters | Accuracy | Speed |
|-------|------------|----------|-------|
| Teacher (PhoWhisper) | 21M | 95%+ | Baseline |
| Student (ResNet18) | 11.5M | 85-90% | 2x faster |
| Student (MobileNetV3) | 5M | 80-85% | 3x faster |

##  Advanced Usage

### Training with Dummy Data
For testing without VIVOS dataset:
```bash
python scripts/train_vietnamese_kws.py \
    --config configs/vietnamese_kws_config.yaml \
    --student_model tiny \
    --use_dummy_data \
    --output_dir ./outputs_test
```

### Model Evaluation
```python
# Load trained model
import torch
from src.models.kws_models import create_tiny_kws_student

model = create_tiny_kws_student(num_classes=10)
checkpoint = torch.load('outputs_resnet18/checkpoints/best_kws_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    outputs = model(audio_features)
    predictions = torch.argmax(outputs['logits'], dim=-1)
```

<!-- ## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
# Reduce batch size in config
training:
  batch_size: 8  # Instead of 32
```

**2. Dataset Not Found**
```bash
# Use dummy data for testing
python scripts/train_vietnamese_kws.py --use_dummy_data
```

**3. Low Accuracy Scores**
- Check keyword frequency in dataset
- Increase number of epochs
- Adjust distillation temperature

##  Monitoring & Logging

### TensorBoard Metrics
```bash
tensorboard --logdir outputs_resnet18/tensorboard
```

Available metrics:
- Training/Validation Loss
- Accuracy per epoch
- Per-class Precision/Recall/F1
- Learning rate schedule
- Distillation loss components

##  Performance Benchmarks

### Training Time (50 epochs)
- **RTX 3080**: ~2-3 hours
- **RTX 4090**: ~1-2 hours  
- **CPU Only**: ~12-24 hours

### Inference Speed
- **ResNet18**: ~5ms per sample (GPU)
- **MobileNetV3**: ~3ms per sample (GPU)
- **Teacher**: ~10ms per sample (GPU) -->

##  License

This project is licensed under the MIT License.

##  Acknowledgments

- **VinAI Research** for PhoWhisper model
- **VIVOS** dataset creators
- **Hugging Face** for transformers library
- **PyTorch** team for the framework

---

**Happy Training! 🚀**

> This repository provides a complete framework for Vietnamese keyword spotting with state-of-the-art knowledge distillation techniques.

## Features

- **Teacher Model**: vinai/PhoWhisper-base for Vietnamese ASR
- **Student Models**: ResNet18, MobileNetV3, QuartzNet
- **Knowledge Distillation**: Feature matching, attention transfer, output distillation
- **Evaluation**: WER, CER metrics for Vietnamese speech
- **Flexible Configuration**: Hydra-based configuration system

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train_vietnamese_kws.py \
    --config configs/vietnamese_kws_config.yaml \
    --student_model tiny \
    --data_dir ./data/vivos \
    --output_dir ./outputs_frequent_words_test
```

### Evaluation
```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pt
```

### Inference
```bash
python scripts/inference.py --audio_path path/to/audio.wav --model_path checkpoints/best_model.pt
```

## Configuration

Configuration files are located in the `configs/` directory. You can modify training parameters, model architectures, and distillation settings.

## Models

- **PhoWhisper Teacher**: Pre-trained Vietnamese Whisper model
- **ResNet18 Student**: Adapted for ASR with 1D convolutions


## License

MIT License