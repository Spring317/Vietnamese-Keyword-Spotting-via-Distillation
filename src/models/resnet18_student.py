"""
ResNet18 student model adapted for ASR tasks.
Uses 1D convolutions for processing audio spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class BasicBlock1D(nn.Module):
    """1D BasicBlock for ResNet18 adapted for ASR."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18ASR(nn.Module):
    """
    ResNet18 model adapted for ASR tasks.
    Processes mel-spectrogram features and outputs logits for Vietnamese ASR.
    """

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 51865,  # PhoWhisper vocab size
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        teacher_hidden_size: int = 512,
    ):
        super().__init__()
        
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        
        # Input projection layer
        self.input_projection = nn.Conv1d(num_mel_bins, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Feature adaptation layers for knowledge distillation
        self.feature_adapters = nn.ModuleDict({
            'layer1': nn.Linear(64, teacher_hidden_size),
            'layer2': nn.Linear(128, teacher_hidden_size),
            'layer3': nn.Linear(256, teacher_hidden_size),
            'layer4': nn.Linear(512, teacher_hidden_size),
        })
        
        # Sequence modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)  # *2 for bidirectional
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BasicBlock1D.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )

        layers = []
        layers.append(BasicBlock1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            x: Input tensor of shape (batch_size, num_mel_bins, seq_len)
            
        Returns:
            Dictionary containing intermediate features
        """
        features = {}
        
        # Input processing
        x = self.input_projection(x)  # (B, 64, seq_len)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x1 = self.layer1(x)  # (B, 64, seq_len)
        features['layer1'] = x1
        
        x2 = self.layer2(x1)  # (B, 128, seq_len)
        features['layer2'] = x2
        
        x3 = self.layer3(x2)  # (B, 256, seq_len)
        features['layer3'] = x3
        
        x4 = self.layer4(x3)  # (B, 512, seq_len)
        features['layer4'] = x4
        
        # Adapt features for knowledge distillation
        adapted_features = {}
        for layer_name, feat in features.items():
            # Average pool over spatial dimension and adapt
            pooled_feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # (B, channels)
            adapted_features[layer_name] = self.feature_adapters[layer_name](pooled_feat)
        
        return {
            'raw_features': features,
            'adapted_features': adapted_features,
            'final_features': x4
        }
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ResNet18 ASR model.
        
        Args:
            x: Input tensor of shape (batch_size, num_mel_bins, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        batch_size = x.size(0)
        
        # Extract features
        if return_features:
            feature_dict = self.extract_features(x)
            x = feature_dict['final_features']
        else:
            # Standard forward pass
            x = self.input_projection(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # Sequence modeling
        # Transpose for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (B, seq_len, 512)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden_size * 2)
        
        # Apply dropout and classify
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # (B, seq_len, vocab_size)
        
        result = {'logits': logits}
        
        if return_features:
            result.update(feature_dict)
            result['lstm_features'] = lstm_out
            
        return result
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_resnet18_asr(
    num_mel_bins: int = 80,
    vocab_size: int = 51865,
    hidden_size: int = 512,
    num_layers: int = 4,
    dropout: float = 0.1,
    teacher_hidden_size: int = 512,
) -> ResNet18ASR:
    """
    Create a ResNet18 ASR model.
    
    Args:
        num_mel_bins: Number of mel-frequency bins
        vocab_size: Vocabulary size
        hidden_size: Hidden size for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        teacher_hidden_size: Hidden size of teacher model for feature adaptation
        
    Returns:
        ResNet18ASR model
    """
    return ResNet18ASR(
        num_mel_bins=num_mel_bins,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_hidden_size=teacher_hidden_size,
    )