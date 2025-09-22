"""
MobileNetV3 student model adapted for ASR tasks.
Efficient architecture designed for mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class HSwish(nn.Module):
    """Hard Swish activation function."""
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HSigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for 1D convolutions."""
    
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HSigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MobileBottleneck1D(nn.Module):
    """MobileNetV3 bottleneck block adapted for 1D convolutions."""
    
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel: int,
        stride: int,
        exp: int,
        se: bool = False,
        nl: str = 'RE'
    ):
        super().__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup
        
        # Activation function
        if nl == 'HS':
            nlin_layer = HSwish
        else:
            nlin_layer = nn.ReLU
        
        # Expansion
        if exp != inp:
            self.conv_expand = nn.Sequential(
                nn.Conv1d(inp, exp, 1, 1, 0, bias=False),
                nn.BatchNorm1d(exp),
                nlin_layer(inplace=True)
            )
        else:
            self.conv_expand = None
        
        # Depthwise
        self.conv_depthwise = nn.Sequential(
            nn.Conv1d(exp, exp, kernel, stride, kernel // 2, groups=exp, bias=False),
            nn.BatchNorm1d(exp),
            nlin_layer(inplace=True)
        )
        
        # SE
        if se:
            self.se = SEModule(exp)
        else:
            self.se = None
        
        # Pointwise
        self.conv_pointwise = nn.Sequential(
            nn.Conv1d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expansion
        if self.conv_expand is not None:
            out = self.conv_expand(x)
        else:
            out = x
        
        # Depthwise
        out = self.conv_depthwise(out)
        
        # SE
        if self.se is not None:
            out = self.se(out)
        
        # Pointwise
        out = self.conv_pointwise(out)
        
        # Residual connection
        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3ASR(nn.Module):
    """
    MobileNetV3 model adapted for ASR tasks.
    Efficient architecture for mobile deployment.
    """

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 51865,  # PhoWhisper vocab size
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        teacher_hidden_size: int = 512,
        width_mult: float = 1.0,
    ):
        super().__init__()
        
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        
        # MobileNetV3-Small configuration
        # [kernel, exp, out, se, nl, stride]
        mobile_setting = [
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, False, 'RE', 2],
            [3, 88, 24, False, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1],
        ]
        
        # Scale channels according to width multiplier
        input_channel = int(16 * width_mult)
        last_channel = int(1024 * width_mult)
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_mel_bins, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm1d(input_channel),
            HSwish(inplace=True)
        )
        
        # Mobile blocks
        self.blocks = nn.ModuleList()
        self.block_channels = []
        
        for i, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            output_channel = int(c * width_mult)
            exp_channel = int(exp * width_mult)
            
            self.blocks.append(
                MobileBottleneck1D(input_channel, output_channel, k, s, exp_channel, se, nl)
            )
            self.block_channels.append(output_channel)
            input_channel = output_channel
        
        # Last conv layer
        self.conv_last = nn.Sequential(
            nn.Conv1d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm1d(last_channel),
            HSwish(inplace=True)
        )
        
        # Feature adaptation layers for knowledge distillation
        self.feature_adapters = nn.ModuleDict()
        adaptation_points = [2, 4, 7, 10]  # Key blocks for feature extraction
        for i, block_idx in enumerate(adaptation_points):
            if block_idx < len(self.block_channels):
                self.feature_adapters[f'block_{block_idx}'] = nn.Linear(
                    self.block_channels[block_idx], teacher_hidden_size
                )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Sequence modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=last_channel,
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
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for knowledge distillation.
        
        Args:
            x: Input tensor of shape (batch_size, num_mel_bins, seq_len)
            
        Returns:
            Dictionary containing intermediate features
        """
        features = {}
        adapted_features = {}
        
        # First conv
        x = self.conv1(x)
        
        # Mobile blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Store features at key points
            adapter_name = f'block_{i}'
            if adapter_name in self.feature_adapters:
                features[adapter_name] = x
                # Average pool and adapt
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                adapted_features[adapter_name] = self.feature_adapters[adapter_name](pooled_feat)
        
        # Last conv
        x = self.conv_last(x)
        
        return {
            'raw_features': features,
            'adapted_features': adapted_features,
            'final_features': x
        }
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MobileNetV3 ASR model.
        
        Args:
            x: Input tensor of shape (batch_size, num_mel_bins, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        # Extract features
        if return_features:
            feature_dict = self.extract_features(x)
            x = feature_dict['final_features']
        else:
            # Standard forward pass
            x = self.conv1(x)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.conv_last(x)
        
        # Sequence modeling
        # Transpose for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (B, seq_len, last_channel)
        
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


def create_mobilenetv3_asr(
    num_mel_bins: int = 80,
    vocab_size: int = 51865,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    teacher_hidden_size: int = 512,
    width_mult: float = 1.0,
) -> MobileNetV3ASR:
    """
    Create a MobileNetV3 ASR model.
    
    Args:
        num_mel_bins: Number of mel-frequency bins
        vocab_size: Vocabulary size
        hidden_size: Hidden size for LSTM layers
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        teacher_hidden_size: Hidden size of teacher model for feature adaptation
        width_mult: Width multiplier for channels
        
    Returns:
        MobileNetV3ASR model
    """
    return MobileNetV3ASR(
        num_mel_bins=num_mel_bins,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        teacher_hidden_size=teacher_hidden_size,
        width_mult=width_mult,
    )