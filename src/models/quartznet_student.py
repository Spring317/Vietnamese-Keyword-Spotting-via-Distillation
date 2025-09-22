"""
QuartzNet student model for ASR tasks.
Specialized architecture designed specifically for speech recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class SeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, 1,
            stride=1, padding=0, dilation=1,
            groups=1, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class QuartzBlock(nn.Module):
    """QuartzNet block with multiple separable convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_sub_blocks: int = 5,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        
        self.residual = residual and (in_channels == out_channels)
        self.num_sub_blocks = num_sub_blocks
        
        # Sub-blocks
        self.sub_blocks = nn.ModuleList()
        for i in range(num_sub_blocks):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels
                
            sub_block = nn.Sequential(
                SeparableConv1d(
                    in_ch, out_channels, kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True) if i < num_sub_blocks - 1 else nn.Identity(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.sub_blocks.append(sub_block)
        
        # Residual connection
        if self.residual:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
        # Final activation
        self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # Pass through sub-blocks
        for sub_block in self.sub_blocks:
            x = sub_block(x)
        
        # Add residual
        x = x + residual
        x = self.final_activation(x)
        
        return x


class QuartzNetASR(nn.Module):
    """
    QuartzNet model for ASR tasks.
    Based on the QuartzNet architecture from NVIDIA.
    """

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 51865,  # PhoWhisper vocab size
        hidden_size: int = 512,
        teacher_hidden_size: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        
        # QuartzNet5x5 configuration
        # [out_channels, kernel_size, num_sub_blocks]
        quartz_config = [
            [256, 33, 1],   # C1
            [256, 39, 5],   # B1
            [256, 51, 5],   # B2
            [512, 63, 5],   # B3
            [512, 75, 5],   # B4
            [512, 87, 5],   # B5
            [1024, 1, 1],   # C2
            [1024, 1, 1],   # C3
        ]
        
        # Prologue: Initial convolution
        self.prologue = nn.Sequential(
            nn.Conv1d(num_mel_bins, 256, kernel_size=33, stride=2, padding=16),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # QuartzNet blocks
        self.blocks = nn.ModuleList()
        self.block_channels = []
        in_channels = 256
        
        for i, (out_channels, kernel_size, num_sub_blocks) in enumerate(quartz_config):
            if i < 6:  # B1-B5 and C1 use QuartzBlock
                block = QuartzBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_sub_blocks=num_sub_blocks,
                    dropout=dropout,
                    residual=(i > 0)  # No residual for C1
                )
            else:  # C2, C3 use regular conv
                block = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            
            self.blocks.append(block)
            self.block_channels.append(out_channels)
            in_channels = out_channels
        
        # Feature adaptation layers for knowledge distillation
        self.feature_adapters = nn.ModuleDict()
        adaptation_points = [1, 3, 5, 7]  # Key blocks for feature extraction
        for block_idx in adaptation_points:
            if block_idx < len(self.block_channels):
                self.feature_adapters[f'block_{block_idx}'] = nn.Linear(
                    self.block_channels[block_idx], teacher_hidden_size
                )
        
        # Epilogue: Output projection
        self.epilogue = nn.Sequential(
            nn.Conv1d(1024, vocab_size, kernel_size=1),
            nn.Dropout(dropout)
        )
        
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
        
        # Prologue
        x = self.prologue(x)
        
        # QuartzNet blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Store features at key points
            adapter_name = f'block_{i}'
            if adapter_name in self.feature_adapters:
                features[adapter_name] = x
                # Average pool and adapt
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                adapted_features[adapter_name] = self.feature_adapters[adapter_name](pooled_feat)
        
        result = {'final_features': x}
        result.update(features)
        result.update(adapted_features)
        return result
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the QuartzNet ASR model.
        
        Args:
            x: Input tensor of shape (batch_size, num_mel_bins, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        result = {'logits': torch.empty(0)}  # Initialize result
        
        # Extract features
        if return_features:
            feature_dict = self.extract_features(x)
            x = feature_dict['final_features']
            # Add features to result
            for key, value in feature_dict.items():
                if key != 'final_features':
                    result[key] = value
        else:
            # Standard forward pass
            x = self.prologue(x)
            
            for block in self.blocks:
                x = block(x)
        
        # Output projection
        logits = self.epilogue(x)  # (B, vocab_size, seq_len)
        
        # Transpose to (B, seq_len, vocab_size)
        logits = logits.transpose(1, 2)
        
        result['logits'] = logits
        return result
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class QuartzNetSmallASR(nn.Module):
    """
    Smaller QuartzNet model for ASR tasks.
    Reduced version for faster training and inference.
    """

    def __init__(
        self,
        num_mel_bins: int = 80,
        vocab_size: int = 51865,
        hidden_size: int = 256,
        teacher_hidden_size: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_mel_bins = num_mel_bins
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        
        # Smaller QuartzNet configuration
        quartz_config = [
            [128, 11, 1],   # C1
            [128, 13, 3],   # B1
            [128, 15, 3],   # B2
            [256, 17, 3],   # B3
            [256, 19, 3],   # B4
            [512, 1, 1],    # C2
            [512, 1, 1],    # C3
        ]
        
        # Prologue
        self.prologue = nn.Sequential(
            nn.Conv1d(num_mel_bins, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # QuartzNet blocks
        self.blocks = nn.ModuleList()
        self.block_channels = []
        in_channels = 128
        
        for i, (out_channels, kernel_size, num_sub_blocks) in enumerate(quartz_config):
            if i < 5:  # B1-B4 and C1
                block = QuartzBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_sub_blocks=num_sub_blocks,
                    dropout=dropout,
                    residual=(i > 0)
                )
            else:  # C2, C3
                block = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                )
            
            self.blocks.append(block)
            self.block_channels.append(out_channels)
            in_channels = out_channels
        
        # Feature adaptation layers
        self.feature_adapters = nn.ModuleDict()
        adaptation_points = [1, 3, 5]
        for block_idx in adaptation_points:
            if block_idx < len(self.block_channels):
                self.feature_adapters[f'block_{block_idx}'] = nn.Linear(
                    self.block_channels[block_idx], teacher_hidden_size
                )
        
        # Output projection
        self.epilogue = nn.Sequential(
            nn.Conv1d(512, vocab_size, kernel_size=1),
            nn.Dropout(dropout)
        )
        
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
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate features for knowledge distillation."""
        features = {}
        adapted_features = {}
        
        x = self.prologue(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            adapter_name = f'block_{i}'
            if adapter_name in self.feature_adapters:
                features[adapter_name] = x
                pooled_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                adapted_features[adapter_name] = self.feature_adapters[adapter_name](pooled_feat)
        
        result = {'final_features': x}
        result.update(features)
        result.update(adapted_features)
        return result
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through the QuartzNet model."""
        result = {'logits': torch.empty(0)}  # Initialize result
        
        if return_features:
            feature_dict = self.extract_features(x)
            x = feature_dict['final_features']
            # Add features to result
            for key, value in feature_dict.items():
                if key != 'final_features':
                    result[key] = value
        else:
            x = self.prologue(x)
            for block in self.blocks:
                x = block(x)
        
        logits = self.epilogue(x)
        logits = logits.transpose(1, 2)
        
        result['logits'] = logits
        return result
    
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get the model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_quartznet_asr(
    num_mel_bins: int = 80,
    vocab_size: int = 51865,
    hidden_size: int = 512,
    teacher_hidden_size: int = 512,
    dropout: float = 0.1,
    model_size: str = 'large',
) -> nn.Module:
    """
    Create a QuartzNet ASR model.
    
    Args:
        num_mel_bins: Number of mel-frequency bins
        vocab_size: Vocabulary size
        hidden_size: Hidden size
        teacher_hidden_size: Hidden size of teacher model for feature adaptation
        dropout: Dropout rate
        model_size: Model size ('large' or 'small')
        
    Returns:
        QuartzNet ASR model
    """
    if model_size == 'large':
        return QuartzNetASR(
            num_mel_bins=num_mel_bins,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            teacher_hidden_size=teacher_hidden_size,
            dropout=dropout,
        )
    elif model_size == 'small':
        return QuartzNetSmallASR(
            num_mel_bins=num_mel_bins,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            teacher_hidden_size=teacher_hidden_size,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")