"""
Tiny Student Models designed specifically for small datasets like VIVOS.
Much smaller capacity to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TinyResNetASR(nn.Module):
    """
    Extremely small ResNet for ASR to prevent overfitting on small datasets.
    Only ~1M parameters instead of 80M+.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 64,        # Very small
        num_layers: int = 2,          # Only 2 layers
        dropout: float = 0.6,         # High dropout
        input_dim: int = 40,          # Reduced mel features
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Tiny feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Minimal transformer layers
        self.transformer_layers = nn.ModuleList([
            TinyTransformerLayer(hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Small output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize with small weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize with small weights to prevent overfitting."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_features: [batch_size, seq_len, input_dim]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = input_features.shape
        
        # Feature extraction
        features = self.feature_extractor(input_features)  # [B, T, H]
        features = self.layer_norm(features)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            features = layer(features, attention_mask)
        
        # Output projection
        logits = self.output_projection(features)  # [B, T, V]
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            # Flatten for cross entropy
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            
            # Ignore padding tokens (assuming -100)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            outputs["loss"] = loss
        
        return outputs
    
    def get_hidden_states(self, input_features: torch.Tensor) -> torch.Tensor:
        """Get hidden states for distillation."""
        features = self.feature_extractor(input_features)
        features = self.layer_norm(features)
        
        for layer in self.transformer_layers:
            features = layer(features)
            
        return features


class TinyTransformerLayer(nn.Module):
    """
    Extremely simple transformer layer.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.6):
        super().__init__()
        
        # Tiny self-attention
        self.self_attention = TinyAttention(hidden_size, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Small feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.ff_norm(hidden_states + ff_output)
        
        return hidden_states


class TinyAttention(nn.Module):
    """
    Minimal self-attention mechanism.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.6):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // 2  # Single attention head
        
        # Tiny projections
        self.query = nn.Linear(hidden_size, self.head_dim)
        self.key = nn.Linear(hidden_size, self.head_dim)
        self.value = nn.Linear(hidden_size, self.head_dim)
        
        self.output = nn.Linear(self.head_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Projections
        q = self.query(hidden_states)  # [B, T, H/2]
        k = self.key(hidden_states)    # [B, T, H/2]
        v = self.value(hidden_states)  # [B, T, H/2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Output projection
        output = self.output(attn_output)
        
        return output


def create_tiny_resnet_asr(**kwargs) -> TinyResNetASR:
    """Create tiny ResNet ASR model."""
    return TinyResNetASR(**kwargs)


class TinyMobileNetASR(nn.Module):
    """
    Extremely tiny MobileNet for ASR.
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 48,        # Even smaller
        dropout: float = 0.6,
        input_dim: int = 40,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Tiny depthwise separable convolutions
        self.conv_layers = nn.Sequential(
            # First layer
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Depthwise
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Pointwise
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Small classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, vocab_size)
        )
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Transpose for conv1d: [B, T, F] -> [B, F, T]
        features = input_features.transpose(1, 2)
        
        # Apply convolutions
        features = self.conv_layers(features)
        
        # Transpose back: [B, F, T] -> [B, T, F]
        features = features.transpose(1, 2)
        
        # Classify
        logits = self.classifier(features)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            outputs["loss"] = loss
        
        return outputs
    
    def get_hidden_states(self, input_features: torch.Tensor) -> torch.Tensor:
        features = input_features.transpose(1, 2)
        features = self.conv_layers(features)
        return features.transpose(1, 2)


def create_tiny_mobilenet_asr(**kwargs) -> TinyMobileNetASR:
    """Create tiny MobileNet ASR model."""
    return TinyMobileNetASR(**kwargs)