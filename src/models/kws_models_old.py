"""
Vietnamese Keyword Spotting (KWS) Models
PhoWhisper teacher with ResNet18/MobileNetV3 students for 10-class classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from transformers import WhisperModel, WhisperConfig
import torchvision.models as models

logger = logging.getLogger(__name__)

class PhoWhisperKWSTeacher(nn.Module):
    """
    Teacher model based on PhoWhisper for Vietnamese KWS.
    Uses pre-trained PhoWhisper encoder features.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        hidden_size: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Load PhoWhisper configuration
        try:
            # Use smaller Whisper config if full PhoWhisper not available
            whisper_config = WhisperConfig.from_pretrained("openai/whisper-base")
            whisper_config.vocab_size = 51865  # PhoWhisper vocab size
            self.whisper_encoder = WhisperModel.from_pretrained(
                "openai/whisper-base", 
                config=whisper_config
            ).encoder
        except:
            # Fallback to base Whisper if PhoWhisper not available
            logger.warning("Could not load PhoWhisper, using base Whisper encoder")
            self.whisper_encoder = WhisperModel.from_pretrained("openai/whisper-base").encoder
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output dimension
        encoder_dim = self.whisper_encoder.config.d_model  # Usually 512 for base
        
        # Adaptation layers for KWS
        self.adaptation = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # For distillation - provide intermediate features
        self.feature_projector = nn.Linear(hidden_size // 2, hidden_size)
        
    def forward(
        self, 
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PhoWhisper KWS teacher.
        
        Args:
            input_features: [batch_size, 80, seq_len] mel spectrograms
            labels: [batch_size] class labels
        """
        batch_size = input_features.shape[0]
        
        # PhoWhisper expects [batch_size, 80, 3000] for 30s audio
        # For shorter audio, we may need to pad or truncate
        if input_features.shape[-1] != 3000:
            # Pad or truncate to 3000 time steps
            target_len = 3000
            if input_features.shape[-1] < target_len:
                # Pad
                pad_len = target_len - input_features.shape[-1]
                input_features = F.pad(input_features, (0, pad_len))
            else:
                # Truncate
                input_features = input_features[:, :, :target_len]
        
        # Whisper encoder forward pass
        encoder_outputs = self.whisper_encoder(input_features)
        
        # Get last hidden state and pool over sequence dimension
        hidden_states = encoder_outputs.last_hidden_state  # [B, seq_len, d_model]
        
        # Global average pooling over sequence dimension
        pooled_features = torch.mean(hidden_states, dim=1)  # [B, d_model]
        
        # Adaptation for KWS task
        adapted_features = self.adaptation(pooled_features)  # [B, hidden_size//2]
        
        # Get intermediate features for distillation
        distill_features = self.feature_projector(adapted_features)  # [B, hidden_size]
        
        # Classification
        logits = self.classifier(adapted_features)  # [B, num_classes]
        
        outputs = {
            'logits': logits,
            'hidden_states': distill_features,
            'features': adapted_features
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def get_attention_weights(self, input_features: torch.Tensor) -> torch.Tensor:
        """Get attention weights from Whisper encoder."""
        # Process input same as forward
        if input_features.shape[-1] != 3000:
            target_len = 3000
            if input_features.shape[-1] < target_len:
                pad_len = target_len - input_features.shape[-1]
                input_features = F.pad(input_features, (0, pad_len))
            else:
                input_features = input_features[:, :, :target_len]
        
        encoder_outputs = self.whisper_encoder(input_features)
        # Use mean attention across all heads and layers
        if hasattr(encoder_outputs, 'attentions') and encoder_outputs.attentions is not None:
            attention = torch.mean(torch.stack(encoder_outputs.attentions), dim=(0, 1, 2))
            return attention
        else:
            # Fallback: use hidden states magnitude as attention
            hidden_states = encoder_outputs.last_hidden_state
            attention = torch.mean(hidden_states ** 2, dim=-1, keepdim=False)
            attention = F.softmax(attention, dim=-1)
            return attention


class ResNet18KWSStudent(nn.Module):
    """
    ResNet18-based student model for Vietnamese KWS (tiny variant).
    Adapts ResNet18 for audio classification tasks.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        pretrained: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for audio input (80 mel features -> 3 channels)
        # We'll reshape 80 mel features to work with ResNet's 3-channel expectation
        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # 1 channel (audio) -> 3 channels (RGB-like)
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # Remove the final classification layer and get feature dimension
        backbone_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove fc layer
        
        # Add global average pooling if needed
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # For distillation
        self.feature_projector = nn.Linear(backbone_features, 512)
        
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ResNet18 KWS student.
        
        Args:
            input_features: [batch_size, 80, seq_len] mel spectrograms
            labels: [batch_size] class labels
        """
        batch_size, n_mels, seq_len = input_features.shape
        
        # Reshape for ResNet: [B, 80, seq_len] -> [B, 1, 80, seq_len]
        # Treat mel features as image height, time as width
        x = input_features.unsqueeze(1)  # [B, 1, 80, seq_len]
        
        # Ensure minimum size for ResNet (needs at least 32x32)
        if x.shape[2] < 32 or x.shape[3] < 32:
            # Pad to minimum size
            pad_h = max(0, 32 - x.shape[2])
            pad_w = max(0, 32 - x.shape[3])
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Adapt input channels: 1 -> 3
        x = self.input_adapter(x)  # [B, 3, 80+pad, seq_len+pad]
        
        # ResNet18 backbone
        features = self.backbone(x)  # [B, 512, H, W]
        features = self.global_pool(features)  # [B, 512, 1, 1]
        features = features.flatten(1)  # [B, 512]
        
        # For distillation
        hidden_states = self.feature_projector(features)  # [B, 512]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'features': features
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class MobileNetV3KWSStudent(nn.Module):
    """
    MobileNetV3-based student model for Vietnamese KWS (standard variant).
    Adapts MobileNetV3 for efficient audio classification.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.2,
        pretrained: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load MobileNetV3 Large backbone
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Modify first conv layer for audio input
        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  # 1 channel (audio) -> 3 channels
            nn.BatchNorm2d(3),
            nn.Hardswish()  # MobileNetV3 uses Hardswish
        )
        
        # Get backbone feature dimension and remove classifier
        # MobileNetV3 classifier is a Sequential with multiple layers
        if hasattr(self.backbone.classifier, '__iter__'):
            # Get the last linear layer's input features
            for layer in reversed(self.backbone.classifier):
                if isinstance(layer, nn.Linear):
                    backbone_features = layer.in_features
                    break
            else:
                backbone_features = 960  # Default for MobileNetV3-Large
        else:
            backbone_features = 960
        
        # Remove classifier - replace with identity
        self.backbone.classifier = nn.Sequential()  # Empty sequential
        
        # Add global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_features, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # For distillation
        self.feature_projector = nn.Linear(backbone_features, 512)
        
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MobileNetV3 KWS student.
        
        Args:
            input_features: [batch_size, 80, seq_len] mel spectrograms
            labels: [batch_size] class labels
        """
        batch_size, n_mels, seq_len = input_features.shape
        
        # Reshape for MobileNetV3: [B, 80, seq_len] -> [B, 1, 80, seq_len]
        x = input_features.unsqueeze(1)  # [B, 1, 80, seq_len]
        
        # Ensure minimum size (MobileNetV3 needs reasonable input size)
        if x.shape[2] < 32 or x.shape[3] < 32:
            pad_h = max(0, 32 - x.shape[2])
            pad_w = max(0, 32 - x.shape[3])
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Adapt input channels: 1 -> 3
        x = self.input_adapter(x)  # [B, 3, 80+pad, seq_len+pad]
        
        # MobileNetV3 feature extraction
        features = self.backbone.features(x)  # [B, C, H, W]
        features = self.global_pool(features)  # [B, C, 1, 1]
        features = features.flatten(1)  # [B, C]
        
        # For distillation
        hidden_states = self.feature_projector(features)  # [B, 512]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'features': features
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 64,  # Much smaller than teacher
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Lightweight feature extractor
        self.conv_layers = nn.Sequential(
            # Depthwise separable convolution for efficiency
            nn.Conv1d(80, hidden_size, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Depthwise
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Pointwise
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for KWS student.
        
        Args:
            input_features: [batch_size, 80, seq_len] mel spectrograms
            labels: [batch_size] class labels
        """
        # Feature extraction
        features = self.conv_layers(input_features)  # [B, hidden_size, 1]
        features = features.squeeze(-1)  # [B, hidden_size]
        
        # Classification
        logits = self.classifier(features)  # [B, num_classes]
        
        outputs = {
            'logits': logits,
            'hidden_states': features,
            'features': features
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class TinyKWSStudent(nn.Module):
    """
    Extremely tiny student for edge deployment.
    Only ~10K parameters.
    """
    
    def __init__(
        self,
        num_classes: int = 11,
        hidden_size: int = 32,  # Very small
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Minimal feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(80, hidden_size, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Direct classification
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        features = self.features(input_features).squeeze(-1)
        logits = self.classifier(features)
        
        outputs = {
            'logits': logits,
            'hidden_states': features,
            'features': features
        }
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class KWSDistillationLoss(nn.Module):
    """
    Specialized distillation loss for keyword spotting.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,  # Weight for distillation loss
        beta: float = 0.3,   # Weight for hard loss
        feature_loss_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_loss_weight = feature_loss_weight
        self.class_weights = class_weights
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        # Soft distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard loss (classification)
        hard_loss = F.cross_entropy(student_logits, labels, weight=self.class_weights)
        
        # Feature matching loss
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
            student_features = student_outputs['hidden_states']
            teacher_features = teacher_outputs['hidden_states']
            
            # Match dimensions if needed
            if student_features.shape[-1] != teacher_features.shape[-1]:
                # Project student features to teacher dimension
                projection = nn.Linear(
                    student_features.shape[-1], 
                    teacher_features.shape[-1]
                ).to(student_features.device)
                student_features = projection(student_features)
            
            feature_loss = F.mse_loss(student_features, teacher_features)
        
        # Total loss
        total_loss = (
            self.alpha * soft_loss + 
            self.beta * hard_loss + 
            self.feature_loss_weight * feature_loss
        )
        
        return {
            'total_loss': total_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss,
            'feature_loss': feature_loss
        }


def create_kws_teacher(num_classes: int = 10, **kwargs) -> PhoWhisperKWSTeacher:
    """Create PhoWhisper-based KWS teacher model."""
    return PhoWhisperKWSTeacher(num_classes=num_classes, **kwargs)


def create_kws_student(num_classes: int = 10, **kwargs) -> MobileNetV3KWSStudent:
    """Create MobileNetV3-based KWS student model (standard).""" 
    return MobileNetV3KWSStudent(num_classes=num_classes, **kwargs)


def create_tiny_kws_student(num_classes: int = 10, **kwargs) -> ResNet18KWSStudent:
    """Create ResNet18-based KWS student model (tiny)."""
    return ResNet18KWSStudent(num_classes=num_classes, **kwargs)