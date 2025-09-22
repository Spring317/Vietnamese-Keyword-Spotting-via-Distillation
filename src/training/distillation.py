"""
Knowledge distillation framework for Vietnamese ASR.
Implements various distillation strategies including feature matching,
attention transfer, and output distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined loss function for knowledge distillation with improved scaling.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3, 
        gamma: float = 0.2,
        temperature: float = 6.0,
        feature_loss_weight: float = 0.5,
        attention_loss_weight: float = 0.3,
        label_smoothing: float = 0.1,
        loss_scaling: float = 1.0,  # Add loss scaling factor
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.label_smoothing = label_smoothing
        self.loss_scaling = loss_scaling
        
        # Loss functions with label smoothing
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.feature_loss = nn.MSELoss()
        
        # Track loss history for monitoring
        self.loss_history = {
            'soft_loss': [],
            'hard_loss': [],
            'feature_loss': [],
            'total_loss': []
        }
        """
        Initialize distillation loss.
        
        Args:
            alpha: Weight for soft target loss
            beta: Weight for hard target loss  
            gamma: Weight for feature distillation loss
            temperature: Temperature for knowledge distillation
            feature_loss_weight: Weight for feature matching loss
            attention_loss_weight: Weight for attention transfer loss
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        
        # Loss functions
        self.hard_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.feature_loss = nn.MSELoss()
        self.attention_loss = nn.MSELoss()
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Dictionary containing student model outputs
            teacher_outputs: Dictionary containing teacher model outputs
            targets: Ground truth targets (optional)
            
        Returns:
            Dictionary containing various loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Extract student and teacher logits
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs.get('soft_targets', teacher_outputs.get('logits'))
        
        # Check if teacher logits exist
        if teacher_logits is None:
            logger.warning("No teacher logits found, returning zero loss")
            return {'total_loss': torch.tensor(0.0, requires_grad=True)}
        
        # Handle different shapes and batch size mismatches
        if len(student_logits.shape) == 3:  # [batch, seq, vocab]
            # Average over sequence dimension for classification
            student_logits = student_logits.mean(dim=1)  # [batch, vocab]
        
        if len(teacher_logits.shape) == 3:  # [batch, seq, vocab]
            # Average over sequence dimension for classification
            teacher_logits = teacher_logits.mean(dim=1)  # [batch, vocab]
        
        # Ensure same batch size by taking minimum
        min_batch_size = min(student_logits.size(0), teacher_logits.size(0))
        student_logits = student_logits[:min_batch_size]
        teacher_logits = teacher_logits[:min_batch_size]
        
        vocab_size = student_logits.size(-1)
        
        # 1. Soft target loss (knowledge distillation)
        # Apply temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence with temperature scaling and loss scaling
        soft_loss = self.soft_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        # Apply loss scaling to prevent too rapid convergence
        soft_loss = soft_loss * self.loss_scaling
        
        losses['soft_loss'] = soft_loss
        total_loss += self.alpha * soft_loss
        
        # Track loss for monitoring
        self.loss_history['soft_loss'].append(soft_loss.item())
        if len(self.loss_history['soft_loss']) > 1000:  # Keep recent history
            self.loss_history['soft_loss'] = self.loss_history['soft_loss'][-500:]
        
        # 2. Hard target loss (ground truth)
        if targets is not None:
            # Reshape for cross entropy
            student_logits_flat = student_logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            hard_loss = self.hard_loss(student_logits_flat, targets_flat)
            losses['hard_loss'] = hard_loss
            total_loss += self.beta * hard_loss
        
        # 3. Feature distillation loss
        feature_loss = self._compute_feature_loss(student_outputs, teacher_outputs)
        if feature_loss is not None:
            losses['feature_loss'] = feature_loss
            total_loss += self.gamma * self.feature_loss_weight * feature_loss
        
        # 4. Attention transfer loss
        attention_loss = self._compute_attention_loss(student_outputs, teacher_outputs)
        if attention_loss is not None:
            losses['attention_loss'] = attention_loss
            total_loss += self.gamma * self.attention_loss_weight * attention_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_feature_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute feature matching loss between student and teacher."""
        feature_losses = []
        
        # Look for adapted features in student outputs
        for key in student_outputs:
            if key.startswith('block_') and key in teacher_outputs:
                student_feat = student_outputs[key]
                teacher_feat = teacher_outputs[key]
                
                # Ensure same dimensions
                if student_feat.shape == teacher_feat.shape:
                    loss = self.feature_loss(student_feat, teacher_feat)
                    feature_losses.append(loss)
        
        # Look for encoder features
        if 'encoder_features' in teacher_outputs:
            teacher_encoder_feats = teacher_outputs['encoder_features']
            # Handle case where encoder_features might be a tensor instead of dict
            if isinstance(teacher_encoder_feats, dict):
                for layer_name, teacher_feat in teacher_encoder_feats.items():
                    if layer_name in student_outputs:
                        student_feat = student_outputs[layer_name]
                        if student_feat.shape == teacher_feat.shape:
                            loss = self.feature_loss(student_feat, teacher_feat)
                            feature_losses.append(loss)
            elif isinstance(teacher_encoder_feats, torch.Tensor):
                # If it's a tensor, try to match with student features
                if 'encoder_features' in student_outputs:
                    student_encoder_feats = student_outputs['encoder_features']
                    if isinstance(student_encoder_feats, torch.Tensor):
                        # Match batch sizes
                        min_batch = min(student_encoder_feats.size(0), teacher_encoder_feats.size(0))
                        student_feat = student_encoder_feats[:min_batch]
                        teacher_feat = teacher_encoder_feats[:min_batch]
                        
                        if student_feat.shape == teacher_feat.shape:
                            loss = self.feature_loss(student_feat, teacher_feat)
                            feature_losses.append(loss)
        
        if feature_losses:
            return torch.stack(feature_losses).mean()
        return None
    
    def _compute_attention_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute attention transfer loss."""
        attention_losses = []
        
        # Look for attention weights in teacher outputs
        if 'attention_weights' in teacher_outputs:
            teacher_attentions = teacher_outputs['attention_weights']
            
            # For now, we don't have explicit attention in student models
            # This can be extended based on specific student architectures
            pass
        
        if attention_losses:
            return torch.stack(attention_losses).mean()
        return None


class KnowledgeDistillationTrainer:
    """
    Trainer for knowledge distillation in ASR.
    Handles the training loop, loss computation, and model updates.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        distillation_loss: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            distillation_loss: Distillation loss function
            optimizer: Optimizer for student model
            device: Training device
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_loss = distillation_loss
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_features: bool = True,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch containing input_features and optionally labels
            return_features: Whether to extract features for distillation
            
        Returns:
            Dictionary containing loss values
        """
        self.student_model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_features = batch['input_features'].to(self.device)
        labels = batch.get('labels')
        if labels is not None:
            labels = labels.to(self.device)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_features, return_features=return_features)
        
        # Student forward pass
        student_outputs = self.student_model(input_features, return_features=return_features)
        
        # Compute distillation loss
        loss_dict = self.distillation_loss(student_outputs, teacher_outputs, labels)
        
        # Backward pass
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Convert losses to float for logging
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in loss_dict.items()}
        
        return loss_values
    
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_features: bool = True,
    ) -> Dict[str, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            return_features: Whether to extract features
            
        Returns:
            Dictionary containing loss values
        """
        self.student_model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_features = batch['input_features'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            # Teacher forward pass
            teacher_outputs = self.teacher_model(input_features, return_features=return_features)
            
            # Student forward pass
            student_outputs = self.student_model(input_features, return_features=return_features)
            
            # Compute distillation loss
            loss_dict = self.distillation_loss(student_outputs, teacher_outputs, labels)
        
        # Convert losses to float
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in loss_dict.items()}
        
        return loss_values
    
    def save_checkpoint(self, filepath: str, epoch: int, best_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Union[int, float]]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        
        return {
            'epoch': checkpoint['epoch'],
            'best_loss': checkpoint['best_loss']
        }


def create_distillation_trainer(
    teacher_model: nn.Module,
    student_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distillation_config: Optional[Dict] = None,
) -> KnowledgeDistillationTrainer:
    """
    Create a knowledge distillation trainer.
    
    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        optimizer: Optimizer for student model
        device: Training device
        distillation_config: Configuration for distillation loss
        
    Returns:
        KnowledgeDistillationTrainer instance
    """
    if distillation_config is None:
        distillation_config = {}
    
    distillation_loss = DistillationLoss(**distillation_config)
    
    return KnowledgeDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        distillation_loss=distillation_loss,
        optimizer=optimizer,
        device=device,
    )