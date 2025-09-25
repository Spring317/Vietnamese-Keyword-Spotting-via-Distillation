#!/usr/bin/env python3
"""
Vietnamese Keyword Spotting (KWS) Training Pipeline
Knowledge distillation for 10 Vietnamese commands classification.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.vietnamese_kws_dataset import create_kws_dataloader, VietnameseKWSDataset
from src.models.kws_models import (
    create_kws_teacher, 
    create_kws_student, 
    create_tiny_kws_student,
    KWSDistillationLoss
)

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'kws_training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class KWSTrainer:
    """
    Trainer for Vietnamese KWS with knowledge distillation.
    """
    
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        device,
        config: Dict,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.class_weights = class_weights
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        
        # Setup scheduler with warmup
        warmup_epochs = 5
        total_steps = len(train_loader) * config['training']['num_epochs']
        warmup_steps = len(train_loader) * warmup_epochs
        
        # Use a more sophisticated scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['training']['optimizer']['learning_rate'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
        
        # Setup distillation loss
        self.criterion = KWSDistillationLoss(
            temperature=config['distillation']['temperature'],
            alpha=config['distillation']['alpha'],
            beta=config['distillation']['beta'],
            feature_loss_weight=config['distillation']['feature_loss_weight'],
            class_weights=self.class_weights
        )
        
        # Move models to device
        self.teacher_model = self.teacher_model.to(device)
        self.student_model = self.student_model.to(device)
        self.criterion = self.criterion.to(device)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        
    def train_epoch(self, epoch: int, writer: SummaryWriter) -> float:
        """Train one epoch."""
        self.student_model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_features, labels)
                
                # Student forward pass
                student_outputs = self.student_model(input_features, labels)
                
                # Calculate distillation loss
                loss_dict = self.criterion(student_outputs, teacher_outputs, labels)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # Log losses
                epoch_losses.append(total_loss.item())
                
                # Log to tensorboard
                global_step = epoch * len(self.train_loader) + batch_idx
                if batch_idx % 10 == 0:
                    writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                    writer.add_scalar('Train/Soft_Loss', loss_dict['soft_loss'].item(), global_step)
                    writer.add_scalar('Train/Hard_Loss', loss_dict['hard_loss'].item(), global_step)
                    writer.add_scalar('Train/Feature_Loss', loss_dict['feature_loss'].item(), global_step)
                    writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                logging.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate_epoch(self, epoch: int, writer: SummaryWriter) -> Tuple[float, Dict]:
        """Validate one epoch."""
        self.student_model.eval()
        val_losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    # Move batch to device
                    input_features = batch['input_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Teacher outputs
                    teacher_outputs = self.teacher_model(input_features, labels)
                    
                    # Student outputs
                    student_outputs = self.student_model(input_features, labels)
                    
                    # Calculate loss
                    loss_dict = self.criterion(student_outputs, teacher_outputs, labels)
                    val_losses.append(loss_dict['total_loss'].item())
                    
                    # Predictions
                    predictions = torch.argmax(student_outputs['logits'], dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    logging.error(f"Error in validation: {e}")
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Classification report - determine actual number of classes from data
        unique_labels = sorted(set(all_labels + all_predictions))
        num_unique_classes = len(unique_labels)
        
        # Create target names for actual classes present in data
        target_names = []
        for class_id in unique_labels:
            if class_id < 9:  # Keywords are 0-8
                keyword = list(VietnameseKWSDataset.KEYWORDS.values())[class_id]
                target_names.append(f"Class_{class_id}_{keyword}")
            else:  # Negative class is 9
                target_names.append(f"Class_{class_id}_Negative")
        
        try:
            class_report = classification_report(
                all_labels, 
                all_predictions, 
                labels=unique_labels,  # Specify which labels to include
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            # Log detailed metrics
            logging.info(f"Per-class metrics:")
            for i, class_name in enumerate(target_names):
                if class_name in class_report and isinstance(class_report[class_name], dict):
                    class_metrics = class_report[class_name]
                    precision = class_metrics.get('precision', 0)
                    recall = class_metrics.get('recall', 0) 
                    f1 = class_metrics.get('f1-score', 0)
                    logging.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
                    
        except Exception as e:
            logging.warning(f"Could not generate classification report: {e}")
            class_report = {"accuracy": accuracy}
            target_names = []  # Set empty target_names if classification report fails
        
        # Log to tensorboard
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        # Log per-class metrics to tensorboard
        if isinstance(class_report, dict) and target_names:
            for i, class_name in enumerate(target_names):
                if class_name in class_report and isinstance(class_report[class_name], dict):
                    class_metrics = class_report[class_name]
                    precision = class_metrics.get('precision', 0)
                    recall = class_metrics.get('recall', 0)
                    f1 = class_metrics.get('f1-score', 0)
                    
                    # Extract keyword name for tensorboard
                    keyword = class_name.split('_')[-1] if '_' in class_name else class_name
                    writer.add_scalar(f'Val/Precision_{keyword}', precision, epoch)
                    writer.add_scalar(f'Val/Recall_{keyword}', recall, epoch)
                    writer.add_scalar(f'Val/F1_{keyword}', f1, epoch)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report
        }
        
        return avg_val_loss, metrics
    
    def train(self, num_epochs: int, output_dir: Path, writer: SummaryWriter, logger: logging.Logger):
        """Main training loop."""
        best_val_loss = float('inf')
        best_accuracy = 0.0
        patience_counter = 0  # Initialize patience counter
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch, writer)
            
            # Validate
            val_loss, metrics = self.validate_epoch(epoch, writer)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Accuracy: {metrics['accuracy']:.4f}")
            
            # Save best model
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_path = output_dir / 'checkpoints' / 'best_kws_model.pt'
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': metrics['accuracy'],
                    'config': self.config
                }, best_model_path)
                
                logger.info(f"Saved best model (accuracy: {best_accuracy:.4f}) to {best_model_path}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training'].get('early_stopping_patience', 5):
                logger.info("Early stopping triggered!")
                break
        
        logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Vietnamese KWS Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--student_model', type=str, required=True,
                       choices=['standard', 'tiny'], help='Student model type')
    parser.add_argument('--data_dir', type=str, default='./data/vivos', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs_kws', help='Output directory')
    parser.add_argument('--use_dummy_data', action='store_true', help='Use dummy data for testing')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'
    tensorboard_dir = output_dir / 'tensorboard'
    
    for dir_path in [log_dir, checkpoint_dir, tensorboard_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting Vietnamese KWS training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Student model: {args.student_model}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, train_dataset = create_kws_dataloader(
        data_dir=args.data_dir,
        split="train",
        batch_size=config['training']['batch_size'],
        shuffle=True,
        max_samples=config['data'].get('max_samples_train'),
        use_dummy_data=args.use_dummy_data
    )
    
    val_loader, val_dataset = create_kws_dataloader(
        data_dir=args.data_dir,
        split="test",
        batch_size=config['training']['batch_size'],
        shuffle=False,
        max_samples=config['data'].get('max_samples_val'),
        use_dummy_data=args.use_dummy_data
    )
    
    # Log dataset stats
    train_stats = train_dataset.get_stats()
    val_stats = val_dataset.get_stats()
    
    logger.info(f"Train dataset: {train_stats['total_samples']} samples")
    logger.info(f"Val dataset: {val_stats['total_samples']} samples")
    logger.info(f"Keyword distribution: {train_stats['keyword_counts']}")
    
    # Create models
    logger.info("Creating models...")
    teacher_model = create_kws_teacher(**config['models']['teacher'])
    
    if args.student_model == 'standard':
        student_model = create_kws_student(**config['models']['student'])
    elif args.student_model == 'tiny':
        student_model = create_tiny_kws_student(**config['models']['tiny_student'])
    else:
        raise ValueError(f"Unknown student model: {args.student_model}")
    
    # Log model parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    logger.info(f"Teacher model parameters: {teacher_params:,}")
    logger.info(f"Student model parameters: {student_params:,}")
    logger.info(f"Compression ratio: {teacher_params / student_params:.1f}x")
    
    # Calculate class weights for imbalanced dataset
    keyword_counts = train_stats['keyword_counts']
    total_samples = sum(keyword_counts.values())
    num_classes = len(keyword_counts)
    
    # Calculate inverse frequency weights
    class_weights = torch.zeros(num_classes)
    for class_id, count in keyword_counts.items():
        if count > 0:
            class_weights[class_id] = total_samples / (num_classes * count)
        else:
            # Handle classes with 0 samples by setting high weight
            class_weights[class_id] = 10.0  # Arbitrary high weight
    
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create trainer
    trainer = KWSTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        class_weights=class_weights
    )
    
    # Setup tensorboard
    writer = SummaryWriter(str(tensorboard_dir))
    
    # Start training
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        output_dir=output_dir,
        writer=writer,
        logger=logger
    )
    
    writer.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()