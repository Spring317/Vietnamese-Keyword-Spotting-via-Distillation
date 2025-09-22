"""
Loss monitoring and early stopping utilities for knowledge distillation training.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LossMonitor:
    """Monitor training and validation losses to detect overfitting."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        smoothing_window: int = 10,
        validation_frequency: int = 100,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.validation_frequency = validation_frequency
        
        # Loss history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # Overfitting detection
        self.overfitting_threshold = 0.1  # 10% increase in val loss vs train loss
        
    def update(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        step: int = 0,
    ) -> Dict[str, bool]:
        """Update loss history and check for overfitting/early stopping."""
        self.train_losses.append(train_loss)
        
        status = {
            'should_stop': False,
            'overfitting_detected': False,
            'loss_exploded': False,
            'converged': False
        }
        
        # Check for loss explosion
        if train_loss > 100 or np.isnan(train_loss) or np.isinf(train_loss):
            logger.warning(f"Loss explosion detected: {train_loss}")
            status['loss_exploded'] = True
            status['should_stop'] = True
            return status
        
        # Check for too rapid convergence (might indicate overfitting)
        if len(self.train_losses) > 10:
            recent_losses = self.train_losses[-10:]
            if all(loss < 0.001 for loss in recent_losses):
                logger.warning("Very rapid convergence detected - possible overfitting")
                status['overfitting_detected'] = True
        
        # Validation loss monitoring
        if val_loss is not None:
            self.val_losses.append(val_loss)
            
            # Early stopping based on validation loss
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                status['should_stop'] = True
            
            # Overfitting detection
            if len(self.train_losses) >= self.smoothing_window and len(self.val_losses) >= self.smoothing_window:
                smooth_train = np.mean(self.train_losses[-self.smoothing_window:])
                smooth_val = np.mean(self.val_losses[-self.smoothing_window:])
                
                if smooth_val > smooth_train * (1 + self.overfitting_threshold):
                    logger.warning(f"Overfitting detected: val_loss ({smooth_val:.4f}) >> train_loss ({smooth_train:.4f})")
                    status['overfitting_detected'] = True
        
        # Convergence detection
        if len(self.train_losses) >= 50:
            recent_variance = np.var(self.train_losses[-20:])
            if recent_variance < 1e-8:
                logger.info("Loss convergence detected - training may have plateaued")
                status['converged'] = True
        
        return status
    
    def get_recommendations(self) -> List[str]:
        """Get training recommendations based on loss patterns."""
        recommendations = []
        
        if len(self.train_losses) < 10:
            return recommendations
        
        recent_train = np.mean(self.train_losses[-10:])
        
        # Too low loss
        if recent_train < 0.001:
            recommendations.extend([
                "Loss is very low - consider:",
                "1. Reducing learning rate by 10x",
                "2. Increasing temperature to 8.0-10.0", 
                "3. Adding more regularization (dropout, weight decay)",
                "4. Increasing batch size",
                "5. Adding validation monitoring"
            ])
        
        # High variance in loss
        if len(self.train_losses) >= 20:
            recent_variance = np.var(self.train_losses[-20:])
            if recent_variance > 0.1:
                recommendations.extend([
                    "High loss variance detected:",
                    "1. Reduce learning rate",
                    "2. Increase batch size",
                    "3. Add gradient clipping"
                ])
        
        # Validation vs training loss
        if len(self.val_losses) >= 5:
            recent_train = np.mean(self.train_losses[-5:])
            recent_val = np.mean(self.val_losses[-5:])
            
            if recent_val > recent_train * 1.2:
                recommendations.extend([
                    "Possible overfitting:",
                    "1. Increase regularization",
                    "2. Reduce model complexity",
                    "3. Add early stopping",
                    "4. Use data augmentation"
                ])
        
        return recommendations


class AdaptiveLossScaling:
    """Dynamically adjust loss scaling based on training dynamics."""
    
    def __init__(
        self,
        initial_scale: float = 1.0,
        target_loss_range: tuple = (0.01, 1.0),
        adjustment_frequency: int = 100,
    ):
        self.scale = initial_scale
        self.target_min, self.target_max = target_loss_range
        self.adjustment_frequency = adjustment_frequency
        self.step_count = 0
        self.loss_history = []
        
    def update(self, loss: float) -> float:
        """Update scaling factor based on recent loss values."""
        self.loss_history.append(loss)
        self.step_count += 1
        
        if self.step_count % self.adjustment_frequency == 0 and len(self.loss_history) >= 10:
            recent_loss = np.mean(self.loss_history[-10:])
            
            # Adjust scaling
            if recent_loss < self.target_min:
                # Loss too low - increase difficulty
                self.scale *= 1.5
                logger.info(f"Loss too low ({recent_loss:.4f}), increasing scale to {self.scale:.2f}")
            elif recent_loss > self.target_max:
                # Loss too high - decrease difficulty  
                self.scale *= 0.8
                logger.info(f"Loss too high ({recent_loss:.4f}), decreasing scale to {self.scale:.2f}")
            
            # Keep reasonable bounds
            self.scale = max(0.1, min(10.0, self.scale))
            
            # Clear old history
            if len(self.loss_history) > 200:
                self.loss_history = self.loss_history[-100:]
        
        return self.scale


def create_loss_monitor(config: Dict) -> LossMonitor:
    """Create loss monitor from configuration."""
    monitor_config = config.get('monitoring', {})
    
    return LossMonitor(
        patience=monitor_config.get('patience', 5),
        min_delta=monitor_config.get('min_delta', 1e-4),
        smoothing_window=monitor_config.get('smoothing_window', 10),
        validation_frequency=monitor_config.get('validation_frequency', 100),
    )