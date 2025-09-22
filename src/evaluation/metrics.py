"""
Simple WER-only metrics for Vietnamese ASR evaluation.
"""

import torch
import jiwer
from typing import List, Dict
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


class SimpleWERMetrics:
    """
    Simple WER (Word Error Rate) metrics for Vietnamese ASR.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.predictions = []
        self.references = []
    
    def normalize_text(self, text: str) -> str:
        """
        Simple text normalization for Vietnamese.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def update(self, prediction: str, reference: str):
        """
        Update metrics with a single prediction-reference pair.
        
        Args:
            prediction: Predicted text
            reference: Reference (ground truth) text
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        self.predictions.append(pred_norm)
        self.references.append(ref_norm)
    
    def update_batch(self, predictions: List[str], references: List[str]):
        """
        Update metrics with a batch of predictions and references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
        """
        for pred, ref in zip(predictions, references):
            self.update(pred, ref)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute WER metric.
        
        Returns:
            Dictionary containing WER and number of samples
        """
        if not self.predictions or not self.references:
            return {'wer': 0.0, 'num_samples': 0}
        
        try:
            # Compute WER using jiwer
            wer = jiwer.wer(self.references, self.predictions)
            
            return {
                'wer': wer,
                'num_samples': len(self.predictions)
            }
            
        except Exception as e:
            logger.error(f"Error computing WER: {e}")
            return {'wer': 0.0, 'num_samples': len(self.predictions)}
    
    def get_wer(self) -> float:
        """
        Get current WER value.
        
        Returns:
            WER as float
        """
        metrics = self.compute()
        return metrics['wer']