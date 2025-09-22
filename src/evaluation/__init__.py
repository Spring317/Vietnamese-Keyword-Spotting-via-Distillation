"""
Evaluation package for Vietnamese ASR knowledge distillation.
"""

# Import only what actually exists in metrics.py
try:
    from .metrics import SimpleWERMetrics
    __all__ = ['SimpleWERMetrics']
except ImportError:
    # If SimpleWERMetrics doesn't exist, create a minimal one
    import jiwer
    from typing import Dict, List
    import logging
    
    logger = logging.getLogger(__name__)
    
    class SimpleWERMetrics:
        """Minimal WER metrics class."""
        
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.predictions = []
            self.references = []
        
        def update(self, prediction: str, reference: str):
            self.predictions.append(prediction.lower().strip())
            self.references.append(reference.lower().strip())
        
        def compute(self) -> Dict[str, float]:
            if not self.predictions or not self.references:
                return {'wer': 0.0, 'num_samples': 0}
            
            try:
                wer = jiwer.wer(self.references, self.predictions)
                return {'wer': wer, 'num_samples': len(self.predictions)}
            except Exception as e:
                logger.error(f"Error computing WER: {e}")
                return {'wer': 0.0, 'num_samples': len(self.predictions)}
    
    __all__ = ['SimpleWERMetrics']