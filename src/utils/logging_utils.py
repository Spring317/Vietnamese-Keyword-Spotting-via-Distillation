"""
Logging utilities for the ASR distillation pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    debug: bool = False,
    format_string: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        debug: Enable debug mode
        format_string: Custom format string
    """
    if debug:
        log_level = "DEBUG"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Set specific logger levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('torchaudio').setLevel(logging.WARNING)
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


class TensorBoardLogger:
    """Simple wrapper for TensorBoard logging."""
    
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard not available. Install tensorboard to enable logging.")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


class WandbLogger:
    """Simple wrapper for Wandb logging."""
    
    def __init__(self, project_name: str, config: dict = None, name: str = None):
        try:
            import wandb
            wandb.init(project=project_name, config=config, name=name)
            self.enabled = True
        except ImportError:
            logging.warning("Wandb not available. Install wandb to enable logging.")
            self.enabled = False
    
    def log(self, data: dict, step: Optional[int] = None):
        """Log data to wandb."""
        if self.enabled:
            import wandb
            wandb.log(data, step=step)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            import wandb
            wandb.finish()