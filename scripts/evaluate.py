#!/usr/bin/env python3
"""
Evaluation script for Vietnamese ASR student models.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import yaml
from transformers import WhisperProcessor
from tqdm import tqdm
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import (
    create_resnet18_asr,
    create_mobilenetv3_asr,
    create_quartznet_asr,
)
from data import create_data_loaders
from evaluation.metrics import ASRMetrics
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Vietnamese ASR student models')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--student_model', type=str, required=True,
                       choices=['resnet18', 'mobilenetv3', 'quartznet'],
                       help='Student model architecture')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--beam_size', type=int, default=5,
                       help='Beam size for decoding')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_student_model(model_name: str, config: dict, device: torch.device):
    """Create student model based on configuration."""
    model_config = config['student_models'][model_name]
    
    if model_name == 'resnet18':
        model = create_resnet18_asr(**model_config)
    elif model_name == 'mobilenetv3':
        model = create_mobilenetv3_asr(**model_config)
    elif model_name == 'quartznet':
        model = create_quartznet_asr(**model_config)
    else:
        raise ValueError(f"Unknown student model: {model_name}")
    
    model.to(device)
    return model


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'student_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_model_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def beam_search_decode(logits: torch.Tensor, processor: WhisperProcessor, beam_size: int = 5) -> list:
    """
    Simple beam search decoding for sequence generation.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        processor: Whisper processor for tokenization
        beam_size: Beam size for search
        
    Returns:
        List of decoded texts
    """
    batch_size, seq_len, vocab_size = logits.shape
    decoded_texts = []
    
    for b in range(batch_size):
        # Simple greedy decoding for now (can be improved to proper beam search)
        sequence_logits = logits[b]  # [seq_len, vocab_size]
        predicted_ids = torch.argmax(sequence_logits, dim=-1)  # [seq_len]
        
        # Convert to list and remove padding
        predicted_ids = predicted_ids.tolist()
        
        # Decode to text
        try:
            text = processor.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            decoded_texts.append(text.strip())
        except Exception as e:
            logging.warning(f"Error decoding sequence: {e}")
            decoded_texts.append("")
    
    return decoded_texts


def evaluate_model(model, data_loader, processor, device, beam_size=5):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained student model
        data_loader: Test data loader
        processor: Whisper processor
        device: Evaluation device
        beam_size: Beam size for decoding
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    metrics = ASRMetrics()
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            input_features = batch['input_features'].to(device)
            
            # Forward pass
            outputs = model(input_features, return_features=False)
            logits = outputs['logits']
            
            # Decode predictions
            predictions = beam_search_decode(logits, processor, beam_size)
            
            # Get references if available
            references = batch.get('text', [])
            
            # Store results
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            # Compute metrics for this batch
            if references:
                for pred, ref in zip(predictions, references):
                    metrics.update(pred, ref)
    
    # Compute final metrics
    results = metrics.compute()
    
    # Add prediction samples
    results['predictions'] = all_predictions[:10]  # First 10 samples
    results['references'] = all_references[:10]
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create student model
    logger.info(f"Creating student model: {args.student_model}")
    student_model = create_student_model(args.student_model, config, device)
    
    # Load model checkpoint
    logger.info(f"Loading model from {args.model_path}")
    student_model = load_model_checkpoint(student_model, args.model_path, device)
    
    # Log model info
    num_params = student_model.get_num_parameters()
    model_size = student_model.get_model_size_mb()
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Create Whisper processor
    whisper_processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
    
    # Create test data loader
    logger.info("Creating test data loader...")
    data_loaders = create_data_loaders(
        train_data_path=args.test_data,  # Use test_data as train_data to create loader
        audio_config=config['data'].get('audio_config', {}),
        whisper_processor=whisper_processor,
        batch_size=args.batch_size,
        num_workers=4,
        max_audio_length=config['data'].get('max_audio_length'),
        max_text_length=config['data'].get('max_text_length'),
    )
    
    test_loader = data_loaders['train']  # Actually the test data
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate_model(
        model=student_model,
        data_loader=test_loader,
        processor=whisper_processor,
        device=device,
        beam_size=args.beam_size,
    )
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"WER: {results.get('wer', 'N/A'):.4f}")
    logger.info(f"CER: {results.get('cer', 'N/A'):.4f}")
    logger.info(f"BLEU: {results.get('bleu', 'N/A'):.4f}")
    
    # Add metadata to results
    results['model_info'] = {
        'model_type': args.student_model,
        'model_path': args.model_path,
        'parameters': num_params,
        'size_mb': model_size,
        'beam_size': args.beam_size,
    }
    
    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()