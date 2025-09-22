#!/usr/bin/env python3
"""
Inference script for Vietnamese ASR student models.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import yaml
from transformers import WhisperProcessor
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import (
    create_resnet18_asr,
    create_mobilenetv3_asr,
    create_quartznet_asr,
)
from data import AudioPreprocessor
from utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with Vietnamese ASR student models')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--student_model', type=str, required=True,
                       choices=['resnet18', 'mobilenetv3', 'quartznet'],
                       help='Student model architecture')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to audio file for inference')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for transcription')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--beam_size', type=int, default=1,
                       help='Beam size for decoding (1 for greedy)')
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


def decode_predictions(logits: torch.Tensor, processor: WhisperProcessor, beam_size: int = 1) -> str:
    """
    Decode model predictions to text.
    
    Args:
        logits: Model output logits [1, seq_len, vocab_size]
        processor: Whisper processor
        beam_size: Beam size for decoding
        
    Returns:
        Decoded text
    """
    # Simple greedy decoding
    predicted_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]
    predicted_ids = predicted_ids.squeeze(0).tolist()  # [seq_len]
    
    # Remove padding and special tokens
    # Filter out common padding/special token IDs
    filtered_ids = []
    for token_id in predicted_ids:
        if token_id not in [0, 1, 2, 50257, -100]:  # Common special tokens
            filtered_ids.append(token_id)
    
    try:
        text = processor.decode(filtered_ids, skip_special_tokens=True)
        return text.strip()
    except Exception as e:
        logging.warning(f"Error decoding sequence: {e}")
        return ""


def run_inference(model, audio_path: str, audio_preprocessor: AudioPreprocessor, 
                 processor: WhisperProcessor, device: torch.device, beam_size: int = 1):
    """
    Run inference on a single audio file.
    
    Args:
        model: Trained student model
        audio_path: Path to audio file
        audio_preprocessor: Audio preprocessing pipeline
        processor: Whisper processor
        device: Inference device
        beam_size: Beam size for decoding
        
    Returns:
        Dictionary containing inference results
    """
    model.eval()
    
    # Preprocess audio
    start_time = time.time()
    audio_features = audio_preprocessor(audio_path)
    preprocessing_time = time.time() - start_time
    
    # Add batch dimension
    audio_features = audio_features.unsqueeze(0).to(device)  # [1, n_mels, time]
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(audio_features, return_features=False)
        logits = outputs['logits']
    inference_time = time.time() - start_time
    
    # Decode predictions
    start_time = time.time()
    transcription = decode_predictions(logits, processor, beam_size)
    decoding_time = time.time() - start_time
    
    total_time = preprocessing_time + inference_time + decoding_time
    
    return {
        'transcription': transcription,
        'timing': {
            'preprocessing': preprocessing_time,
            'inference': inference_time,
            'decoding': decoding_time,
            'total': total_time,
        },
        'audio_info': {
            'path': audio_path,
            'features_shape': list(audio_features.shape),
        }
    }


def main():
    """Main inference function."""
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
    try:
        num_params = student_model.get_num_parameters()
        model_size = student_model.get_model_size_mb()
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model size: {model_size:.2f} MB")
    except:
        logger.info("Could not retrieve model size information")
    
    # Create audio preprocessor
    audio_config = config['data'].get('audio_config', {})
    audio_preprocessor = AudioPreprocessor(**audio_config)
    
    # Create Whisper processor
    whisper_processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
    
    # Check if audio file exists
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Run inference
    logger.info(f"Running inference on {audio_path}")
    results = run_inference(
        model=student_model,
        audio_path=str(audio_path),
        audio_preprocessor=audio_preprocessor,
        processor=whisper_processor,
        device=device,
        beam_size=args.beam_size,
    )
    
    # Display results
    transcription = results['transcription']
    timing = results['timing']
    
    print("\n" + "="*50)
    print("TRANSCRIPTION RESULTS")
    print("="*50)
    print(f"Audio file: {audio_path}")
    print(f"Model: {args.student_model}")
    print(f"Transcription: {transcription}")
    print("\nTiming Information:")
    print(f"  Preprocessing: {timing['preprocessing']:.3f}s")
    print(f"  Inference: {timing['inference']:.3f}s")
    print(f"  Decoding: {timing['decoding']:.3f}s")
    print(f"  Total: {timing['total']:.3f}s")
    print("="*50)
    
    # Save to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logger.info(f"Transcription saved to {output_path}")
    
    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()