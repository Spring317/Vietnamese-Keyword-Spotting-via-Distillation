"""
Data loading and preprocessing for Vietnamese ASR.
Simplified version with dummy data support for testing.
"""

import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing utilities for Vietnamese ASR.
    Handles mel-spectrogram extraction and normalization.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalize: bool = True,
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel-frequency bins
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            f_min: Minimum frequency
            f_max: Maximum frequency (None for sample_rate/2)
            power: Power for mel spectrogram
            normalize: Whether to normalize features
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.power = power
        self.normalize = normalize
        
        # Create mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=power,
        )
        
        # Amplitude to DB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio tensor of shape (1, num_samples)
        """
        try:
            # Load audio using torchaudio
            waveform, orig_sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if orig_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return empty tensor on error
            return torch.zeros(1, self.sample_rate)
    
    def extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features from waveform.
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Mel-spectrogram features of shape (n_mels, time_steps)
        """
        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Remove batch dimension if present
        if mel_spec_db.dim() == 3:
            mel_spec_db = mel_spec_db.squeeze(0)
        
        # Normalize if requested
        if self.normalize:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        Process audio file to mel-spectrogram features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Mel-spectrogram features
        """
        waveform = self.load_audio(audio_path)
        features = self.extract_mel_features(waveform)
        return features


class VietnameseASRDataset(Dataset):
    """
    Dataset for Vietnamese ASR with support for various data formats.
    """
    
    def __init__(
        self,
        data_path: str,
        audio_preprocessor: AudioPreprocessor,
        whisper_processor: Optional[WhisperProcessor] = None,
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
        split: str = 'train',
    ):
        """
        Initialize Vietnamese ASR dataset.
        
        Args:
            data_path: Path to dataset directory or manifest file
            audio_preprocessor: Audio preprocessing pipeline
            whisper_processor: Whisper processor for tokenization
            max_audio_length: Maximum audio length in seconds
            max_text_length: Maximum text length in tokens
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.audio_preprocessor = audio_preprocessor
        self.whisper_processor = whisper_processor
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.split = split
        
        # Load dataset metadata
        self.samples = self._load_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load dataset samples from various formats."""
        samples = []
        
        if os.path.isfile(self.data_path):
            # Load from manifest file (JSON lines or JSON)
            if self.data_path.endswith('.jsonl'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line.strip())
                        samples.append(sample)
            elif self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    elif isinstance(data, dict) and self.split in data:
                        samples = data[self.split]
        elif os.path.isdir(self.data_path):
            # Load from directory structure
            samples = self._load_from_directory()
        
        return samples
    
    def _load_from_directory(self) -> List[Dict[str, str]]:
        """Load samples from directory structure."""
        samples = []
        data_dir = Path(self.data_path)
        
        # Look for common dataset structures
        if (data_dir / 'train').exists():
            # Structure: data_path/{split}/audio and data_path/{split}/transcripts
            split_dir = data_dir / self.split
            audio_dir = split_dir / 'audio'
            transcript_file = split_dir / 'transcripts.txt'
            
            if audio_dir.exists() and transcript_file.exists():
                # Load transcripts
                transcripts = {}
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 2:
                            audio_id, transcript = parts
                            transcripts[audio_id] = transcript
                
                # Match audio files with transcripts
                for audio_file in audio_dir.glob('*.wav'):
                    audio_id = audio_file.stem
                    if audio_id in transcripts:
                        samples.append({
                            'audio_path': str(audio_file),
                            'text': transcripts[audio_id],
                            'audio_id': audio_id
                        })
        else:
            # Flat structure: look for pairs of audio and text files
            for audio_file in data_dir.glob('*.wav'):
                text_file = audio_file.with_suffix('.txt')
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    samples.append({
                        'audio_path': str(audio_file),
                        'text': text,
                        'audio_id': audio_file.stem
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing audio features and labels
        """
        sample = self.samples[idx]
        
        # Load and process audio
        audio_features = self.audio_preprocessor(sample['audio_path'])
        
        # Apply max length filtering
        if self.max_audio_length is not None:
            max_frames = int(self.max_audio_length * self.audio_preprocessor.sample_rate / self.audio_preprocessor.hop_length)
            if audio_features.shape[-1] > max_frames:
                audio_features = audio_features[:, :max_frames]
        
        result = {
            'input_features': audio_features,
            'audio_id': sample.get('audio_id', f'sample_{idx}'),
        }
        
        # Process text if available and processor is provided
        if 'text' in sample and self.whisper_processor is not None:
            # Tokenize text
            text = sample['text']
            
            # Apply max length filtering
            if self.max_text_length is not None:
                tokens = self.whisper_processor.tokenizer.encode(text)
                if len(tokens) > self.max_text_length:
                    tokens = tokens[:self.max_text_length]
                    text = self.whisper_processor.tokenizer.decode(tokens)
            
            # Convert to tensor
            labels = self.whisper_processor.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_text_length or 448,
            )['input_ids'].squeeze(0)
            
            result['labels'] = labels
            result['text'] = text
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched samples with padding
    """
    # Separate keys
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key in ['audio_id', 'text']:
            # Keep as list for string data
            batched[key] = [sample[key] for sample in batch]
        elif key == 'input_features':
            # Pad audio features
            features = [sample[key] for sample in batch]
            
            # Find max length
            max_len = max(feat.shape[-1] for feat in features)
            
            # Pad features
            padded_features = []
            for feat in features:
                if feat.shape[-1] < max_len:
                    padding = torch.zeros(feat.shape[0], max_len - feat.shape[-1])
                    feat = torch.cat([feat, padding], dim=-1)
                padded_features.append(feat)
            
            batched[key] = torch.stack(padded_features)
        elif key == 'labels':
            # Pad labels
            labels = [sample[key] for sample in batch]
            
            # Find max length
            max_len = max(len(label) for label in labels)
            
            # Pad labels with -100 (ignore index)
            padded_labels = []
            for label in labels:
                if len(label) < max_len:
                    padding = torch.full((max_len - len(label),), -100, dtype=label.dtype)
                    label = torch.cat([label, padding])
                padded_labels.append(label)
            
            batched[key] = torch.stack(padded_labels)
    
    return batched


def create_data_loaders(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    audio_config: Optional[Dict] = None,
    whisper_processor: Optional[WhisperProcessor] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    max_audio_length: Optional[int] = None,
    max_text_length: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        test_data_path: Path to test data
        audio_config: Audio preprocessing configuration
        whisper_processor: Whisper processor for tokenization
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_audio_length: Maximum audio length in seconds
        max_text_length: Maximum text length in tokens
        
    Returns:
        Dictionary containing data loaders
    """
    if audio_config is None:
        audio_config = {}
    
    # Create audio preprocessor
    audio_preprocessor = AudioPreprocessor(**audio_config)
    
    # Create datasets
    datasets = {}
    
    # Training dataset
    train_dataset = VietnameseASRDataset(
        data_path=train_data_path,
        audio_preprocessor=audio_preprocessor,
        whisper_processor=whisper_processor,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        split='train',
    )
    datasets['train'] = train_dataset
    
    # Validation dataset
    if val_data_path:
        val_dataset = VietnameseASRDataset(
            data_path=val_data_path,
            audio_preprocessor=audio_preprocessor,
            whisper_processor=whisper_processor,
            max_audio_length=max_audio_length,
            max_text_length=max_text_length,
            split='val',
        )
        datasets['val'] = val_dataset
    
    # Test dataset
    if test_data_path:
        test_dataset = VietnameseASRDataset(
            data_path=test_data_path,
            audio_preprocessor=audio_preprocessor,
            whisper_processor=whisper_processor,
            max_audio_length=max_audio_length,
            max_text_length=max_text_length,
            split='test',
        )
        datasets['test'] = test_dataset
    
    # Create data loaders
    data_loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    config: Optional[Dict] = None,
) -> DataLoader:
    """
    Create a single data loader (wrapper for compatibility).
    
    Args:
        data_path: Path to data
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle data
        config: Configuration dictionary
        
    Returns:
        DataLoader instance
    """
    if config is None:
        config = {}
    
    # Extract relevant config
    audio_config = config.get('data', {}).get('audio_config', {})
    max_audio_length = config.get('data', {}).get('max_audio_length')
    max_text_length = config.get('data', {}).get('max_text_length')
    
    # Create Whisper processor if needed
    whisper_processor = None
    try:
        from transformers import WhisperProcessor
        whisper_processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
    except Exception as e:
        logger.warning(f"Could not load WhisperProcessor: {e}")
    
    # Create audio preprocessor
    audio_preprocessor = AudioPreprocessor(**audio_config)
    
    # Determine split from path
    split = 'train'
    if 'val' in data_path.lower() or 'dev' in data_path.lower():
        split = 'val'
    elif 'test' in data_path.lower():
        split = 'test'
    
    # Create dataset
    dataset = VietnameseASRDataset(
        data_path=data_path,
        audio_preprocessor=audio_preprocessor,
        whisper_processor=whisper_processor,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        split=split,
    )
    
    # Create data loader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


class VIVOSDataset(VietnameseASRDataset):
    """
    Specialized dataset class for VIVOS dataset.
    VIVOS has a specific structure: train/test splits with prompts.txt files.
    """
    
    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load VIVOS dataset samples."""
        samples = []
        data_path = Path(self.data_path)
        
        # VIVOS structure: 
        # VIVOS/
        #   train/
        #     waves/
        #       VIVOSSPK01/
        #         VIVOSDEV01_R001.wav
        #     prompts.txt
        #   test/
        #     waves/
        #       VIVOSSPK01/
        #         VIVOSDEV01_R001.wav  
        #     prompts.txt
        
        # Look for prompts.txt file
        prompts_file = data_path / 'prompts.txt'
        if not prompts_file.exists():
            # Try different locations
            possible_prompts = [
                data_path / self.split / 'prompts.txt',
                data_path / f'{self.split}_prompts.txt',
                data_path / 'transcript.txt',
            ]
            
            for prompt_path in possible_prompts:
                if prompt_path.exists():
                    prompts_file = prompt_path
                    break
            else:
                logger.error(f"Could not find prompts.txt in {data_path}")
                return []
        
        # Load transcripts from prompts.txt
        transcripts = {}
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # VIVOS format: filename transcript
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        filename = parts[0]
                        transcript = parts[1]
                        transcripts[filename] = transcript
        except Exception as e:
            logger.error(f"Error reading prompts file {prompts_file}: {e}")
            return []
        
        logger.info(f"Loaded {len(transcripts)} transcripts from {prompts_file}")
        
        # Find audio files
        waves_dir = data_path / 'waves'
        if not waves_dir.exists():
            # Try different locations
            possible_wave_dirs = [
                data_path / self.split / 'waves',
                data_path / 'audio',
                data_path / self.split / 'audio',
                data_path,  # Audio files directly in data_path
            ]
            
            for wave_dir in possible_wave_dirs:
                if wave_dir.exists():
                    waves_dir = wave_dir
                    break
            else:
                logger.error(f"Could not find waves directory in {data_path}")
                return []
        
        # Recursively find all .wav files
        audio_files = list(waves_dir.rglob('*.wav'))
        logger.info(f"Found {len(audio_files)} audio files in {waves_dir}")
        
        # Match audio files with transcripts
        for audio_file in audio_files:
            # Extract filename without extension
            filename = audio_file.stem
            
            # Look for transcript with exact match or variations
            transcript = None
            
            # Try exact match first
            if filename in transcripts:
                transcript = transcripts[filename]
            else:
                # Try variations (VIVOS sometimes has different naming)
                for transcript_key in transcripts:
                    if transcript_key in filename or filename in transcript_key:
                        transcript = transcripts[transcript_key]
                        break
            
            if transcript:
                samples.append({
                    'audio_path': str(audio_file),
                    'text': transcript,
                    'audio_id': filename
                })
            else:
                logger.warning(f"No transcript found for {filename}")
        
        logger.info(f"Successfully matched {len(samples)} audio-transcript pairs")
        return samples


def create_vivos_dataloader(
    vivos_path: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    config: Optional[Dict] = None,
) -> DataLoader:
    """
    Create data loader specifically for VIVOS dataset.
    
    Args:
        vivos_path: Path to VIVOS dataset root or split directory
        split: Dataset split ('train' or 'test')
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        config: Configuration dictionary
        
    Returns:
        DataLoader for VIVOS dataset
    """
    if config is None:
        config = {}
    
    # Extract relevant config
    audio_config = config.get('data', {}).get('audio_config', {})
    max_audio_length = config.get('data', {}).get('max_audio_length')
    max_text_length = config.get('data', {}).get('max_text_length')
    
    # Create Whisper processor
    whisper_processor = None
    try:
        from transformers import WhisperProcessor
        whisper_processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
    except Exception as e:
        logger.warning(f"Could not load WhisperProcessor: {e}")
    
    # Create audio preprocessor
    audio_preprocessor = AudioPreprocessor(**audio_config)
    
    # Determine data path for split
    vivos_root = Path(vivos_path)
    
    # Check if path already points to split directory
    if (vivos_root / 'prompts.txt').exists() or (vivos_root / 'waves').exists():
        split_path = vivos_root
    else:
        # Assume it's root directory, navigate to split
        split_path = vivos_root / split
        if not split_path.exists():
            raise ValueError(f"Split directory {split_path} does not exist")
    
    # Create VIVOS dataset
    dataset = VIVOSDataset(
        data_path=str(split_path),
        audio_preprocessor=audio_preprocessor,
        whisper_processor=whisper_processor,
        max_audio_length=max_audio_length,
        max_text_length=max_text_length,
        split=split,
    )
    
    # Create data loader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )