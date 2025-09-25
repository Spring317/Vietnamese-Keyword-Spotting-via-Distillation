"""
Vietnamese Keyword Spotting (KWS) Dataset
Converts VIVOS ASR data into keyword spotting task with 10 Vietnamese commands.
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore')


class VietnameseKWSDataset(Dataset):
    """
    Vietnamese Keyword Spotting Dataset
    
    Converts VIVOS transcripts into keyword detection:
    - 9 target keywords (classes 0-8)  
    - 1 negative class (class 9) for non-keyword speech
    """
    
    # Define our 9 Vietnamese keywords (most frequent words from VIVOS dataset)
    KEYWORDS = {
        0: "có",        # Have/exist (2,243 occurrences - 1.38%)
        1: "là",        # Is/be (1,854 occurrences - 1.14%)
        2: "không",     # No/not (1,838 occurrences - 1.13%)
        3: "một",       # One/a (1,777 occurrences - 1.10%)
        4: "của",       # Of/belonging to (1,698 occurrences - 1.05%)
        5: "và",        # And (1,583 occurrences - 0.98%)
        6: "người",     # Person/people (1,393 occurrences - 0.86%)
        7: "những",     # The/those (1,366 occurrences - 0.84%)
        8: "tôi"        # I/me (1,291 occurrences - 0.80%)
    }
    
    NEGATIVE_CLASS = 9   # For non-keyword speech
    NUM_CLASSES = 10     # 9 keywords + 1 negative
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_samples: Optional[int] = None,
        segment_length: float = 2.0,  # 2 second segments
        negative_ratio: float = 2.0,  # 2x negative samples per positive
        use_dummy_data: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor or WhisperProcessor.from_pretrained("vinai/PhoWhisper-base")
        self.max_samples = max_samples
        self.segment_length = segment_length
        self.negative_ratio = negative_ratio
        self.use_dummy_data = use_dummy_data
        
        # Initialize data
        self.samples = []
        self.keyword_counts = {i: 0 for i in range(self.NUM_CLASSES)}
        
        if use_dummy_data:
            self._create_dummy_data()
        else:
            self._load_vivos_data()
            
        logger.info(f"Loaded {len(self.samples)} KWS samples for {split}")
        logger.info(f"Keyword distribution: {self.keyword_counts}")
    
    def _load_vivos_data(self):
        """Load and process VIVOS data for keyword spotting."""
        split_dir = self.data_dir / self.split
        prompts_file = split_dir / "prompts.txt"
        waves_dir = split_dir / "waves"
        
        if not prompts_file.exists() or not waves_dir.exists():
            logger.error(f"VIVOS data not found in {split_dir}")
            logger.info("Creating dummy data instead...")
            self._create_dummy_data()
            return
        
        logger.info(f"Loading VIVOS data from {split_dir}")
        
        # Load transcripts
        transcripts = {}
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ' ' in line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        transcripts[file_id] = transcript.lower()
        
        logger.info(f"Loaded {len(transcripts)} transcripts")
        
        # Process each transcript to find keywords
        positive_samples = []
        negative_candidates = []
        
        for file_id, transcript in transcripts.items():
            audio_path = self._find_audio_file(file_id, waves_dir)
            
            if not audio_path:
                continue
                
            # Check for keywords in transcript
            found_keywords = self._find_keywords_in_transcript(transcript)
            
            if found_keywords:
                # Create positive samples for each keyword found
                for keyword_class in found_keywords:
                    positive_samples.append({
                        'audio_path': str(audio_path),
                        'transcript': transcript,
                        'keyword_class': keyword_class,
                        'keyword': self.KEYWORDS[keyword_class],
                        'label': keyword_class,
                        'file_id': file_id
                    })
                    self.keyword_counts[keyword_class] += 1
            else:
                # Candidate for negative samples
                negative_candidates.append({
                    'audio_path': str(audio_path),
                    'transcript': transcript,
                    'keyword_class': None,
                    'keyword': None,
                    'label': self.NEGATIVE_CLASS,
                    'file_id': file_id
                })
        
        logger.info(f"Found {len(positive_samples)} positive samples")
        logger.info(f"Found {len(negative_candidates)} negative candidates")
        
        # Create balanced dataset
        self.samples.extend(positive_samples)
        
        # Add negative samples
        num_negatives = min(
            len(negative_candidates),
            int(len(positive_samples) * self.negative_ratio)
        )
        
        if num_negatives > 0:
            selected_negatives = random.sample(negative_candidates, num_negatives)
            self.samples.extend(selected_negatives)
            self.keyword_counts[self.NEGATIVE_CLASS] = num_negatives
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        # Limit samples if requested
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            # Recount after limiting
            self.keyword_counts = {i: 0 for i in range(self.NUM_CLASSES)}
            for sample in self.samples:
                self.keyword_counts[sample['label']] += 1
        
        # Log final statistics
        logger.info(f"Final dataset: {len(self.samples)} samples")
        for class_id, count in self.keyword_counts.items():
            if class_id < 9:  # Keywords are 0-8, negative class is 9
                keyword = self.KEYWORDS[class_id]
                logger.info(f"  Class {class_id} ('{keyword}'): {count} samples")
            else:
                logger.info(f"  Class {class_id} (negative): {count} samples")
    
    def _find_audio_file(self, file_id: str, waves_dir: Path) -> Optional[Path]:
        """Find the audio file for a given file_id in VIVOS structure."""
        # VIVOS structure: waves/SPEAKER_DIR/file_id.wav
        for speaker_dir in waves_dir.glob("*"):
            if speaker_dir.is_dir():
                audio_file = speaker_dir / f"{file_id}.wav"
                if audio_file.exists():
                    return audio_file
        return None
    
    def _find_keywords_in_transcript(self, transcript: str) -> Set[int]:
        """Find keywords in transcript and return their class IDs."""
        found = set()
        
        # Normalize transcript
        transcript = transcript.lower().strip()
        
        # Check each keyword
        for class_id, keyword in self.KEYWORDS.items():
            # Use word boundary matching to avoid partial matches
            # For Vietnamese, we'll use simpler matching due to word boundary complexity
            if self._contains_keyword(transcript, keyword):
                found.add(class_id)
        
        return found
    
    def _contains_keyword(self, transcript: str, keyword: str) -> bool:
        """Check if transcript contains the keyword with proper word boundaries."""
        # For Vietnamese, we'll use space-separated matching
        words = transcript.split()
        keyword_words = keyword.split()
        
        if len(keyword_words) == 1:
            # Single word keyword
            return keyword in words
        else:
            # Multi-word keyword (like "lại đây")
            keyword_phrase = " ".join(keyword_words)
            return keyword_phrase in transcript
    
    def _create_dummy_data(self):
        """Create dummy data for testing."""
        logger.info("Creating dummy KWS data")
        
        # Create some dummy samples for each class
        samples_per_class = 50 if not self.max_samples else min(50, self.max_samples // self.NUM_CLASSES)
        
        # Example Vietnamese sentences with keywords (using most frequent words)
        keyword_templates = {
            0: ["tôi có", "bạn có", "anh có", "chị có"],
            1: ["đó là", "đây là", "anh là", "tôi là"],
            2: ["không có", "không được", "không phải", "không thể"],
            3: ["một người", "một cái", "một lần", "một chút"],
            4: ["của tôi", "của bạn", "của anh", "của chị"],
            5: ["tôi và", "anh và", "bạn và", "chúng tôi và"],
            6: ["người ta", "người này", "người nào", "người đó"],
            7: ["những người", "những cái", "những gì", "những lúc"],
            8: ["tôi muốn", "tôi nghĩ", "tôi biết", "tôi thấy"]
        }
        
        negative_templates = [
            "hôm nay thời tiết đẹp",
            "tôi rất vui được gặp bạn",
            "cảm ơn bạn rất nhiều",
            "việt nam là đất nước xinh đẹp",
            "học tập là điều quan trọng"
        ]
        
        for class_id in range(self.NUM_CLASSES):
            for i in range(samples_per_class):
                if class_id < 9:  # Keywords are 0-8, negative class is 9
                    # Positive samples
                    templates = keyword_templates[class_id]
                    transcript = random.choice(templates)
                    keyword = self.KEYWORDS[class_id]
                else:
                    # Negative samples
                    transcript = random.choice(negative_templates)
                    keyword = None
                
                self.samples.append({
                    'audio_path': None,  # Will create dummy audio in __getitem__
                    'transcript': transcript,
                    'keyword_class': class_id if class_id < 9 else None,
                    'keyword': keyword,
                    'label': class_id,
                    'file_id': f'dummy_{class_id}_{i:03d}'
                })
                
                self.keyword_counts[class_id] += 1
        
        # Shuffle
        random.shuffle(self.samples)
        
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            # Recount
            self.keyword_counts = {i: 0 for i in range(self.NUM_CLASSES)}
            for sample in self.samples:
                self.keyword_counts[sample['label']] += 1
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load audio
        if sample['audio_path'] and Path(sample['audio_path']).exists():
            try:
                # Load audio file
                waveform, sample_rate = torchaudio.load(sample['audio_path'])
                
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Segment audio to fixed length
                target_length = int(self.segment_length * 16000)  # 2 seconds at 16kHz
                
                if waveform.shape[1] > target_length:
                    # Random crop for training, center crop for validation
                    if self.split == "train":
                        start = random.randint(0, waveform.shape[1] - target_length)
                    else:
                        start = (waveform.shape[1] - target_length) // 2
                    waveform = waveform[:, start:start + target_length]
                elif waveform.shape[1] < target_length:
                    # Pad with zeros
                    padding = target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                
                # Convert to numpy for processor
                audio = waveform.squeeze().numpy()
                
            except Exception as e:
                logger.error(f"Error loading audio {sample['audio_path']}: {e}")
                # Fall back to dummy audio
                audio = self._create_dummy_audio()
        else:
            # Create dummy audio
            audio = self._create_dummy_audio()
        
        # Process audio to mel spectrogram
        try:
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features = inputs.input_features.squeeze(0)  # Remove batch dimension
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            # Create dummy features
            input_features = torch.randn(80, 3000)  # Standard whisper input shape
        
        return {
            'input_features': input_features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'transcript': sample['transcript'],
            'keyword': sample['keyword'] or "",
            'audio_path': sample['audio_path'] or "",
            'file_id': sample['file_id']
        }
    
    def _create_dummy_audio(self) -> torch.Tensor:
        """Create dummy audio for testing."""
        # Create 2 seconds of dummy audio at 16kHz
        duration = self.segment_length
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        # Create some random noise with a simple tone
        t = torch.linspace(0, duration, num_samples)
        frequency = random.uniform(200, 800)  # Random frequency
        audio = 0.1 * torch.sin(2 * torch.pi * frequency * t)
        audio += 0.05 * torch.randn(num_samples)  # Add noise
        
        return audio.numpy()
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        total_samples = sum(self.keyword_counts.values())
        weights = []
        
        for class_id in range(self.NUM_CLASSES):
            count = self.keyword_counts[class_id]
            if count > 0:
                weight = total_samples / (self.NUM_CLASSES * count)
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_samples': len(self.samples),
            'num_classes': self.NUM_CLASSES,
            'keyword_counts': self.keyword_counts,
            'keywords': self.KEYWORDS,
            'segment_length': self.segment_length
        }
    
    def analyze_vivos_keywords(self) -> Dict:
        """Analyze how many keywords are actually found in VIVOS dataset."""
        if self.use_dummy_data:
            return {"message": "Using dummy data, no VIVOS analysis available"}
        
        split_dir = self.data_dir / self.split
        prompts_file = split_dir / "prompts.txt"
        
        if not prompts_file.exists():
            return {"error": "VIVOS prompts file not found"}
        
        # Load all transcripts
        transcripts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ' ' in line:
                    _, transcript = line.split(' ', 1)
                    transcripts.append(transcript.lower())
        
        # Analyze keyword occurrences
        keyword_stats = {}
        for class_id, keyword in self.KEYWORDS.items():
            count = 0
            for transcript in transcripts:
                if self._contains_keyword(transcript, keyword):
                    count += 1
            keyword_stats[keyword] = count
        
        total_transcripts = len(transcripts)
        return {
            'total_transcripts': total_transcripts,
            'keyword_occurrences': keyword_stats,
            'coverage_percentage': {
                keyword: (count / total_transcripts) * 100 
                for keyword, count in keyword_stats.items()
            }
        }


def create_kws_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    use_dummy_data: bool = False
):
    """Create a DataLoader for Vietnamese KWS dataset."""
    
    dataset = VietnameseKWSDataset(
        data_dir=data_dir,
        split=split,
        max_samples=max_samples,
        use_dummy_data=use_dummy_data
    )
    
    def collate_fn(batch):
        """Custom collate function for KWS batches."""
        input_features = torch.stack([item['input_features'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'input_features': input_features,
            'labels': labels,
            'transcripts': [item['transcript'] for item in batch],
            'keywords': [item['keyword'] for item in batch],
            'audio_paths': [item['audio_path'] for item in batch],
            'file_ids': [item['file_id'] for item in batch]
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ), dataset


# Utility function to analyze VIVOS for keywords
def analyze_vivos_for_keywords(vivos_path: str, split: str = "train") -> Dict:
    """Analyze VIVOS dataset to see keyword distribution."""
    dataset = VietnameseKWSDataset(
        data_dir=vivos_path,
        split=split,
        use_dummy_data=False,
        max_samples=None  # Analyze all data
    )
    
    return dataset.analyze_vivos_keywords()


if __name__ == "__main__":
    # Quick test and analysis
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze VIVOS for Vietnamese KWS')
    parser.add_argument('--vivos_path', type=str, default='./data/vivos', 
                       help='Path to VIVOS dataset')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'test'], help='Dataset split')
    
    args = parser.parse_args()
    
    print("🔍 Analyzing VIVOS dataset for Vietnamese keywords...")
    print("=" * 60)
    
    # Analyze keyword coverage
    stats = analyze_vivos_for_keywords(args.vivos_path, args.split)
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
    elif 'message' in stats:
        print(f"ℹ️  {stats['message']}")
    else:
        print(f"📊 Total transcripts in {args.split} split: {stats['total_transcripts']}")
        print("\n🎯 Keyword occurrences:")
        
        for keyword, count in stats['keyword_occurrences'].items():
            percentage = stats['coverage_percentage'][keyword]
            print(f"  '{keyword}': {count} times ({percentage:.2f}%)")
        
        total_positive = sum(stats['keyword_occurrences'].values())
        total_transcripts = stats['total_transcripts']
        print(f"\n📈 Total positive samples: {total_positive}")
        print(f"📈 Potential negative samples: {total_transcripts - total_positive}")
        print(f"📈 Positive ratio: {(total_positive / total_transcripts) * 100:.2f}%")