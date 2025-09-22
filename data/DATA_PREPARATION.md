# Data Preparation Guide for Vietnamese ASR

This guide explains how to prepare your Vietnamese speech data for the knowledge distillation pipeline.

## Supported Data Formats

The pipeline supports multiple data formats:

### 1. Directory Structure Format
```
data/
├── train/
│   ├── audio/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   └── transcripts.txt
├── val/
│   ├── audio/
│   └── transcripts.txt
└── test/
    ├── audio/
    └── transcripts.txt
```

The `transcripts.txt` file should contain:
```
audio_001	Xin chào, tôi là trợ lý AI.
audio_002	Hôm nay thời tiết rất đẹp.
...
```

### 2. Paired Files Format
```
data/
├── audio_001.wav
├── audio_001.txt
├── audio_002.wav
├── audio_002.txt
└── ...
```

Each `.txt` file contains the transcription for the corresponding audio file.

### 3. JSON Manifest Format
Single JSON file:
```json
{
  "train": [
    {
      "audio_path": "path/to/audio_001.wav",
      "text": "Xin chào, tôi là trợ lý AI.",
      "audio_id": "audio_001"
    },
    ...
  ],
  "val": [...],
  "test": [...]
}
```

Or JSONL format (one JSON object per line):
```
{"audio_path": "path/to/audio_001.wav", "text": "Xin chào, tôi là trợ lý AI.", "audio_id": "audio_001"}
{"audio_path": "path/to/audio_002.wav", "text": "Hôm nay thời tiết rất đẹp.", "audio_id": "audio_002"}
```

## Audio Requirements

- **Format**: WAV files preferred (MP3, FLAC also supported)
- **Sample Rate**: 16kHz (will be resampled automatically)
- **Channels**: Mono preferred (stereo will be converted)
- **Duration**: Recommended 1-30 seconds per utterance
- **Quality**: Clear speech, minimal background noise

## Text Requirements

- **Language**: Vietnamese text
- **Encoding**: UTF-8
- **Format**: Plain text, one utterance per line
- **Content**: Should match the spoken audio exactly
- **Punctuation**: Include natural punctuation

## Example Vietnamese Datasets

### 1. Common Voice Vietnamese
```bash
# Download from Mozilla Common Voice
# https://commonvoice.mozilla.org/vi

# Extract and organize
mkdir -p data/common_voice/{train,val,test}/audio
# Process the TSV files to create the required structure
```

### 2. FOSD (Free Open Speech Dataset)
```bash
# Download Vietnamese subset
# Organize according to the supported formats
```

### 3. Custom Dataset
If you have your own Vietnamese speech data:

1. **Prepare audio files**:
   ```bash
   # Convert to proper format if needed
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

2. **Create transcriptions**:
   - Ensure text matches audio exactly
   - Use proper Vietnamese spelling and diacritics
   - Remove excessive punctuation

3. **Organize data**:
   ```bash
   # Create directory structure
   mkdir -p data/{train,val,test}
   
   # Split your data (e.g., 80/10/10 split)
   # Move files to appropriate directories
   ```

## Data Validation

Use this script to validate your data:

```python
import os
import librosa
from pathlib import Path

def validate_dataset(data_path):
    """Validate dataset format and content."""
    errors = []
    
    for audio_file in Path(data_path).glob("**/*.wav"):
        try:
            # Check audio can be loaded
            y, sr = librosa.load(audio_file, sr=16000)
            
            # Check duration
            duration = len(y) / sr
            if duration < 0.5 or duration > 30:
                errors.append(f"Duration issue: {audio_file} ({duration:.2f}s)")
            
            # Check corresponding text file exists
            text_file = audio_file.with_suffix('.txt')
            if not text_file.exists():
                errors.append(f"Missing text: {text_file}")
            else:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if not text:
                        errors.append(f"Empty text: {text_file}")
                        
        except Exception as e:
            errors.append(f"Error loading {audio_file}: {e}")
    
    return errors

# Run validation
errors = validate_dataset("./data/train")
if errors:
    print("Validation errors found:")
    for error in errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
else:
    print("Dataset validation passed!")
```

## Configuration

Update your config file with the correct paths:

```yaml
data:
  train_path: "./data/train"  # or path to your manifest file
  val_path: "./data/val"      # optional
  test_path: "./data/test"    # optional
  
  max_audio_length: 30        # maximum audio length in seconds
  max_text_length: 448        # maximum text length in tokens
```

## Tips for Better Results

1. **Quality over quantity**: Clean, well-aligned data is better than large noisy datasets
2. **Balanced speakers**: Include diverse speakers (age, gender, dialect)
3. **Varied content**: Mix different domains (news, conversation, etc.)
4. **Consistent format**: Ensure all files follow the same naming and format conventions
5. **Proper splits**: Use proper train/val/test splits with no speaker overlap

## Troubleshooting

### Common Issues:

1. **Audio loading errors**: Check file format and corruption
2. **Encoding issues**: Ensure UTF-8 encoding for text files
3. **Path issues**: Use absolute paths or ensure relative paths are correct
4. **Memory issues**: Reduce batch size or max_audio_length for large files

### Performance Tips:

1. **Preprocessing**: Consider preprocessing audio to mel-spectrograms offline
2. **Caching**: Use SSD storage for faster data loading
3. **Workers**: Adjust num_workers based on your CPU cores
4. **Batch size**: Optimize based on your GPU memory