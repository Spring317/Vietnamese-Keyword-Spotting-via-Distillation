# Vietnamese Keyword Spotting (KWS) Configuration
# 10 Common Vietnamese Commands for Voice Control

## Selected Keywords

We've chosen 10 common Vietnamese commands that are:
1. Frequently used in voice interfaces
2. Phonetically distinct to avoid confusion
3. Short enough for reliable spotting
4. Practical for real applications

### Keyword List:

1. **"chào"** (Hello/Hi) - Greeting command
2. **"dừng"** (Stop) - Stop/pause command  
3. **"đi"** (Go) - Go/start command
4. **"lại đây"** (Come here) - Approach command
5. **"mở"** (Open) - Open command
6. **"đóng"** (Close) - Close command
7. **"bật"** (Turn on) - Activate command
8. **"tắt"** (Turn off) - Deactivate command
9. **"tìm"** (Find/Search) - Search command
10. **"gọi"** (Call) - Call/contact command

### Class Mapping:
- Class 0: "chào"
- Class 1: "dừng"  
- Class 2: "đi"
- Class 3: "lại đây"
- Class 4: "mở"
- Class 5: "đóng"
- Class 6: "bật"
- Class 7: "tắt"
- Class 8: "tìm"
- Class 9: "gọi"

### Additional Notes:
- Each keyword will have positive samples (containing the keyword)
- Negative samples will be created from non-keyword speech segments
- Target is binary classification per keyword + negative class
- Much simpler than full ASR - only 11 classes total (10 keywords + 1 negative)

## Dataset Characteristics:
- **Input**: Short audio segments (1-3 seconds)
- **Output**: Classification into one of 11 classes
- **Evaluation**: Accuracy, precision, recall per keyword
- **Real-world usage**: Voice command detection in Vietnamese applications