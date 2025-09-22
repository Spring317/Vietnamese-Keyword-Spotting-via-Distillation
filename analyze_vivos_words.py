#!/usr/bin/env python3
"""
VIVOS Dataset Word Analysis Script
Comprehensive assessment of all words in the VIVOS Vietnamese dataset.
"""

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple

def load_prompts_file(file_path: str) -> List[str]:
    """Load and parse prompts file."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by space, first part is ID, rest is sentence
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    sentences.append(parts[1])
    return sentences

def preprocess_text(text: str) -> List[str]:
    """Clean and tokenize Vietnamese text."""
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove punctuation but keep Vietnamese characters
    text = re.sub(r'[^\w\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Filter out empty strings and very short tokens
    words = [word for word in words if len(word) >= 1]
    
    return words

def analyze_dataset_words(data_dir: str) -> Dict:
    """Comprehensive word analysis of VIVOS dataset."""
    
    analysis = {
        'total_sentences': 0,
        'total_words': 0,
        'unique_words': 0,
        'word_frequencies': Counter(),
        'word_lengths': [],
        'sentence_lengths': [],
        'splits': {}
    }
    
    data_path = Path(data_dir)
    
    # Process each split
    for split in ['train', 'test']:
        split_path = data_path / split / 'prompts.txt'
        
        if split_path.exists():
            print(f"Processing {split} split...")
            sentences = load_prompts_file(str(split_path))
            
            split_info = {
                'sentences': len(sentences),
                'words': 0,
                'word_frequencies': Counter(),
                'avg_sentence_length': 0
            }
            
            for sentence in sentences:
                words = preprocess_text(sentence)
                
                # Update counters
                analysis['word_frequencies'].update(words)
                split_info['word_frequencies'].update(words)
                
                # Track lengths
                analysis['word_lengths'].extend([len(word) for word in words])
                analysis['sentence_lengths'].append(len(words))
                
                split_info['words'] += len(words)
            
            split_info['avg_sentence_length'] = split_info['words'] / split_info['sentences'] if split_info['sentences'] > 0 else 0
            analysis['splits'][split] = split_info
            
            analysis['total_sentences'] += split_info['sentences']
            analysis['total_words'] += split_info['words']
    
    analysis['unique_words'] = len(analysis['word_frequencies'])
    
    return analysis

def create_word_assessment_report(analysis: Dict, output_dir: str):
    """Create comprehensive word assessment report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Basic Statistics Report
    print("=" * 70)
    print("VIVOS DATASET WORD ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total sentences: {analysis['total_sentences']:,}")
    print(f"   Total words: {analysis['total_words']:,}")
    print(f"   Unique words: {analysis['unique_words']:,}")
    print(f"   Vocabulary size: {analysis['unique_words']:,}")
    print(f"   Average words per sentence: {analysis['total_words']/analysis['total_sentences']:.1f}")
    
    # Split breakdown
    print(f"\n📈 SPLIT BREAKDOWN:")
    for split, info in analysis['splits'].items():
        print(f"   {split.capitalize()}:")
        print(f"     Sentences: {info['sentences']:,}")
        print(f"     Words: {info['words']:,}")
        print(f"     Avg sentence length: {info['avg_sentence_length']:.1f}")
        print(f"     Unique words: {len(info['word_frequencies']):,}")
    
    # 2. Most Frequent Words
    print(f"\n🔤 TOP 50 MOST FREQUENT WORDS:")
    most_common = analysis['word_frequencies'].most_common(50)
    for i, (word, count) in enumerate(most_common, 1):
        percentage = (count / analysis['total_words']) * 100
        print(f"   {i:2d}. {word:<15} {count:>6,} ({percentage:5.2f}%)")
    
    # 3. Word Length Analysis
    word_lengths = analysis['word_lengths']
    avg_word_length = sum(word_lengths) / len(word_lengths)
    
    print(f"\n📏 WORD LENGTH ANALYSIS:")
    print(f"   Average word length: {avg_word_length:.2f} characters")
    print(f"   Shortest word: {min(word_lengths)} characters")
    print(f"   Longest word: {max(word_lengths)} characters")
    
    # Word length distribution
    length_dist = Counter(word_lengths)
    print(f"   Word length distribution:")
    for length in sorted(length_dist.keys())[:15]:  # Show first 15 lengths
        count = length_dist[length]
        percentage = (count / len(word_lengths)) * 100
        print(f"     {length:2d} chars: {count:>6,} words ({percentage:5.2f}%)")
    
    # 4. Sentence Length Analysis
    sentence_lengths = analysis['sentence_lengths']
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    print(f"\n📝 SENTENCE LENGTH ANALYSIS:")
    print(f"   Average sentence length: {avg_sentence_length:.1f} words")
    print(f"   Shortest sentence: {min(sentence_lengths)} words")
    print(f"   Longest sentence: {max(sentence_lengths)} words")
    
    # 5. Vocabulary Richness
    total_words = analysis['total_words']
    unique_words = analysis['unique_words']
    type_token_ratio = unique_words / total_words
    
    print(f"\n🎯 VOCABULARY RICHNESS:")
    print(f"   Type-Token Ratio (TTR): {type_token_ratio:.4f}")
    print(f"   Vocabulary diversity: {'High' if type_token_ratio > 0.1 else 'Medium' if type_token_ratio > 0.05 else 'Low'}")
    
    # Words appearing only once
    hapax_legomena = sum(1 for count in analysis['word_frequencies'].values() if count == 1)
    hapax_percentage = (hapax_legomena / unique_words) * 100
    print(f"   Hapax legomena (words appearing once): {hapax_legomena:,} ({hapax_percentage:.1f}%)")
    
    # 6. Frequency Distribution Analysis
    freq_dist = Counter(analysis['word_frequencies'].values())
    print(f"\n📊 FREQUENCY DISTRIBUTION:")
    print(f"   Words appearing 1 time: {freq_dist[1]:,}")
    print(f"   Words appearing 2-5 times: {sum(freq_dist[i] for i in range(2, 6)):,}")
    print(f"   Words appearing 6-10 times: {sum(freq_dist[i] for i in range(6, 11)):,}")
    print(f"   Words appearing 11-50 times: {sum(freq_dist[i] for i in range(11, 51)):,}")
    print(f"   Words appearing 51+ times: {sum(freq_dist[i] for i in range(51, max(freq_dist.keys())+1)):,}")
    
    # 7. Common Vietnamese Words Assessment
    vietnamese_function_words = [
        'và', 'của', 'có', 'được', 'không', 'là', 'với', 'trong', 'để', 'đã',
        'sẽ', 'một', 'những', 'các', 'này', 'đó', 'khi', 'nếu', 'về', 'từ',
        'trên', 'dưới', 'sau', 'trước', 'nhưng', 'hoặc', 'cho', 'bằng', 'vào', 'ra'
    ]
    
    print(f"\n🇻🇳 VIETNAMESE FUNCTION WORDS COVERAGE:")
    function_word_coverage = []
    for word in vietnamese_function_words:
        count = analysis['word_frequencies'].get(word, 0)
        if count > 0:
            percentage = (count / analysis['total_words']) * 100
            function_word_coverage.append((word, count, percentage))
    
    function_word_coverage.sort(key=lambda x: x[1], reverse=True)
    for word, count, percentage in function_word_coverage[:20]:
        print(f"   {word:<10} {count:>6,} ({percentage:5.2f}%)")
    
    # 8. Save detailed word list
    word_list_file = output_path / 'vivos_word_frequencies.txt'
    print(f"\n💾 SAVING DETAILED WORD LIST:")
    print(f"   File: {word_list_file}")
    
    with open(word_list_file, 'w', encoding='utf-8') as f:
        f.write("VIVOS Dataset - Complete Word Frequency List\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Total words: {analysis['total_words']:,}\\n")
        f.write(f"Unique words: {analysis['unique_words']:,}\\n\\n")
        f.write("Rank\\tWord\\tFrequency\\tPercentage\\n")
        f.write("-" * 40 + "\\n")
        
        for i, (word, count) in enumerate(analysis['word_frequencies'].most_common(), 1):
            percentage = (count / analysis['total_words']) * 100
            f.write(f"{i}\\t{word}\\t{count}\\t{percentage:.4f}%\\n")
    
    print(f"   Saved {analysis['unique_words']:,} words to file")
    
    # 9. KWS Relevance Assessment
    target_keywords = ['chào', 'dừng', 'đi', 'mở', 'đóng', 'bật', 'tắt', 'tìm', 'gọi']
    
    print(f"\n🎯 KWS TARGET KEYWORDS ASSESSMENT:")
    print(f"   Checking for target keywords in dataset...")
    
    keyword_stats = []
    for keyword in target_keywords:
        count = analysis['word_frequencies'].get(keyword, 0)
        if count > 0:
            percentage = (count / analysis['total_words']) * 100
            keyword_stats.append((keyword, count, percentage))
            print(f"   ✅ '{keyword}': {count:,} occurrences ({percentage:.4f}%)")
        else:
            print(f"   ❌ '{keyword}': Not found in dataset")
    
    if keyword_stats:
        print(f"\\n   Found {len(keyword_stats)}/{len(target_keywords)} target keywords")
        print(f"   Average keyword frequency: {sum(x[1] for x in keyword_stats)/len(keyword_stats):.1f}")
    else:
        print(f"   ⚠️  WARNING: No target keywords found in dataset!")
        print(f"   This dataset may not be suitable for KWS training.")
    
    print(f"\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE")
    print(f"=" * 70)

def main():
    """Run complete word analysis."""
    data_dir = "/home/quydx/distile_asr_phoWhisper/data/vivos"
    output_dir = "/home/quydx/distile_asr_phoWhisper/word_analysis"
    
    print("Starting VIVOS dataset word analysis...")
    
    # Perform analysis
    analysis = analyze_dataset_words(data_dir)
    
    # Generate report
    create_word_assessment_report(analysis, output_dir)

if __name__ == "__main__":
    main()