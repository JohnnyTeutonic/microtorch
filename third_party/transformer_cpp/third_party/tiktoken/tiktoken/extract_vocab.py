#!/usr/bin/env python3
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path

def extract_from_cl100k():
    """Extract vocabulary and merges from cl100k_base encoding."""
    try:
        import tiktoken
        # Get the cl100k_base encoding
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Extract vocabulary
        vocab = {}
        for token, token_id in enc._mergeable_ranks.items():
            # Convert bytes to string representation
            token_str = token.decode('utf-8', errors='replace')
            vocab[token_str] = token_id
        
        # Extract merge rules
        merges = []
        seen = set()
        for token_bytes, _ in enc._mergeable_ranks.items():
            if len(token_bytes) > 1:
                # This is a merged token, find its components
                components = []
                for byte in token_bytes:
                    components.append(bytes([byte]).decode('utf-8', errors='replace'))
                for i in range(len(components) - 1):
                    merge_pair = (components[i], components[i + 1])
                    if merge_pair not in seen:
                        merges.append(merge_pair)
                        seen.add(merge_pair)
        
        return vocab, merges
    except ImportError:
        print("tiktoken package not found. Please install with: pip install tiktoken")
        return None, None

def read_vocabulary_pairs(file_path: str) -> List[Tuple[str, int]]:
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    token, id_str = line.split('\t')
                    # Remove quotes if present
                    token = token.strip('"')
                    pairs.append((token, int(id_str)))
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
    return pairs

def extract_merges(pairs: List[Tuple[str, str]], min_frequency: int = 2) -> List[Tuple[str, str]]:
    """Extract merge rules from vocabulary pairs with sophisticated pattern recognition.
    
    Args:
        pairs: List of (token, id) tuples
        min_frequency: Minimum frequency for a merge pair to be considered
    
    Returns:
        List of (first, second) merge pairs ordered by priority
    """
    # Count frequencies of character pairs and subwords
    pair_frequencies = {}
    subword_patterns = {}
    prefix_patterns = {}
    suffix_patterns = {}
    
    # First pass: collect statistics
    for token, _ in pairs:
        if len(token) <= 1:
            continue
            
        # Skip special tokens
        if token.startswith('<|') and token.endswith('|>'):
            continue
            
        # Convert token to character list
        chars = list(token)
        
        # Count character pair frequencies
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1
            
        # Identify common subwords (length 2-4)
        for length in range(2, min(5, len(token) + 1)):
            for i in range(len(token) - length + 1):
                subword = token[i:i + length]
                subword_patterns[subword] = subword_patterns.get(subword, 0) + 1
                
        # Track prefixes and suffixes
        for length in range(1, min(4, len(token))):
            prefix = token[:length]
            suffix = token[-length:]
            prefix_patterns[prefix] = prefix_patterns.get(prefix, 0) + 1
            suffix_patterns[suffix] = suffix_patterns.get(suffix, 0) + 1

    # Filter and sort patterns
    merges = []
    seen = set()
    
    # Helper function to add merge rule
    def add_merge_rule(first: str, second: str, priority: int):
        pair = (first, second)
        if pair not in seen:
            merges.append((pair, priority))
            seen.add(pair)
    
    # Add high-frequency character pairs
    for (first, second), freq in pair_frequencies.items():
        if freq >= min_frequency:
            add_merge_rule(first, second, freq)
    
    # Add common prefixes with space
    for prefix, freq in prefix_patterns.items():
        if freq >= min_frequency and len(prefix) > 1:
            add_merge_rule('Ġ', prefix, freq * 2)  # Higher priority for space + prefix
    
    # Add common subwords
    for subword, freq in subword_patterns.items():
        if freq >= min_frequency and len(subword) > 1:
            # Split subword into first and rest
            add_merge_rule(subword[0], subword[1:], freq)
    
    # Add special cases for common English patterns
    common_patterns = [
        ('t', 'h'),  # 'th'
        ('i', 'n'),  # 'in'
        ('e', 'r'),  # 'er'
        ('Ġ', 't'),  # space + 't'
        ('Ġ', 'a'),  # space + 'a'
        ('Ġ', 'i'),  # space + 'i'
        ('Ġ', 's'),  # space + 's'
        ('i', 'ng'),  # 'ing'
        ('e', 'd'),   # 'ed'
    ]
    
    for first, second in common_patterns:
        add_merge_rule(first, second, 1000)  # High priority for common patterns
    
    # Sort merges by priority (frequency) in descending order
    sorted_merges = sorted(merges, key=lambda x: x[1], reverse=True)
    
    # Return only the merge pairs, without priorities
    return [pair for pair, _ in sorted_merges]

def create_vocab_file(vocab_dict: Dict[str, int], output_path: str):
    """Create the vocabulary JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=4, ensure_ascii=False)
        
def create_merges_file(merges: List[Tuple[str, str]], output_path: str):
    """Create the merges file with metadata and priority information."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('#version: 0.2\n')
        f.write('#format: <first> <second> [priority]\n')
        
        # Write special merges first
        f.write('# Special merges\n')
        special_tokens = [
            ('Ġ', 't'),
            ('Ġ', 'a'),
            ('Ġ', 'i'),
            ('Ġ', 'n'),
            ('Ġ', 's'),
        ]
        for first, second in special_tokens:
            f.write(f'{first} {second}\n')
            
        # Write common English patterns
        f.write('\n# Common English patterns\n')
        for first, second in merges[:100]:  # Top 100 most frequent patterns
            f.write(f'{first} {second}\n')
            
        # Write remaining merges
        f.write('\n# Additional merges\n')
        for first, second in merges[100:]:
            f.write(f'{first} {second}\n')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract vocabulary and merge rules')
    parser.add_argument('--encoding', choices=['cl100k', 'custom'], default='cl100k',
                      help='Encoding type to use (cl100k or custom from vocabulary_pairs.txt)')
    parser.add_argument('--output-dir', default='tiktoken_data',
                      help='Output directory for vocabulary and merges files')
    parser.add_argument('--min-frequency', type=int, default=2,
                      help='Minimum frequency for a merge pair to be considered')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.encoding == 'cl100k':
        vocab, merges = extract_from_cl100k()
        if vocab is None:
            return
        
        # Create files with cl100k prefix
        create_vocab_file(vocab, output_dir / 'cl100k_base.vocab')
        create_merges_file(merges, output_dir / 'cl100k_base.merges')
        print(f"Created cl100k vocabulary file with {len(vocab)} tokens")
        print(f"Created cl100k merges file with {len(merges)} merge rules")
    
    else:  # custom encoding from vocabulary_pairs.txt
        # Read vocabulary pairs
        pairs = read_vocabulary_pairs('vocabulary_pairs.txt')
        if not pairs:
            print("No vocabulary pairs found. Please create vocabulary_pairs.txt")
            return
            
        # Extract merges with specified minimum frequency
        merges = extract_merges(pairs, min_frequency=args.min_frequency)
        
        # Convert pairs to dictionary
        vocab_dict = {token: id for token, id in pairs}
        
        # Create files
        create_vocab_file(vocab_dict, output_dir / 'custom.vocab')
        create_merges_file(merges, output_dir / 'custom.merges')
        print(f"Created custom vocabulary file with {len(pairs)} tokens")
        print(f"Created custom merges file with {len(merges)} merge rules")

if __name__ == '__main__':
    main()