"""
BPE Tokenizer implementation.
"""
import json
import os
import re
from collections import defaultdict
from typing import BinaryIO, Iterable, Iterator


class BPETokenizer:
    """Byte-Pair Encoding Tokenizer."""
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Build reverse vocab for encoding
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        
        # Build merge lookup
        self.merge_lookup = {}
        for i, (a, b) in enumerate(merges):
            self.merge_lookup[(a, b)] = i
        
        # Special token pattern for pre-tokenization
        if self.special_tokens:
            # Sort by length (longest first) to avoid partial matches
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(s) for s in sorted_specials]
            self.special_pattern = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self.special_pattern = None
    
    def _get_pair_counts(self, word: list[bytes]) -> dict[tuple[bytes, bytes], int]:
        """Get counts of all adjacent pairs in a word."""
        counts = defaultdict(int)
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += 1
        return counts
    
    def _merge_word(self, word: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        """Merge all occurrences of a pair in a word."""
        a, b = pair
        merged = a + b
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        import regex as re
        # GPT-2 pretokenization pattern - splits text into "words"
        PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Split by special tokens
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]
        
        token_ids = []
        
        for part in parts:
            if not part:
                continue
            
            # Check if this is a special token
            if part in self.special_tokens:
                token_ids.append(self.token_to_id[part.encode('utf-8')])
                continue
            
            # Pretokenization: split into words using GPT-2 regex
            words = PAT.findall(part)
            
            for word_str in words:
                # Convert word to bytes
                word_bytes = word_str.encode('utf-8')
                
                # Start with individual bytes
                word = [bytes([b]) for b in word_bytes]
                
                # Apply merges (only within this word, not across words)
                while len(word) > 1:
                    # Find the pair with the lowest merge rank
                    min_rank = float('inf')
                    min_pair = None
                    
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i + 1])
                        if pair in self.merge_lookup:
                            rank = self.merge_lookup[pair]
                            if rank < min_rank:
                                min_rank = rank
                                min_pair = pair
                    
                    if min_pair is None:
                        break
                    
                    word = self._merge_word(word, min_pair)
                
                # Convert tokens to IDs
                for token in word:
                    if token in self.token_to_id:
                        token_ids.append(self.token_to_id[token])
                    else:
                        # Unknown token - shouldn't happen with proper vocab
                        raise ValueError(f"Unknown token: {token}")
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of text to token IDs."""
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.vocab[idx] for idx in token_ids]
        text_bytes = b''.join(tokens)
        return text_bytes.decode('utf-8', errors='replace')
    
    def save(self, path: str | os.PathLike) -> None:
        """Save tokenizer to a directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save vocab as JSON (convert bytes to string representation)
        vocab_str = {str(k): v.decode('utf-8', errors='replace') for k, v in self.vocab.items()}
        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(vocab_str, f, ensure_ascii=False)
        
        # Save merges
        with open(os.path.join(path, 'merges.txt'), 'w', encoding='utf-8') as f:
            for a, b in self.merges:
                f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")
        
        # Save special tokens
        with open(os.path.join(path, 'special_tokens.txt'), 'w', encoding='utf-8') as f:
            for token in self.special_tokens:
                f.write(token + '\n')
    
    @classmethod
    def load(cls, path: str | os.PathLike) -> 'BPETokenizer':
        """Load tokenizer from a directory."""
        # Load vocab
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            vocab_str = json.load(f)
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_str.items()}
        
        # Load merges
        merges = []
        with open(os.path.join(path, 'merges.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        a, b = parts
                        merges.append((a.encode('utf-8'), b.encode('utf-8')))
        
        # Load special tokens
        special_tokens_path = os.path.join(path, 'special_tokens.txt')
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'r', encoding='utf-8') as f:
                special_tokens = [line.strip() for line in f if line.strip()]
        else:
            special_tokens = []
        
        return cls(vocab, merges, special_tokens)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """Create a BPE tokenizer from vocab, merges, and special tokens."""
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.
    
    Returns the vocabulary and merges.
    """
    # GPT-2 pre-tokenization regex pattern
    # This pattern splits on whitespace and punctuation while keeping them as tokens
    import regex as re
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Build initial vocabulary with all bytes
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens to vocab
    for i, special in enumerate(special_tokens):
        vocab[256 + i] = special.encode('utf-8')
    
    # Pre-tokenize: split by special tokens first, then apply regex
    special_bytes = [token.encode('utf-8') for token in special_tokens]
    
    # Collect word frequencies
    word_freqs = defaultdict(int)
    
    # First, split text by special tokens
    # Sort special tokens by length (longest first) to handle overlapping tokens
    sorted_specials = sorted(special_bytes, key=len, reverse=True)
    
    # Build regex pattern for special tokens (use original strings)
    if special_tokens:
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_pattern = '|'.join(re.escape(s) for s in sorted_special_tokens)
        parts = re.split(f'({special_pattern})', text)
    else:
        parts = [text]
    
    for part in parts:
        if not part:
            continue
        part_bytes = part.encode('utf-8')
        # Check if it's a special token
        if part_bytes in special_bytes:
            word_freqs[tuple([part_bytes])] += 1
        else:
            # Apply regex pre-tokenization
            pre_tokenized = PAT.findall(part)
            for token in pre_tokenized:
                token_bytes = token.encode('utf-8')
                # Split into individual bytes
                word = tuple(bytes([b]) for b in token_bytes)
                word_freqs[word] += 1
    
    # BPE training - high performance with full incremental updates
    merges = []
    target_merges = vocab_size - 256 - len(special_tokens)
    
    # Build four data structures
    # 1. words: list of word token lists (bytes objects)
    words = [list(word) for word in word_freqs.keys()]
    # 2. word_counts: parallel list of frequencies
    word_counts = list(word_freqs.values())
    # 3. pair_counts: pair -> total frequency
    pair_counts = {}
    # 4. pair_to_words: pair -> set of word indices
    pair_to_words = {}
    
    for word_idx, (word_tokens, freq) in enumerate(zip(words, word_counts)):
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word_idx)
    
    for _ in range(target_merges):
        if not pair_counts:
            break
        
        # Find best pair
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        
        if pair_counts[best_pair] < 1:
            break
        
        bp0, bp1 = best_pair
        new_token = bp0 + bp1
        merges.append(best_pair)
        vocab[len(vocab)] = new_token
        
        # Get affected word indices
        affected_indices = list(pair_to_words.get(best_pair, set()))
        
        # Process each affected word
        for word_idx in affected_indices:
            old_word = words[word_idx]
            freq = word_counts[word_idx]
            
            # Remove old pairs from this word
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(word_idx)
            
            # Build new word with merge applied
            new_word = []
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and old_word[i] == bp0 and old_word[i + 1] == bp1:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            
            words[word_idx] = new_word
            
            # Add new pairs from this word
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_idx)
        
        # Clean up merged pair
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
    
    return vocab, merges
