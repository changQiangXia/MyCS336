"""Deduplication utilities for text documents."""

import os
import hashlib
from pathlib import Path
from typing import Iterator

import mmh3
from xopen import xopen


def _get_all_lines(input_files: list[os.PathLike]) -> Iterator[tuple[str, os.PathLike]]:
    """Yield all lines from all input files with their source file."""
    for filepath in input_files:
        with xopen(filepath, 'rt') as f:
            for line in f:
                yield line.rstrip('\n\r'), filepath


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """Perform exact line-level deduplication across documents.
    
    Lines that appear in multiple files are removed from ALL files.
    This is useful for removing boilerplate/template content like headers/footers.
    
    Args:
        input_files: List of input file paths.
        output_directory: Directory to write deduplicated output files.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # First pass: count how many files each line appears in
    line_counts = {}
    for filepath in input_files:
        seen_in_this_file = set()
        with xopen(filepath, 'rt') as f:
            for line in f:
                stripped = line.rstrip('\n\r')
                if stripped not in seen_in_this_file:
                    seen_in_this_file.add(stripped)
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # Lines that appear in more than 1 file are considered duplicates
    duplicate_lines = {line for line, count in line_counts.items() if count > 1}
    
    # Second pass: write files without duplicate lines
    for filepath in input_files:
        output_path = output_directory / Path(filepath).name
        
        with xopen(filepath, 'rt') as infile, xopen(output_path, 'wt') as outfile:
            for line in infile:
                stripped = line.rstrip('\n\r')
                # Write line only if it's not a duplicate
                if stripped not in duplicate_lines:
                    outfile.write(line)


def _get_shingles(text: str, n: int) -> set[str]:
    """Get n-gram shingles from text."""
    words = text.split()
    if len(words) < n:
        return set()
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))


def _compute_minhash(shingles: set[str], num_hashes: int, seed: int = 0) -> list[int]:
    """Compute MinHash signature for a set of shingles."""
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            # Use mmh3 for hashing
            hash_val = mmh3.hash(shingle, seed=i + seed)
            min_hash = min(min_hash, hash_val)
        signature.append(min_hash)
    return signature


def _get_band_id(signature: list[int], band_start: int, band_end: int) -> str:
    """Get band ID for LSH."""
    band = signature[band_start:band_end]
    # Create a hash of the band
    band_str = ','.join(map(str, band))
    return hashlib.md5(band_str.encode()).hexdigest()


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """Perform MinHash-based fuzzy deduplication across documents.
    
    Uses Locality Sensitive Hashing (LSH) to efficiently find near-duplicate documents.
    
    Args:
        input_files: List of input file paths.
        num_hashes: Number of hash functions for MinHash.
        num_bands: Number of bands for LSH.
        ngrams: N-gram size for shingles.
        jaccard_threshold: Jaccard similarity threshold for duplicates.
        output_directory: Directory to write deduplicated output files.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Load all documents
    documents = []
    for filepath in input_files:
        with xopen(filepath, 'rt') as f:
            content = f.read()
            documents.append((filepath, content))
    
    # Compute MinHash signatures for all documents
    signatures = []
    for filepath, content in documents:
        shingles = _get_shingles(content, ngrams)
        if shingles:
            signature = _compute_minhash(shingles, num_hashes)
            signatures.append((filepath, content, signature, shingles))
        else:
            # Document too short, keep it
            signatures.append((filepath, content, None, shingles))
    
    # LSH: Build buckets
    rows_per_band = num_hashes // num_bands
    buckets: dict[str, list[int]] = {}  # band_id -> list of doc indices
    
    for doc_idx, (_, _, signature, _) in enumerate(signatures):
        if signature is None:
            continue
        for band_idx in range(num_bands):
            band_start = band_idx * rows_per_band
            band_end = band_start + rows_per_band
            band_id = f"{band_idx}_{_get_band_id(signature, band_start, band_end)}"
            if band_id not in buckets:
                buckets[band_id] = []
            buckets[band_id].append(doc_idx)
    
    # Find candidate pairs and compute actual Jaccard similarity
    num_docs = len(documents)
    is_duplicate = [False] * num_docs
    
    for band_id, doc_indices in buckets.items():
        if len(doc_indices) < 2:
            continue
        # Compare all pairs in this bucket
        for i in range(len(doc_indices)):
            for j in range(i + 1, len(doc_indices)):
                idx1, idx2 = doc_indices[i], doc_indices[j]
                if is_duplicate[idx1] or is_duplicate[idx2]:
                    continue
                
                # Compute Jaccard similarity
                shingles1 = signatures[idx1][3]
                shingles2 = signatures[idx2][3]
                
                if not shingles1 or not shingles2:
                    continue
                
                intersection = len(shingles1 & shingles2)
                union = len(shingles1 | shingles2)
                
                if union > 0:
                    jaccard = intersection / union
                    if jaccard >= jaccard_threshold:
                        # Mark the second document as duplicate
                        is_duplicate[idx2] = True
    
    # Write non-duplicate documents
    for doc_idx, (filepath, content, _, _) in enumerate(signatures):
        if not is_duplicate[doc_idx]:
            output_path = output_directory / Path(filepath).name
            with xopen(output_path, 'wt') as f:
                f.write(content)
