"""Text quality classification utilities."""

import re
from typing import Any


def classify_quality(text: str) -> tuple[Any, float]:
    """Classify text quality as wiki (high) or cc (low).
    
    Args:
        text: Input text to classify.
        
    Returns:
        Tuple of (classification, score).
        classification is 'wiki' for high quality or 'cc' for low quality.
    """
    # Simple heuristics based on text characteristics
    lines = text.split('\n')
    words = text.split()
    
    if len(words) == 0:
        return 'cc', 0.99
    
    # Calculate features
    avg_word_len = sum(len(w) for w in words) / len(words)
    
    # Check for navigation/footer content (low quality indicators)
    nav_keywords = ['home', 'menu', 'back', 'login', 'register', 'faq', 'search']
    nav_count = sum(1 for word in words if word.lower() in nav_keywords)
    nav_ratio = nav_count / len(words) if words else 0
    
    # Check for copyright/footer text
    footer_keywords = ['copyright', 'rights reserved', 'all rights', '©', 'phpbb', 'powered by']
    has_footer = any(keyword in text.lower() for keyword in footer_keywords)
    
    # Check for educational/encyclopedic style (high quality indicators)
    has_sections = bool(re.search(r'\d+\.\s+\w+', text))  # Numbered sections like "1. Introduction"
    has_citations = bool(re.search(r'\([^)]*\d{4}[^)]*\)', text))  # Citations like "(Author 2020)"
    
    # Scoring
    score = 0.5
    
    # Positive indicators for wiki quality
    if avg_word_len > 5:
        score += 0.15
    if len(words) > 200:
        score += 0.1
    if has_sections:
        score += 0.15
    if has_citations:
        score += 0.15
    
    # Negative indicators (cc quality)
    if nav_ratio > 0.05:
        score -= 0.2
    if has_footer:
        score -= 0.15
    if len(lines) < 5 and len(words) < 50:
        score -= 0.2
    
    score = max(0.1, min(0.99, score))
    
    if score > 0.5:
        return 'wiki', score
    else:
        return 'cc', 1.0 - score


def gopher_quality_filter(text: str) -> bool:
    """Apply Gopher quality filter rules.
    
    Based on the paper "Quality at a Glance: An Audit of Web-Crawled Datasets"
    by Rae et al. (2021).
    
    Rules:
    1. Between 50 and 100,000 non-symbol words
    2. Mean word length between 3 and 10 characters
    3. Less than 30% of lines end with ellipsis (...)
    4. At least 80% of words contain at least one alphabetic character
    
    Args:
        text: Input text to filter.
        
    Returns:
        True if text passes all filters, False otherwise.
    """
    # Split into words and lines
    words = text.split()
    lines = text.split('\n')
    
    if len(words) == 0:
        return False
    
    # Rule 1: Between 50 and 100,000 non-symbol words
    # Non-symbol words = words with at least one alphanumeric character
    non_symbol_words = [w for w in words if any(c.isalnum() for c in w)]
    if len(non_symbol_words) < 50 or len(non_symbol_words) > 100000:
        return False
    
    # Rule 2: Mean word length between 3 and 10 characters
    # Only consider words with alphabetic characters
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if alpha_words:
        mean_word_len = sum(len(w) for w in alpha_words) / len(alpha_words)
        if mean_word_len < 3 or mean_word_len > 10:
            return False
    
    # Rule 3: Less than 30% of lines end with ellipsis
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith('...'))
        ellipsis_ratio = ellipsis_lines / len(lines)
        if ellipsis_ratio > 0.30:
            return False
    
    # Rule 4: At least 80% of words contain at least one alphabetic character
    words_with_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    alpha_ratio = words_with_alpha / len(words)
    if alpha_ratio < 0.80:
        return False
    
    return True
