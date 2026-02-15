"""Content classification utilities for NSFW and toxic speech detection."""

from typing import Any


# Simple keyword-based NSFW detection
NSFW_KEYWORDS = [
    'cock', 'cunt', 'asshole', 'fuck', 'shit', 'porn', 'sex', 'nude',
    'naked', 'xxx', 'adult', 'obscene'
]

# Simple keyword-based toxic speech detection
TOXIC_KEYWORDS = [
    'idiot', 'moron', 'stupid', 'fuck', 'shit', 'damn', 'hell',
    'asshole', 'bastard', 'damned', 'fucking', 'f*ck'
]


def classify_nsfw(text: str) -> tuple[Any, float]:
    """Classify text as NSFW or non-NSFW.
    
    Args:
        text: Input text to classify.
        
    Returns:
        Tuple of (classification, score).
        classification is 'nsfw' or 'non-nsfw'.
    """
    text_lower = text.lower()
    
    # Count NSFW keywords
    count = sum(1 for keyword in NSFW_KEYWORDS if keyword in text_lower)
    
    # Simple scoring: more keywords = higher score
    if count > 0:
        score = min(0.5 + 0.1 * count, 0.99)
        return 'nsfw', score
    else:
        # Check for all-caps shouting (common in toxic posts)
        if text.isupper() and len(text) > 20:
            score = 0.7
            return 'nsfw', score
        return 'non-nsfw', 0.95


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    """Classify text as toxic or non-toxic.
    
    Args:
        text: Input text to classify.
        
    Returns:
        Tuple of (classification, score).
        classification is 'toxic' or 'non-toxic'.
    """
    text_lower = text.lower()
    
    # Count toxic keywords
    count = sum(1 for keyword in TOXIC_KEYWORDS if keyword in text_lower)
    
    # Check for personal attacks pattern
    attack_patterns = ['you', 'your', 'that']
    has_attack = any(p in text_lower for p in attack_patterns)
    
    if count > 0 and has_attack:
        score = min(0.6 + 0.1 * count, 0.99)
        return 'toxic', score
    elif count > 1:
        score = min(0.5 + 0.1 * count, 0.99)
        return 'toxic', score
    else:
        return 'non-toxic', 0.90
