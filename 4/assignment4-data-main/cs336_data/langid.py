"""Language identification utilities."""

import os
from typing import Any

import fasttext

# Load the fasttext model
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lid.176.bin")
_model = None


def _get_model() -> fasttext.FastText:
    """Lazy load the fasttext model."""
    global _model
    if _model is None:
        _model = fasttext.load_model(_MODEL_PATH)
    return _model


def identify_language(text: str) -> tuple[Any, float]:
    """Identify the language of the given text.
    
    Args:
        text: Input text to identify language for.
        
    Returns:
        Tuple of (language_code, confidence_score).
        Language code is a 2-letter ISO code like 'en', 'zh', etc.
    """
    model = _get_model()
    # fasttext returns list of tuples: [(label, score), ...]
    # Label format is '__label__en', so we need to strip '__label__'
    predictions = model.predict(text.replace('\n', ' '), k=1)
    label = predictions[0][0].replace('__label__', '')
    score = float(predictions[1][0])
    return label, score
