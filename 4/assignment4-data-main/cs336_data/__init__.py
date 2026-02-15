import importlib.metadata

from .deduplication import exact_line_deduplication, minhash_deduplication
from .extract import extract_text_from_html_bytes
from .langid import identify_language
from .pii import mask_emails, mask_ips, mask_phone_numbers
from .quality import classify_quality, gopher_quality_filter
from .toxicity import classify_nsfw, classify_toxic_speech

__version__ = importlib.metadata.version("cs336-data")

__all__ = [
    "extract_text_from_html_bytes",
    "identify_language",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ips",
    "classify_nsfw",
    "classify_toxic_speech",
    "classify_quality",
    "gopher_quality_filter",
    "exact_line_deduplication",
    "minhash_deduplication",
]
