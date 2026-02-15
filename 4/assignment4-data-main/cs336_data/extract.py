"""HTML text extraction utilities."""

from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Extract plain text from HTML bytes.
    
    Args:
        html_bytes: Raw HTML content as bytes.
        
    Returns:
        Extracted plain text, or None if extraction fails.
    """
    if html_bytes is None:
        return None
    
    try:
        # Decode bytes to string
        html_str = html_bytes.decode('utf-8', errors='replace')
        text = extract_plain_text(html_str)
        return text
    except Exception:
        return None
