"""PII (Personally Identifiable Information) masking utilities."""

import re


# Email regex pattern
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
)

# Phone number regex patterns (various formats)
PHONE_PATTERN = re.compile(
    r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
)

# IP address regex pattern (IPv4)
IP_PATTERN = re.compile(
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
)

# Replacement strings
EMAIL_REPLACEMENT = "|||EMAIL_ADDRESS|||"
PHONE_REPLACEMENT = "|||PHONE_NUMBER|||"
IP_REPLACEMENT = "|||IP_ADDRESS|||"


def mask_emails(text: str) -> tuple[str, int]:
    """Mask email addresses in text.
    
    Args:
        text: Input text.
        
    Returns:
        Tuple of (masked_text, num_masked).
    """
    matches = list(EMAIL_PATTERN.finditer(text))
    num_masked = len(matches)
    
    # Replace from end to start to preserve positions
    masked_text = text
    for match in reversed(matches):
        masked_text = masked_text[:match.start()] + EMAIL_REPLACEMENT + masked_text[match.end():]
    
    return masked_text, num_masked


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Mask phone numbers in text.
    
    Args:
        text: Input text.
        
    Returns:
        Tuple of (masked_text, num_masked).
    """
    matches = list(PHONE_PATTERN.finditer(text))
    num_masked = len(matches)
    
    masked_text = text
    for match in reversed(matches):
        masked_text = masked_text[:match.start()] + PHONE_REPLACEMENT + masked_text[match.end():]
    
    return masked_text, num_masked


def mask_ips(text: str) -> tuple[str, int]:
    """Mask IP addresses in text.
    
    Args:
        text: Input text.
        
    Returns:
        Tuple of (masked_text, num_masked).
    """
    matches = list(IP_PATTERN.finditer(text))
    num_masked = 0
    masked_text = text
    
    # Validate IP addresses (each octet should be 0-255)
    for match in reversed(matches):
        ip = match.group()
        octets = ip.split('.')
        if all(0 <= int(octet) <= 255 for octet in octets):
            num_masked += 1
            masked_text = masked_text[:match.start()] + IP_REPLACEMENT + masked_text[match.end():]
    
    return masked_text, num_masked
