"""
sanitizer.py

This module provides a function to sanitize input text. It removes unwanted control characters and trims leading/trailing spaces from the input text.

Functions:
- sanitize_text: Cleans the input text by removing non-printable characters and trimming spaces.

"""
import re

def sanitize_text(text):
    """
    Sanitizes the input text by:
    1. Removing leading and trailing spaces.
    2. Removing control characters (ASCII 0-31 and 127).
    
    This ensures that the text is clean and free from characters that could interfere with further processing or cause issues in model inference.

    Args:
    - text (str): The input text to sanitize.

    Returns:
    - str: The sanitized text.
    """
    text = text.strip()  # Remove leading and trailing spaces
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # Remove non-printable ASCII characters (0-31 and 127)
    return text
