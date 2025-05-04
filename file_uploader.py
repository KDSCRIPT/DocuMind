"""
file_uploader.py

This module provides a graphical interface for selecting a PDF file using Tkinter.
It hides the main Tkinter window and prompts the user to choose a file via a native file dialog.

Purpose:
Used to allow the user to upload a PDF interactively, which will later be processed
for chunk extraction and QA tasks.
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the root window so only the file dialog appears
Tk().withdraw()

def file_uploader():
    """
    Opens a file dialog for the user to select a PDF file.

    Explanation:
    - Launches a file picker dialog restricted to PDF files.
    - Returns the path of the selected file if a file is chosen.
    - Raises a ValueError if the user closes the dialog without selecting a file.

    Returns:
    - str: Path to the selected PDF file.

    Raises:
    - ValueError: If no file is selected.
    """
    pdf_path = askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not pdf_path:
        raise ValueError("⚠️ No PDF file")
    return pdf_path
