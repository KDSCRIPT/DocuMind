"""
chunk_extraction.py

This module is responsible for processing a PDF file and breaking it into manageable
chunks of text for downstream NLP tasks like embedding or fine-tuning.

Explanation:
- First, the PDF is opened using PyMuPDF.
- All text from each page is extracted and joined together with double line breaks for separation.
- We then clean up by checking that there is meaningful text present.
- The text is split into individual sentences using NLTK’s sentence tokenizer.
- We then assemble these sentences into chunks, each up to a specified word limit.
- To maintain context between chunks, a word overlap is preserved at the end of each chunk.
- The resulting list of text chunks can be used for embeddings or QA dataset generation.

Function:
- extract_chunks_from_pdf(path, chunk_size, overlap): Extracts sentence-based text
  chunks from the PDF using overlapping sliding windows, and returns a list of chunks.
"""

import pymupdf
from constants import CHUNK_SIZE, OVERLAP
from nltk.tokenize import sent_tokenize

def extract_chunks_from_pdf(path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    doc = pymupdf.open(path)
    text = "\n\n".join(page.get_text() for page in doc)
    if not text.strip():
        raise ValueError("⚠️ No text found in the PDF.")

    sentences = sent_tokenize(text)
    if not sentences:
        raise ValueError("⚠️ Unable to tokenize any sentences.")

    chunks, current, total = [], [], 0
    for sent in sentences:
        wc = len(sent.split())
        if total + wc > chunk_size:
            if current:
                chunks.append(" ".join(current))
            current = current[-overlap:]
            total = sum(len(s.split()) for s in current)
        current.append(sent)
        total += wc
    if current:
        chunks.append(" ".join(current))
    return chunks
