"""
loading_and_caching.py

This module handles the loading or creation of cached data structures required for efficient
retrieval and question-answering from PDF files. It checks for existing embeddings, FAISS index, 
and SQLite chunk database, and regenerates them if not present.

Purpose:
- Avoid redundant computation by caching embeddings and index structures.
- Prepare necessary data for semantic search and retrieval-augmented generation (RAG) flow.
"""

import os
import faiss
import numpy as np
from chunk_extraction import extract_chunks_from_pdf
from constants import BATCH_SIZE
from create_db import create_sqlite_db
import logging

def set_cache_db_and_faiss_index(PDF_PATH, embedding_path, embedder, faiss_path, db_path):
    """
    Loads or generates FAISS index, embeddings, and chunk database from a PDF file.

    Explanation:
    - Checks whether FAISS index, embeddings, and chunk DB already exist in the cache.
    - If cached data exists, loads them directly (saves time).
    - If not, extracts text chunks from the PDF using `extract_chunks_from_pdf`.
    - Encodes these chunks into embeddings using the provided `embedder`.
    - Saves the embeddings to disk.
    - Creates a FAISS index for fast similarity search and stores it.
    - Creates an SQLite database storing the text chunks for later retrieval.

    Args:
    - PDF_PATH (str): Path to the PDF file.
    - embedding_path (str): Path where the embeddings will be saved or loaded from.
    - embedder (SentenceTransformer): Preloaded sentence transformer for encoding text.
    - faiss_path (str): Path to the FAISS index file.
    - db_path (str): Path to the SQLite chunks database.

    Returns:
    - embeddings (np.ndarray): Matrix of chunk embeddings.
    - index (faiss.IndexFlatL2): FAISS index object for similarity search.
    """
    if os.path.exists(faiss_path) and os.path.exists(db_path) and os.path.exists(embedding_path):
        print("📦 Loading cached FAISS index, SQLite DB, and embeddings...")
        index = faiss.read_index(faiss_path)
        embeddings = np.load(embedding_path)
    else:
        print("📄 Extracting and chunking PDF...")
        chunks = extract_chunks_from_pdf(PDF_PATH)
        print(f"🧠 {len(chunks)} chunks extracted. Encoding embeddings...")

        try:
            embeddings = embedder.encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)
            np.save(embedding_path, embeddings)

            dimension = embeddings[0].shape[0]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            faiss.write_index(index, faiss_path)
            create_sqlite_db(db_path, chunks)

        except Exception as e:
            logging.error(f"Embedding failure: {e}")
            raise

    return embeddings, index
