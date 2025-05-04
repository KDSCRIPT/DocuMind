"""
constants.py

This module centralizes all configurable parameters and file paths used throughout
the application. Having these values in one place makes it easier to maintain and 
tweak the system behavior.

Explanation:
- It defines values for how PDF chunks are created (batch size, chunk size, overlap).
- Specifies rate-limiting for user feedback.
- Includes paths to local or remote ML models.
- Sets up names for database files, FAISS index, and log files.
"""

BATCH_SIZE = 16   # Number of sentences/chunks processed per batch
CHUNK_SIZE = 500  # Max words per chunk
OVERLAP = 100   # Overlapping words between chunks
RATE_LIMIT_SECONDS = 30  # Cooldown between user feedback submissions

SENTENCE_TRANSFOMER_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
TEXT_TO_TEXT_MODEL_PATH = "google/flan-t5-base"  # QA model

CHUNK_EMBEDDING_TABLE = "chunks"  # SQLite table for chunks
CHUNKS_DATABASE = "chunks.db"  # SQLite DB path for storing text chunks

FEEDBACK_LOGFILE = "feedback.log"  # Plaintext log file for feedback
FEEDBACK_TABLE = "feedback"  # SQLite table for structured feedback
FEEDBACK_DATABASE = "feedback.db"  # SQLite DB path for feedback

FAISS_INDEX = "faiss.index"  # File path for FAISS vector index

EMBEDDING_FILE="embeddings.npy" #File path for storing embeddings as numpy arrays

LOCAL_MODELS_DIRECTORY="local_models"
LOCAL_SENTENCE_TRANSFOMER_PATH = "all-MiniLM-L6-v2"  # Embedding model
LOCAL_TEXT_TO_TEXT_MODEL_PATH = "flan-t5-base"  # QA model
