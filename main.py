"""
main.py

This is the main script for a PDF-based question-answering chatbot. The program:
1. Handles PDF file upload and text extraction.
2. Loads or generates necessary cached data structures (FAISS index, embeddings, chunk DB).
3. Handles user queries and provides context-based answers using a pre-trained QA model.
4. Collects feedback from users to improve the system and enforces rate limiting on feedback.

Key Components:
- PDF Text Extraction: Extracts content from uploaded PDFs.
- Embedding Generation: Converts text chunks from the PDF into embeddings.
- Semantic Search: Utilizes FAISS index for efficient search of relevant content.
- QA Model: Uses a pre-trained model to generate answers from the retrieved context.
- Feedback Logging: Collects user feedback for potential improvement and logging.
"""

import pymupdf
import os
import hashlib
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer

from constants import RATE_LIMIT_SECONDS, FEEDBACK_DATABASE, FAISS_INDEX, CHUNKS_DATABASE, EMBEDDING_FILE, SENTENCE_TRANSFOMER_PATH, TEXT_TO_TEXT_MODEL_PATH
from file_uploader import file_uploader
from feedback_logger import log_feedback, log_feedback_sqlite
from get_chunk import get_chunks_by_ids
from loading_and_caching import set_cache_db_and_faiss_index
from sanitizer import sanitize_text
from load_qa_model import load_qa_model

def hash_text_content(filepath):
    """
    Hashes the content of the PDF file to create a unique identifier.

    This is used for caching purposes to avoid reprocessing the same file multiple times.

    Args:
    - filepath (str): Path to the PDF file.

    Returns:
    - str: MD5 hash of the content of the PDF file.
    """
    try:
        doc = pymupdf.open(filepath)
        full_text = "\n\n".join(page.get_text() for page in doc)
        return hashlib.md5(full_text.encode('utf-8')).hexdigest()
    except Exception as e:
        logging.error(f"Error hashing file text: {e}")
        raise

# Main process
PDF_PATH = file_uploader()  # Step 1: User selects a PDF file
pdf_hash = hash_text_content(PDF_PATH)  # Generate a unique hash for the file content
logging.basicConfig(filename="errors.log", level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")  # Setup error logging
last_feedback_time = {}  # Stores the timestamp for each question to enforce rate-limiting on feedback
cache_dir = os.path.join("cache", pdf_hash)  # Create a cache directory based on the file hash
os.makedirs(cache_dir, exist_ok=True)  # Ensure the cache directory exists
faiss_path = os.path.join(cache_dir, FAISS_INDEX)
db_path = os.path.join(cache_dir, CHUNKS_DATABASE)
embedding_path = os.path.join(cache_dir, EMBEDDING_FILE)
feedback_db_path = FEEDBACK_DATABASE
embedder = SentenceTransformer(SENTENCE_TRANSFOMER_PATH)  # Load the sentence transformer model
embeddings, index = set_cache_db_and_faiss_index(PDF_PATH, embedding_path, embedder, faiss_path, db_path)  # Generate embeddings and FAISS index
tokenizer, qa_model = load_qa_model()  # Load the QA model

# Truncate Long Contexts
def truncate_context(context, max_tokens=512):
    """
    Truncates the context to fit within the model's token limit.

    Args:
    - context (str): The context to be truncated.
    - max_tokens (int): Maximum token length allowed by the model.

    Returns:
    - str: The truncated context.
    """
    input_ids = tokenizer.encode(context, truncation=True, max_length=max_tokens)
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# Main QA Function
def get_answer(question, top_k=3):
    """
    Given a question, retrieves the relevant context from the PDF and generates an answer.

    The process includes:
    - Embedding the question.
    - Performing semantic search using the FAISS index.
    - Retrieving the most relevant chunks.
    - Generating an answer using the pre-trained QA model.

    Args:
    - question (str): The user's question.
    - top_k (int): The number of top relevant chunks to consider for the context.

    Returns:
    - str: The generated answer.
    """
    try:
        question = sanitize_text(question)  # Clean the question text
        question_vec = embedder.encode([question])  # Get the embedding for the question
        if question_vec.shape != (1, embeddings.shape[1]):
            raise ValueError("Invalid question vector shape.")

        # Search for relevant chunks using FAISS
        _, I = index.search(question_vec, k=top_k)
        ids = [int(i) + 1 for i in I[0]]  # Adjust FAISS indexing
        chunk_map = get_chunks_by_ids(db_path, ids)
        if not chunk_map:
            return "⚠️ Could not retrieve any chunks."

        # Create the context from the retrieved chunks
        context = " ".join(chunk_map[i] for i in ids if i in chunk_map)
        context = truncate_context(context)  # Ensure the context fits within the token limit

        # Generate the answer using the QA model
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        result = qa_model(prompt)[0]['generated_text']
        return result

    except Exception as e:
        logging.error(f"Error in get_answer(): {e}")
        return "⚠️ Error retrieving answer. Check logs."

# Main Command-Line Interface (CLI) Loop
print("\n📘 PDF QA Chatbot")
print("Type your question below. Type 'exit' to quit.\n")
while True:
    try:
        q = input("❓ Your question: ")
        if q.lower() == "exit":
            break
        q = sanitize_text(q)
        if not q:
            print("⚠️ Please enter a valid question.")
            continue

        answer = get_answer(q)  # Get the answer for the question
        print("💬 Answer:", answer)

        feedback = input("📝 Optional feedback: ")
        if feedback.strip():
            now = datetime.now()
            last_time = last_feedback_time.get(q)
            if last_time and (now - last_time).total_seconds() < RATE_LIMIT_SECONDS:
                print(f"⚠️ Feedback rate limit in effect. Try again later.")
                continue

            feedback = sanitize_text(feedback)
            log_feedback(q, answer, feedback)  # Log feedback to the feedback file
            log_feedback_sqlite(feedback_db_path, q, answer, feedback, pdf_hash, PDF_PATH)  # Log feedback in SQLite DB
            last_feedback_time[q] = now  # Update the timestamp for the last feedback

    except Exception as e:
        print("⚠️ Error:", str(e))
        logging.error(f"Main loop exception: {e}")
