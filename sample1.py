import fitz  # PyMuPDF
import nltk
import os
import faiss
import torch
import pickle
import hashlib
import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv
import re

nltk.download('punkt')
load_dotenv()

# --------------------------------------
# Config from .env
# --------------------------------------
PDF_PATH = os.getenv("PDF_PATH")
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
OVERLAP = int(os.getenv("OVERLAP", 100))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 30))

# --------------------------------------
# Logging setup
# --------------------------------------
logging.basicConfig(filename="errors.log", level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")

# --------------------------------------
# Rate Limiting Setup
# --------------------------------------
last_feedback_time = {}

# --------------------------------------
# Text-based hash for caching
# --------------------------------------
def hash_text_content(filepath):
    try:
        doc = fitz.open(filepath)
        full_text = "\n\n".join(page.get_text() for page in doc)
        return hashlib.md5(full_text.encode('utf-8')).hexdigest()
    except Exception as e:
        logging.error(f"Error hashing file text: {e}")
        raise

# --------------------------------------
# Input Sanitization
# --------------------------------------
def sanitize_text(text):
    text = text.strip()
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # remove non-printable
    return text

# --------------------------------------
# Feedback SQLite Logging
# --------------------------------------
def log_feedback(question, answer, feedback):
    with open("feedback.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | Q: {question} | A: {answer} | Feedback: {feedback}\n")

def log_feedback_sqlite(db_path, question, answer, feedback, pdf_hash, pdf_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    timestamp TEXT,
                    question TEXT,
                    answer TEXT,
                    feedback TEXT,
                    pdf_hash TEXT,
                    pdf_path TEXT
                )''')
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), question, answer, feedback, pdf_hash, pdf_path))
    conn.commit()
    conn.close()

# --------------------------------------
# PDF Chunk Extraction
# --------------------------------------
def extract_chunks_from_pdf(path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    doc = fitz.open(path)
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

# --------------------------------------
# SQLite Chunk DB
# --------------------------------------
def create_sqlite_db(db_path, chunks):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, content TEXT)")
    c.executemany("INSERT INTO chunks (content) VALUES (?)", [(chunk,) for chunk in chunks])
    conn.commit()
    conn.close()

def get_chunks_by_ids(db_path, ids):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    q_marks = ','.join('?' for _ in ids)
    c.execute(f"SELECT id, content FROM chunks WHERE id IN ({q_marks})", ids)
    rows = c.fetchall()
    conn.close()
    return {id_: chunk for id_, chunk in rows}

# --------------------------------------
# Load + Cache
# --------------------------------------
pdf_hash = hash_text_content(PDF_PATH)
cache_dir = os.path.join("cache", pdf_hash)
os.makedirs(cache_dir, exist_ok=True)

faiss_path = os.path.join(cache_dir, "faiss.index")
db_path = os.path.join(cache_dir, "chunks.db")
embedding_path = os.path.join(cache_dir, "embeddings.npy")
feedback_db_path = "feedback.db"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(faiss_path) and os.path.exists(db_path) and os.path.exists(embedding_path):
    print("📦 Loading cached FAISS index, SQLite DB, and embeddings...")
    index = faiss.read_index(faiss_path)
    embeddings = np.load(embedding_path)
else:
    print("📄 Extracting and chunking PDF...")
    chunks = extract_chunks_from_pdf(PDF_PATH)
    print(f"🧠 {len(chunks)} chunks extracted. Encoding embeddings...")

    try:
        if torch.cuda.is_available():
            print("⚠️ GPU Memory:", torch.cuda.memory_allocated() / (1024**2), "MB")
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

# --------------------------------------
# QA Model
# --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using {'GPU' if torch.cuda.is_available() else 'CPU'} for inference.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_model = pipeline("text2text-generation", model=MODEL_NAME, tokenizer=tokenizer, max_length=256, device=0 if torch.cuda.is_available() else -1)

# --------------------------------------
# Truncate Long Contexts
# --------------------------------------
def truncate_context(context, max_tokens=512):
    input_ids = tokenizer.encode(context, truncation=True, max_length=max_tokens)
    return tokenizer.decode(input_ids, skip_special_tokens=True)

# --------------------------------------
# Main QA Function
# --------------------------------------
def get_answer(question, top_k=3):
    try:
        question = sanitize_text(question)
        question_vec = embedder.encode([question])
        if question_vec.shape != (1, embeddings.shape[1]):
            raise ValueError("Invalid question vector shape.")

        _, I = index.search(question_vec, k=top_k)
        ids = [int(i) + 1 for i in I[0]]
        chunk_map = get_chunks_by_ids(db_path, ids)
        if not chunk_map:
            return "⚠️ Could not retrieve any chunks."

        context = " ".join(chunk_map[i] for i in ids if i in chunk_map)
        context = truncate_context(context)

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        result = qa_model(prompt)[0]['generated_text']
        return result

    except Exception as e:
        logging.error(f"Error in get_answer(): {e}")
        return "⚠️ Error retrieving answer. Check logs."

# --------------------------------------
# CLI Loop
# --------------------------------------
print("\n📘 PDF QA Chatbot (Enhanced)")
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

        answer = get_answer(q)
        print("💬 Answer:", answer)

        feedback = input("📝 Optional feedback: ")
        if feedback.strip():
            now = datetime.now()
            last_time = last_feedback_time.get(q)
            if last_time and (now - last_time).total_seconds() < RATE_LIMIT_SECONDS:
                print(f"⚠️ Feedback rate limit in effect. Try again later.")
                continue

            feedback = sanitize_text(feedback)
            log_feedback(q, answer, feedback)
            log_feedback_sqlite(feedback_db_path, q, answer, feedback, pdf_hash, PDF_PATH)
            last_feedback_time[q] = now

    except Exception as e:
        print("⚠️ Error:", str(e))
        logging.error(f"Main loop exception: {e}")