"""
feedback_logger.py

This module handles feedback logging in two formats:
1. A simple text log file for quick human-readable access.
2. A structured SQLite database for analytical or systematic review of feedback.

It supports tracking user responses to generated answers, useful for debugging, model evaluation,
and fine-tuning insights.
"""

from datetime import datetime
import sqlite3
from constants import FEEDBACK_LOGFILE, FEEDBACK_TABLE

def log_feedback(question, answer, feedback):
    """
    Logs feedback in a plain text file.

    Explanation:
    - Opens the log file in append mode.
    - Writes a timestamped record of the user's question, the model's answer, and their feedback.
    - This file is useful for quick manual inspection and debugging.
    
    Parameters:
    - question (str): The user question.
    - answer (str): The model's generated answer.
    - feedback (str): The user's written feedback.
    """
    with open(FEEDBACK_LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | Q: {question} | A: {answer} | Feedback: {feedback}\n")

def log_feedback_sqlite(db_path, question, answer, feedback, pdf_hash, pdf_path):
    """
    Logs feedback in a SQLite database with metadata for traceability.

    Explanation:
    - Connects to or creates a SQLite database at the specified path.
    - Creates a feedback table if it doesn't exist.
    - Inserts a row containing the timestamp, question, answer, feedback, PDF hash, and PDF path.
    - This structure enables structured querying and analysis for future tuning or reporting.
    
    Parameters:
    - db_path (str): Path to the SQLite feedback database.
    - question (str): User's question.
    - answer (str): Model-generated answer.
    - feedback (str): User feedback text.
    - pdf_hash (str): Hash of the PDF file used.
    - pdf_path (str): Path to the PDF file.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f'''CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
                    timestamp TEXT,
                    question TEXT,
                    answer TEXT,
                    feedback TEXT,
                    pdf_hash TEXT,
                    pdf_path TEXT
                )''')
    c.execute(f"INSERT INTO {FEEDBACK_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), question, answer, feedback, pdf_hash, pdf_path))
    conn.commit()
    conn.close()
