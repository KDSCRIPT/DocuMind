"""
get_chunk.py

This module provides functionality to retrieve specific text chunks from a SQLite database,
given a list of chunk IDs. It is used to fetch relevant context chunks during the question
answering process, based on semantic similarity.

Purpose:
Supports the QA pipeline by fetching selected chunks from the database using their IDs.
"""

import sqlite3
from constants import CHUNK_EMBEDDING_TABLE

def get_chunks_by_ids(db_path, ids):
    """
    Retrieves text chunks from the SQLite database by their IDs.

    Explanation:
    - Connects to the database specified by `db_path`.
    - Constructs a dynamic SQL query using placeholders for the given list of IDs.
    - Executes the query and fetches the corresponding chunk rows.
    - Returns a dictionary mapping each ID to its corresponding text chunk.

    Args:
    - db_path (str): Path to the SQLite database file.
    - ids (list of int): List of chunk IDs to retrieve.

    Returns:
    - dict: A mapping of chunk IDs to their associated text content.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    q_marks = ','.join('?' for _ in ids)
    c.execute(f"SELECT id, content FROM {CHUNK_EMBEDDING_TABLE} WHERE id IN ({q_marks})", ids)
    rows = c.fetchall()
    conn.close()
    return {id_: chunk for id_, chunk in rows}
