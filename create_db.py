"""
create_db.py

This module handles the creation and population of a SQLite database with text chunks 
extracted from a PDF. It's used to persistently store the document content in a structured format 
for easy retrieval and embedding association.
"""

import sqlite3
from constants import CHUNK_EMBEDDING_TABLE

def create_sqlite_db(db_path, chunks):
    """
    Creates a SQLite database (if it doesn't exist) and stores text chunks in a table.

    Explanation:
    - Connects to (or creates) a SQLite database at the given path.
    - Creates a table (if it doesn't already exist) to store chunked text, with auto-incrementing IDs.
    - Inserts each chunk into the database as a new row.
    - Commits the transaction and closes the connection.
    
    Parameters:
    - db_path (str): Path to the SQLite database file.
    - chunks (list of str): List of text chunks to store.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"CREATE TABLE IF NOT EXISTS {CHUNK_EMBEDDING_TABLE} (id INTEGER PRIMARY KEY, content TEXT)")
    c.executemany(f"INSERT INTO {CHUNK_EMBEDDING_TABLE} (content) VALUES (?)", [(chunk,) for chunk in chunks])
    conn.commit()
    conn.close()
