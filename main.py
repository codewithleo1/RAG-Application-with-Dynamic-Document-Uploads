import os
import sqlite3
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize the model pipeline for text generation
pipe = pipeline("text-generation", model="microsoft/phi-1_5")

DB_FILE = "embeddings/chunks.db"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def initialize_database():
    """Initialize the SQLite database with the chunks table."""
    os.makedirs("embeddings", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            chunk_text TEXT,
            embedding BLOB,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chunk_to_db(chunk_id, chunk_text, embedding, source_file):
    """Save a chunk's details to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO chunks (chunk_id, chunk_text, embedding, source_file)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, chunk_text, embedding.tobytes(), source_file))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Skip if chunk already exists
    conn.close()

def load_and_chunk_docx(filepath, chunk_size=200):
    """Load a .docx file and split it into smaller chunks."""
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    chunks = []
    chunk_id = 0
    for paragraph in paragraphs:
        sentences = paragraph.split(". ")
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= chunk_size:
                chunk += sentence + ". "
            else:
                chunks.append((chunk_id, chunk.strip()))
                chunk = sentence + ". "
                chunk_id += 1
        if chunk:  # Append remaining chunk
            chunks.append((chunk_id, chunk.strip()))
    return chunks

def process_and_store_docx(filepath, chunk_size=200):
    """Process a single .docx file and store its chunks in the database."""
    chunks = load_and_chunk_docx(filepath, chunk_size)
    for chunk_id, chunk_text in chunks:
        unique_id = f"{os.path.basename(filepath)}-chunk-{chunk_id}"
        embedding = np.array(embedder.encode([chunk_text])[0])
        save_chunk_to_db(unique_id, chunk_text, embedding, os.path.basename(filepath))

def fetch_chunks_by_query(query, top_k=3):
    """Fetch the most relevant chunks for a query."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, chunk_text, embedding, source_file FROM chunks")
    rows = cursor.fetchall()
    query_embedding = np.array(embedder.encode([query])[0])
    similarities = []
    for chunk_id, chunk_text, embedding_blob, source_file in rows:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((chunk_id, chunk_text, similarity, source_file))
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
    conn.close()
    return similarities

def generate_answer_with_citation(chunks, query):
    """
    Generate a summarized answer using the model and include citations.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text, similarity, source_file).
        query: User's query.

    Returns:
        str: Generated answer with citations.
    """
    # Combine chunk texts for context
    context = "\n\n".join([f"[{chunk_id}] {chunk_text}" for chunk_id, chunk_text, _, _ in chunks])
    
    # Prepare prompt
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Provide a concise and summarized answer, and cite chunk IDs in parentheses after relevant sentences."
    )

    # Generate answer using the model
    response = pipe(prompt, max_length=300, truncation=True)[0]["generated_text"]

    return response

def process_new_docs(docx_dir):
    """Process all new .docx files in the data folder."""
    for filename in os.listdir(docx_dir):
        if filename.endswith(".docx"):
            filepath = os.path.join(docx_dir, filename)
            process_and_store_docx(filepath)

# Initialize the database
initialize_database()
