# RAG Application for Document Search and Summarization

This is a Retrieval-Augmented Generation (RAG) application designed to process `.docx` files, generate embeddings for document chunks, and answer user queries by retrieving relevant chunks of text and summarizing them with citations.

## Features
- Upload and process `.docx` files.
- Split documents into chunks and generate embeddings for each chunk.
- Retrieve relevant chunks based on user queries using cosine similarity.
- Summarize retrieved chunks and generate an answer with citations.
- Easy deployment using Streamlit for interactive user interface.

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit (for the web interface)
- Hugging Face transformers (for model inference)
- SQLite (for storing document embeddings)
- Sentence Transformers (for generating embeddings)
- PyTorch

### Install Dependencies

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/rag-app.git](https://github.com/codewithleo1/RAG-Application-with-Dynamic-Document-Uploads/
cd rag-app
