import os
import streamlit as st
from main import process_and_store_docx, fetch_chunks_by_query, generate_answer_with_citation, process_new_docs

# Directory for uploaded documents
DOCX_DIR = "data/"
os.makedirs(DOCX_DIR, exist_ok=True)

def main():
    st.title("RAG Application with Dynamic Document Uploads")

    # Document Upload Section
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader("Upload .docx files", type="docx", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            for uploaded_file in uploaded_files:
                filepath = os.path.join(DOCX_DIR, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                process_and_store_docx(filepath)
            st.success("Documents processed and indexed!")

    # Query Section
    st.header("Ask Your Question")
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Searching for the best answers..."):
            results = fetch_chunks_by_query(query, top_k=3)
            
            # Generate summarized answer with citations
            st.subheader("Summarized Answer with Citations:")
            answer = generate_answer_with_citation(results, query)
            st.write(answer)

    # Reprocess New Documents
    if st.button("Reprocess All Documents"):
        with st.spinner("Reprocessing all documents..."):
            process_new_docs(DOCX_DIR)
            st.success("All documents have been reprocessed!")

if __name__ == "__main__":
    main()
