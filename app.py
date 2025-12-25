import streamlit as st
from ingestion.loaders import load_pdfs
from ingestion.chunker import chunk_documents
from vectorstore.faiss_index import index_documents, load_faiss_index
from rag.query_router import route_query
from rag.web_search import tavily_search
from rag.context_builder import build_context
from rag.qa_chain import generate_answer

st.set_page_config(page_title="Hybrid RAG Search Engine")

st.title("📚 Hybrid Multi-Document RAG Search Engine")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Index Documents") and uploaded_files:
    paths = []
    for file in uploaded_files:
        path = f"data/uploads/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())
        paths.append(path)

    docs = load_pdfs(paths)
    chunks = chunk_documents(docs)
    index_documents(chunks)
    st.success("Documents indexed successfully!")

query = st.text_input("Ask a question")

if query:
    db = load_faiss_index()
    route = route_query(query)

    doc_chunks = db.similarity_search(query, k=4) if db else []
    web_results = tavily_search(query) if route in ["web", "hybrid"] else []

    context = build_context(doc_chunks, web_results)
    answer = generate_answer(context, query)

    st.markdown("### ✅ Answer")
    st.write(answer)
