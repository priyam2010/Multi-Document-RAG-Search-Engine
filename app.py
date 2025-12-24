import os
import streamlit as st

from ingestion.loaders import load_pdfs
from ingestion.chunker import chunk_documents
from vectorstore.faiss_index import index_documents, load_faiss_index
from rag.query_router import route_query
from rag.web_search import tavily_search
from rag.context_builder import build_context
from rag.qa_chain import generate_answer

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Hybrid Multi-Document RAG Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Hybrid Multi-Document RAG Search Engine")
st.caption("FAISS • LangChain • Tavily • Streamlit")

# -------------------------------
# Sidebar: Document Management
# -------------------------------
st.sidebar.header("📂 Document Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

use_web = st.sidebar.toggle(
    "🌐 Enable Real-Time Web Search (Tavily)",
    value=True
)

# -------------------------------
# Save Uploaded Files
# -------------------------------
file_paths = []

if uploaded_files:
    os.makedirs("data/uploads", exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join("data/uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)

    st.sidebar.success(f"{len(file_paths)} file(s) ready for indexing")

# -------------------------------
# Index Documents
# -------------------------------
if st.sidebar.button("🔄 Index Documents"):
    if not file_paths:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Indexing documents..."):
            documents = load_pdfs(file_paths)
            chunks = chunk_documents(documents)
            index_documents(chunks)
        st.sidebar.success("Documents indexed successfully!")

# -------------------------------
# Load FAISS Index
# -------------------------------
db = load_faiss_index()

# -------------------------------
# Main Chat Interface
# -------------------------------
query = st.text_input("💬 Ask a question")

if query:
    if not db:
        st.warning("⚠️ No documents indexed yet.")
    else:
        with st.spinner("Generating answer..."):
            route = route_query(query)

            doc_chunks = []
            web_results = []

            if route in ["document", "hybrid"]:
                doc_chunks = db.similarity_search(query, k=5)

            if route in ["web", "hybrid"] and use_web:
                web_results = tavily_search(query)

            context = build_context(doc_chunks, web_results)
            answer = generate_answer(context, query)

        st.subheader("🧠 Answer")

        if route == "document":
            st.markdown("📄 **Document-Based Answer**")
        elif route == "web":
            st.markdown("🌐 **Web-Based Answer**")
        else:
            st.markdown("🔀 **Hybrid Answer (Document + Web)**")

        st.write(answer)

        with st.expander("📄 Document Evidence"):
            if doc_chunks:
                for i, doc in enumerate(doc_chunks, 1):
                    st.markdown(f"**Doc {i}:**")
                    st.write(doc.page_content)
            else:
                st.info("No document evidence used.")

        with st.expander("🌐 Web Evidence"):
            if web_results:
                for i, res in enumerate(web_results, 1):
                    st.markdown(f"**Web {i}:**")
                    st.write(res.get("content", ""))
            else:
                st.info("No web evidence used.")


