import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

INDEX_PATH = "data/faiss"

def index_documents(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_PATH)

def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_PATH, embeddings)
