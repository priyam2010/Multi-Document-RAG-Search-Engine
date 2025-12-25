from langchain.document_loaders import PyPDFLoader

def load_pdfs(paths):
    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)
import os
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

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
def route_query(query: str):
    q = query.lower()
    if "latest" in q or "news" in q:
        return "web"
    if "paper" in q or "document" in q:
        return "document"
    return "hybrid"
from tavily import TavilyClient
import os

def tavily_search(query):
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query).get("results", [])
def build_context(doc_chunks, web_results):
    context = ""

    for d in doc_chunks:
        context += f"[DOC]\n{d.page_content}\n"

    for w in web_results:
        context += f"[WEB]\n{w.get('content', '')}\n"

    return context
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def generate_answer(context, query):
    llm = ChatOpenAI(temperature=0)

    messages = [
        SystemMessage(content="Answer strictly using the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
    ]

    return llm(messages).content
