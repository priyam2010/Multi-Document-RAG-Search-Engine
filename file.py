
OPENAI_API_KEY=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

TAVILY_API_KEY=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
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WikipediaLoader

def load_pdfs(paths):
    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

def load_texts(paths):
    docs = []
    for path in paths:
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs

def load_wikipedia(query):
    loader = WikipediaLoader(query=query, load_max_docs=2)
    return loader.load()
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = "faiss_index"

def index_documents(documents):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(INDEX_PATH)
    return db

def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
def route_query(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["latest", "recent", "current", "news"]):
        return "web"
    if any(word in q for word in ["compare", "vs", "difference"]):
        return "hybrid"
    return "document"
from langchain_community.tools.tavily_search import TavilySearchResults

def tavily_search(query: str, k: int = 3):
    tool = TavilySearchResults(max_results=k)
    return tool.invoke(query)
def build_context(doc_chunks, web_results=None):
    context = ""

    for i, doc in enumerate(doc_chunks):
        context += f"[Doc {i+1}] {doc.page_content}\n"

    if web_results:
        for i, res in enumerate(web_results):
            context += f"[Web {i+1}] {res['content']}\n"

    return context
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_answer(context, question):
    prompt = f"""
Answer ONLY using the context.
Cite sources as [Doc] or [Web].

Context:
{context}

Question:
{question}
"""
    return llm.invoke(prompt).content
