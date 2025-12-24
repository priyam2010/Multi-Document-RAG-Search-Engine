from langchain.document_loaders import PyPDFLoader

def load_pdfs(file_paths):
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs
