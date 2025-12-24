from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def generate_answer(context, query):
    llm = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="Answer using only the context provided."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
    ]
    return llm(messages).content
