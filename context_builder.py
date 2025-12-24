def build_context(doc_chunks, web_results):
    context = ""
    for d in doc_chunks:
        context += d.page_content + "\n"
    for w in web_results:
        context += w.get("content", "") + "\n"
    return context
