def route_query(query):
    q = query.lower()
    if "latest" in q or "news" in q:
        return "web"
    if "document" in q or "paper" in q:
        return "document"
    return "hybrid"
