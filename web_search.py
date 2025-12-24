from tavily import TavilyClient
import os

def tavily_search(query):
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client.search(query).get("results", [])
