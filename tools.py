
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("TAVILY_API_KEY is missing in .env file")

tavily = TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information.
    Returns titles, URLs and snippets."""

    try:
        results = tavily.search(
            query=query,
            max_results=5
        )

        output = []

        for result in results.get("results", []):
            output.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Snippet: {result.get('content', '')[:500]}\n"
            )

        return "\n----\n".join(output) if output else "No search results found."

    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL."""

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav",
            "footer", "header", "noscript"
        ]):
            tag.decompose()

        content = soup.get_text(separator=" ", strip=True)

        return content[:6000] if content else "No readable content found."

    except requests.RequestException as e:
        return f"Could not scrape URL: {str(e)}"

    except Exception as e:
        return f"Unexpected scraping error: {str(e)}"
