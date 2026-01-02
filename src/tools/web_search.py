from __future__ import annotations
import time
from typing import Callable, TYPE_CHECKING, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.node_parser import SentenceSplitter
from ddgs import DDGS

from ..extractors.readers import SimpleWebPageReader

if TYPE_CHECKING:
    from llama_index.core.node_parser import TextSplitter
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import BaseNode, Document
    from llama_index.core.query_engine import BaseQueryEngine


ENGINES = [
    "bing", "brave", "duckduckgo", "google", "mojeek", "yandex", "yahoo"
]


class WebSearchToolSpec(BaseToolSpec):
    """EfficientWebSearch tool spec."""

    spec_functions = ["web_search", "suggest_sites"]

    def __init__(self, store: VectorStoreIndex, splitter: TextSplitter) -> None:
        self.splitter: TextSplitter = splitter
        self.vector_index: VectorStoreIndex = store
        self.reader: SimpleWebPageReader = SimpleWebPageReader(html_to_text=True)
        
    def suggest_sites(self, query: str, max_results: Optional[int] = 5) -> list[dict[str, str]]:
        """
        Find websites related to a topic without answering the question.

        Use this tool when:
        - The user asks for sources, links, or places to read
        - The user wants recommendations of websites
        - The user asks “where can I find…”, “sites about…”, or “good resources for…”

        Do NOT use this tool when:
        - The user asks for a direct answer or explanation
        - The goal is factual synthesis rather than discovery

        Args:
            query (str): the string to search for
            max_results (Optional[int]): The maximum number of results to be returned

        """
        from ddgs import DDGS
        
        params = {
            "query": query,
            "max_results": max_results,
            "backend":"duckduckgo"
        }
        
        with DDGS() as ddg:
            return list(ddg.text(**params))
        
    def web_search(self, query: str) -> str:
        """
        Answer a question using live internet data and retrieval-augmented generation.

        Use this tool when:
        - The user asks a factual or explanatory question
        - The information may be recent, evolving, or niche
        - The answer benefits from reading multiple web pages

        Do NOT use this tool when:
        - The user only wants a list of websites or sources
        - The answer is purely conceptual and unlikely to require web data

        Args:
            query (str): the string to search for
        
        """
        results: list[dict[str, str]] = self.suggest_sites(query, max_results=3)
        urls: list[str] = [r["href"] for r in results]
        
        for i, doc in enumerate(self.reader.load_data(urls)):
            results[i]["text-content"] = doc.text
        
        return results

