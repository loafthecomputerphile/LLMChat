from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.node_parser import SentenceSplitter
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from ..extractors.readers import AsyncWebPageReader

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import BaseNode, Document
    from llama_index.core.query_engine import BaseQueryEngine


class WebSearchToolSpec(BaseToolSpec):
    """EfficientWebSearch tool spec."""

    spec_functions = ["web_search", "duckduckgo_full_search"]

    def __init__(self) -> None:
        self.search_func: Callable[[str, str, int], list[dict[str, str]]] = DuckDuckGoSearchToolSpec().duckduckgo_full_search
        self.splitter: SentenceSplitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self.reader: AsyncWebPageReader = AsyncWebPageReader(html_to_text=True)
        
    def suggest_sites(self, query: str, max_results: Optional[int] = 5) -> list[dict[str, str]]:
        """
        Search the internet for a websites related to the given query and return.   
        Use this when the user requests potential sites to find information or suggestions

        Args:
            query (str): the string to search for
            max_results (Optional[int]): The maximum number of results to be returned

        """
        
        return self.search_func(query, max_results)
        
    def web_search(self, query: str) -> str:
        """
        Search the internet for a websites related to the given query and return .
        Use this tool when the user requests exact/specific information from the internet

        Args:
            query (str): the string to search for

        """
        
        results: list[dict[str, str]] = self.search_func(query, max_results=3)
        urls: list[str] = [r["href"] for r in results]

        docs: list[Document] = self.reader.load_data(urls)
        nodes: BaseNode = self.splitter.get_nodes_from_documents(docs)
        
        engine: BaseQueryEngine = VectorStoreIndex(nodes).as_query_engine(
            similarity_top_k=3
        )
        
        response: Response = engine.query(query)
        return str(response)

