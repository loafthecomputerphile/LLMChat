


from __future__ import annotations
from typing import  TYPE_CHECKING
import random, uuid, io
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core import Document

import pandas as pd

if TYPE_CHECKING:
    from wikipedia.wikipedia import WikipediaPage
    from llama_index.core.node_parser import TextSplitter
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import BaseNode, Document
    from llama_index.core.query_engine import BaseQueryEngine


wiki_prunes: list[tuple[str, str]] = [
    ("class", "reference"),
    ("class", "references"),
    ("role", "navigation"),
]


def table_to_csv(df: pd.DataFrame) -> str:
    """
    Convert a single DataFrame to CSV text with minimal overhead.
    """
    return df.fillna("").to_csv(
        index=False,      # drop the index, unnecessary for LLMs
        quoting=0         # minimal quoting
    )

def convert_tables_parallel(dfs: list[pd.DataFrame]) -> list[str]:
    """
    Convert multiple DataFrames (tables) to CSV in parallel.
    """
    with ThreadPoolExecutor() as pool:
        return list(pool.map(table_to_csv, dfs))


class BetterWikipediaToolSpec(BaseToolSpec):
    """
    Specifies two tools for querying information from Wikipedia.
    """

    spec_functions = ["search_wikipedia", "wiki_search", "load_wiki"]
    
    def __init__(self, store: VectorStoreIndex, splitter: TextSplitter) -> None:
        super().__init__()
        
        self.splitter: TextSplitter = splitter
        self.vector_index: VectorStoreIndex = store
        
    def _load(self, query: str, lang: str = "en") -> dict[str, str]:
        import wikipedia, sys

        pages: list[str] = wikipedia.search(query)
        if len(pages) == 0:
            return {"url":"", "content":"No search results."}
        
        wikipedia.set_user_agent(f"Mozilla/5.0 (LLMBot{random.randint(-sys.maxsize,sys.maxsize)}@hotmail.com)")
        wikipedia.set_lang(lang)
        
        wikipedia_page: WikipediaPage = wikipedia.page(pages[0], auto_suggest=False)
        
        try:
            data: pd.DataFrame = pd.read_html(io.StringIO(wikipedia_page.html()), attrs={"class": "wikitable"})
            tables = "\n\n".join(convert_tables_parallel(data))
        except Exception as e:
            tables=""
        
        return {
            "url":wikipedia_page.url,
            "content":f"{wikipedia_page.content}\n\n # TABLES: \n\n{tables}"
        }
        
    def search_wikipedia(self,  query: str) -> str:
        """
        Search Wikipedia for a topic and answer a question using the page content.

        """
        
        return self._load(query)
        
        try:
            for doc_id, _ in self.vector_index.ref_doc_info.items():
                self.vector_index.delete_ref_doc(doc_id)
            
            data: dict[str, str] = self._load(query, "en")
            
            doc: Document = Document(
                text=data["content"], id_=str(uuid.uuid4()), 
                metadata={"url":data["url"]}
            )
            
            nodes: BaseNode = self.splitter.get_nodes_from_documents([doc])
            self.vector_index.insert_nodes(nodes)
            
            engine: BaseQueryEngine = self.vector_index.as_query_engine(
                llm=Settings.llm, similarity_top_k=20
            )
            
            response: Response = engine.query(query)
            
        except Exception as e:
            return str(e)
        
        return str(response)
        
    def wiki_search(self, query: str) -> str:
        """
        Search Wikipedia for a topic and answer a question using the page content.

        """
        
        return self.search_wikipedia(query)
    
    def load_wiki(self, query: str) -> str:
        """
        Search Wikipedia for a topic and answer a question using the page content.

        """
        
        return self.search_wikipedia(query)
