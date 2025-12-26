import random

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from .utils import extract_readable_text


wiki_prunes: list[tuple[str, str]] = [
    ("class", "reference"),
    ("class", "references"),
    ("role", "navigation"),
]


class BetterWikipediaToolSpec(BaseToolSpec):
    """
    Specifies two tools for querying information from Wikipedia.
    """

    spec_functions = ["load_wiki", "search_wikipedia"]

    def load_wiki(self, page: str, lang: str = "en") -> str:
        """
        Retrieve a Wikipedia page. Useful for learning about a particular concept that isn't private information.

        Args:
            page (str): Title of the page to read.
            lang (str): Language of Wikipedia to read. (default: English)

        """
        import wikipedia, sys
        
        wikipedia.set_user_agent(f"Mozilla/5.0 (LLMBot{random.randint(-sys.maxsize,sys.maxsize)}@hotmail.com)")
        
        wikipedia.set_lang(lang)
        res: str | None = None
        
        try:
            wikipedia_page: wikipedia.WikipediaPage = wikipedia.page(page, auto_suggest=False)
            res = extract_readable_text(wikipedia_page.html(), wiki_prunes)
        except Exception as e:
            return "Unable to load page. Try searching instead."
        
        return res

    def search_wikipedia(self, query: str, lang: str = "en") -> str:
        """
        Search Wikipedia for a page related to the given query.
        Use this tool when `load_data` returns no results.

        Args:
            query (str): the string to search for

        """
        import wikipedia

        pages: list[str] = wikipedia.search(query)
        if len(pages) == 0:
            return "No search results."
        return self.load_wiki(pages[0], lang)
