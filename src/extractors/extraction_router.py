from typing import Callable, TYPE_CHECKING, TypeAlias, Any, Type

import filetype

from ..flags import EXTRACTION_ERROR_FLAG, ExtractionErrors

if TYPE_CHECKING:
    from llama_index.core.node_parser import TextSplitter
    from llama_index.core.schema import BaseNode
    from llama_index.core import Document
    
    
__all__ = ["Extractor", "ExtractionRouter"]


Extractor: TypeAlias = Callable[[str], list[Document]]


class SplitExtractor:
    
    __slots__ = ("extractor", "splitter")
    
    def __init__(self, extractor: Extractor, splitter: Type[TextSplitter] | None = None, splitter_kwargs: dict[str, Any] | None = None) -> None:
        self.extractor: Extractor = extractor
        self.splitter: TextSplitter | None = None
        if splitter:
            self.splitter: TextSplitter = splitter(**(splitter_kwargs if splitter_kwargs else {}))
    
    def run(self, path: str) -> list[Document] | list[BaseNode]:
        if not self.splitter:
            return self.extractor(path)
        
        documents: list[BaseNode] = []
        for document in self.extractor(path):
            documents.append(self.splitter(document))
            
        return documents


class ExtractionRouter:
    
    __slots__ = ("extractors", "file_map")
    
    def __init__(self) -> None:
        self.extractors: dict[str, SplitExtractor] = dict()
        self.file_map: dict[str, str] = dict()
        
    def add_extractor(self, extractor_name: str, extractor: Extractor, splitter: Type[TextSplitter] = None, splitter_kwargs: dict[str, Any] = None) -> None:
        self.extractors[extractor_name] = SplitExtractor(
            extractor=extractor, splitter=splitter, 
            splitter_kwargs=splitter_kwargs
        )
        
    def add_file_mapping(self, extractor_name: str, mime_types: list[str]) -> None:
        if extractor_name not in self.extractors:
            raise ValueError("extractor_namemus be a registerd extractor")
        
        for mime_type in mime_types:
            self.file_map[mime_type] = extractor_name
            
    def extract(self, file_path: str) -> list[Document] | str:
        try:
            mime_type: str = filetype.guess_mime(file_path)
            if mime_type not in self.file_map:
                return ExtractionErrors.FILE_TYPE_NOT_RECOGNIZED
            return self.extractors[self.file_map[mime_type]].run(file_path)
        except Exception as e:
            return ExtractionErrors.UNKNOWN_ERROR
                            


