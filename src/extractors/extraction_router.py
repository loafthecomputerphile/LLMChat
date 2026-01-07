from __future__ import annotations
from typing import Callable, TYPE_CHECKING, TypeAlias, Any, Type
import mimetypes, traceback

import filetype
from llama_index.core import Document

from ..flags import EXTRACTION_ERROR_FLAG, ExtractionErrors
from .extractors import plain_extractor
from .primary_extraction_router import recursive_splitter
from .types import Extractor, ExtractorBundle

if TYPE_CHECKING:
    
    from llama_index.core.node_parser import TextSplitter
    from llama_index.core.schema import BaseNode


__all__ = ["Extractor", "ExtractionRouter", "SplitExtractor", "make_default_router"]


class SplitExtractor:
    
    __slots__ = ("extractor", "splitter")
    
    def __init__(self, extractor: Extractor, splitter: TextSplitter) -> None:
        self.extractor: Extractor = extractor
        self.splitter: TextSplitter = splitter
    
    def __call__(self, path: str) -> list[BaseNode]:
        return self.splitter.get_nodes_from_documents(self.extractor(path))


class ExtractionRouter:
    
    __slots__ = ("extractors", "file_map")
    
    def __init__(self) -> None:
        self.extractors: dict[str, SplitExtractor] = dict()
        self.file_map: dict[str, str] = dict()
    
    def get_extractor_names(self) -> list[str]:
        return list(self.extractors.keys())
         
    def add_extractor(self, bundle: ExtractorBundle) -> None:        
        self.extractors[bundle.name] = SplitExtractor(
            extractor=bundle.extractor if bundle.extractor else plain_extractor, 
            splitter=bundle.splitter if bundle.splitter else recursive_splitter
        )
        
        for mime_type in bundle.types:
            self.file_map[mime_type] = bundle.name
        
    def add_file_mapping(self, extractor_name: str, mime_types: list[str]) -> None:
        if extractor_name not in self.extractors:
            raise ValueError("extractor_namemus be a registerd extractor")
        
        for mime_type in mime_types:
            self.file_map[mime_type] = extractor_name
               
    def extract(self, file_path: str) -> list[BaseNode] | str:
        try:
            if "." not in file_path:
                return ExtractionErrors.FILE_TYPE_NOT_RECOGNIZED
            file_type: str = file_path.split(".")[-1]
            if file_type not in self.file_map:
                return ExtractionErrors.FILE_TYPE_NOT_RECOGNIZED
            return self.extractors[self.file_map[file_type]](file_path)
        except Exception as e:
            traceback.print_exc()
            return ExtractionErrors.UNKNOWN_ERROR
                            

def make_default_router() -> ExtractionRouter:
    from .primary_extraction_router import bundles
    
    router: ExtractionRouter = ExtractionRouter()
    for bundle in bundles:
        router.add_extractor(bundle)
        
    return router