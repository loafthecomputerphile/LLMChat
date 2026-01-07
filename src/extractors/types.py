from __future__ import annotations
from typing import Callable, TypeAlias, TYPE_CHECKING

from pydantic import BaseModel, Field
from llama_index.core import Document
from llama_index.core.node_parser import TextSplitter

Extractor: TypeAlias = Callable[[str], list[Document]]

class ExtractorBundle(BaseModel):
    name: str
    extractor: Extractor | None = Field(default=None)
    splitter: TextSplitter | None = Field(default=None)
    types: list[str] = Field(default_factory=list)