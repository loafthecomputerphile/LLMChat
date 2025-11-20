from __future__ import annotations
from typing import TYPE_CHECKING
import re

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

from src.chat_model import ChatModel, ModelParams
from src.extractors import *
from src import variables

if TYPE_CHECKING:
    from llama_index.core import Document
    

def strip_all_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)



def test_chat_model() -> None:
    splitter: LangchainNodeParser = LangchainNodeParser(
        RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
    )

    router: ExtractionRouter = ExtractionRouter()
    router.add_extractor("text", plain_extractor, splitter)
    router.add_file_mapping("text", ["txt", "csv", "text"])

    model: ChatModel = ChatModel(router)  
    params: ModelParams = ModelParams(
        temperature=0.7, context_window=48000, rag_top_k=4, 
        history_tokens=5120, long_term_memory=True, long_term_tokens=5120, 
        top_k_memory=4
    )
    
    model.load_parameters(params)
    model.load_model(variables.BASE_MODEL)
    
    result: str = model.prompt("/no_think what is 2+2")
    print(result)
    
    
