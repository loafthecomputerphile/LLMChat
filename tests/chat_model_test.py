from __future__ import annotations
from typing import TYPE_CHECKING
import re, asyncio
import warnings

warnings.filterwarnings("ignore")

import pytest
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.tools import FunctionTool

from src.chat_model import ChatModel, ModelParams
from src.extractors import *
from src import variables

if TYPE_CHECKING:
    from llama_index.core import Document
    


TEST_FILE_FOLDER: Path = Path(__file__).parent / "docs" / "rag_test_doc"

def strip_all_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)



def make_model() -> ChatModel:
    splitter: LangchainNodeParser = LangchainNodeParser(
        RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=32)
    )

    router: ExtractionRouter = ExtractionRouter()
    router.add_extractor("text", plain_extractor, splitter)
    router.add_file_mapping("text", ["txt", "csv", "text"])

    model: ChatModel = ChatModel(router)  
    params: ModelParams = ModelParams(
        temperature=0.7, context_window=32000, rag_top_k=4, 
        history_tokens=26880, long_term_memory=True, long_term_tokens=5120, 
        top_k_memory=4
    ) 
    
    model.load_parameters(params)
    model.load_model(variables.BASE_MODEL)
    
    return model


chat_model: ChatModel = make_model()



def test_chat_model(printer) -> None:
    global chat_model
    
    result: str = str(chat_model.prompt("/no_think what is 2+2"))
    printer(result)
    

def test_rag(printer) -> None:
    global chat_model
    
    chat_model.add_documents([
        str(TEST_FILE_FOLDER / "rag_test.txt")
    ])
    
    result: str =  str( chat_model.prompt("/no_think from the docs how old is john"))
    printer(result)
    
    chat_model.add_documents([
        str(TEST_FILE_FOLDER / "rag_test_2.txt")
    ])
    
    result: str = str( chat_model.prompt("""
        /no_think
        Answer these questions:
            - from the document about john how old is he in the beginning of the story and then at the end of the story
            - from the document about mathew what rattled the door
    """))
    printer(result)


def test_tool_calling(printer) -> None:
    global chat_model
    
    def sqrt(number: float) -> float:
        return number ** 0.5
    
    tool: FunctionTool = FunctionTool.from_defaults(
        sqrt,
        name="square_root_function",
        description="use this function to get the square root of a number"
    )
    
    chat_model.add_tools([tool])
    
    result: str = str(chat_model.prompt("/no_think what is the square root of 8367391"))
    printer(result)
    
    

    