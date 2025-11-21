from __future__ import annotations
import os, subprocess, asyncio
from typing import Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory, VectorMemory, SimpleComposableMemory
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from pydantic import BaseModel

from ..paths import PORTABLE_OLLAMA, OLLAMA_HOME_FOLDER, MODELS_FOLDER
from .. import variables
from ..extractors.extraction_router import ExtractionRouter

if TYPE_CHECKING:
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core import Document
    from llama_index.core.query_engine import BaseQueryEngine
    from subprocess import Popen, STARTUPINFO
    


    
class ModelParams(BaseModel):
    temperature: float
    context_window: int
    rag_top_k: int
    history_tokens: int
    long_term_memory: bool
    long_term_tokens: int
    top_k_memory: int
    
    

RAG_PROMPT: str =  """
Answers questions by retrieving relevant passages from the indexed document set currently in memory.
Use it only when the question involves content that may exist in those documents.
Skip this tool for open-ended, speculative, or conversational prompts.
"""

EMBEDDING_DIMENSIONS: int = 384


def create_rag_tool(model: Ollama, embeddimg_model: BaseEmbedding, description: str, top_k: int = 4) -> tuple[QueryEngineTool, BaseIndex]:
    index: VectorStoreIndex = VectorStoreIndex([], embed_model=embeddimg_model)
    
    query_engine: BaseQueryEngine = index.as_query_engine(llm=model, similarity_top_k=top_k)
    query_tool: QueryEngineTool = QueryEngineTool.from_defaults(
        query_engine=query_engine, name="RAGSearch", description=description
    )
    
    return query_tool, index


def make_cutsom_memory(
    memory_tokens: int, use_vector_store: bool, embedding_model: BaseEmbedding | None = None, 
    vector_tokens: int | None = None, top_limit: int | None = None
    ) -> SimpleComposableMemory:
    memory: Memory = Memory.from_defaults(token_limit=memory_tokens)
    vector_memory: VectorMemory | None = None
    
    if use_vector_store:
        vector_memory: VectorMemory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=embedding_model,
            index_kwargs={
                "similarity_top_k": top_limit,
                "max_tokens": vector_tokens,  
            }
        )
        
    return SimpleComposableMemory(
        primary_memory=memory,
        secondary_memory_sources=vector_memory if vector_memory is None else [vector_memory]
    )


def prepend_path(path_to_add: str) -> dict:
    """Return a copy of os.environ with path_to_add placed at the front of PATH."""
    env = os.environ.copy()
    env["PATH"] = path_to_add + os.pathsep + env["PATH"]
    return env


class ChatModel:
    
    __slots__ = ("extraction_router", "agent", "system_prompt", "llm_name", "llm_params", "ollama_server", "error_flag", "memory", "model", "embedding", "vector_store", "tools")
    
    def __init__(self, extractor: ExtractionRouter, tools: list[FunctionTool] = []) -> None:
        self.extraction_router: ExtractionRouter = extractor
        self.embedding: BaseEmbedding = OllamaEmbedding(variables.EMBEDDING_MODEL_NAME, base_url=variables.SERVER_URL)
        self.error_flag: Exception | None = None
        self.agent: FunctionAgent  | None = None
        self.model: Ollama | None = None
        self.system_prompt: str | None = None
        self.llm_name: str | None = None
        self.llm_params: ModelParams | None = None
        self.ollama_server: Popen | None = None
        self.tools: list[FunctionTool] = []
        self.memory: Memory | None = None
        self.vector_store: BaseIndex | None = None
        
        os.environ.setdefault('OLLAMA_HOST', str(variables.SERVER_URL))
        os.environ.setdefault('OLLAMA_MODELS', str(MODELS_FOLDER))
        os.environ.setdefault('OLLAMA_HOME', str(OLLAMA_HOME_FOLDER))
        
        if tools:
            for tool in tools: self.add_tool(tool)
        
    def add_documents(self, paths: list[str]) -> list[str] | None:
        if len(paths) == 0:
            return
        
        with ThreadPoolExecutor(4) as pool:
            results: list[list[Document] | str] = pool.map(
                self.extraction_router.extract, paths
            )
        
        errors: list[str] = [paths[i] for i, el in enumerate(results) if isinstance(el, str)]
        
        if len(errors) == 0:
            return
        
        return errors
    
    def set_temperature(self, value: float) -> None:
        if self.agent is None: 
            return
        self.agent.llm.temperature = value
    
    def set_thinking(self, value: bool) -> None:
        if self.agent is None: 
            return
        self.agent.llm.thinking = value
    
    def set_context_window(self, value: bool) -> None:
        if self.agent is None: 
            return
        self.agent.llm.context_window = value
        
    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        if self.agent is not None:
            self.agent.llm.system_prompt = self.system_prompt
        
    def load_parameters(self, params: ModelParams) -> None:
        self.llm_params = params
        
    def add_tool(self, function: FunctionTool) -> None:
        self.tools.append(function)
            
    def run_ollama_server(self, timeout: float = 20) -> None:
        
        try:
            startup_info: STARTUPINFO = subprocess.STARTUPINFO()
            startup_info.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE
            
            self.ollama_server = subprocess.Popen(
                [str(PORTABLE_OLLAMA), "serve"], shell=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                startupinfo=startup_info
            )
            self.ollama_server.wait(timeout)
            
        except subprocess.TimeoutExpired as e:
            self.error_flag = e
            self.ollama_server.terminate()
            self.ollama_server.wait()
            
        except Exception as e:
            self.error_flag = e
        
    def load_model(self, name: str) -> None:
        self.llm_name = name
        self.run_ollama_server()
        
        self.model = Ollama(
            model=self.llm_name, temperature=self.llm_params.temperature,
            context_window=self.llm_params.context_window, base_url=variables.SERVER_URL
        )
        
        query_tool, self.vector_store = create_rag_tool(
            self.model, self.embedding, RAG_PROMPT, 
            self.llm_params.rag_top_k
        )
        
        self.add_tool(query_tool)
        self._initialize_memory()
        self.agent = FunctionAgent(
            llm=self.model, request_timeout=360.0,
            tools=self.tools, system_prompt=self.system_prompt
        )
    
    def _initialize_memory(self) -> None:
        self.memory = make_cutsom_memory(
            memory_tokens=self.llm_params.history_tokens, use_vector_store=self.llm_params.long_term_memory,
            embedding_model=self.embedding, vector_tokens=self.llm_params.long_term_tokens, 
            top_limit=self.llm_params.top_k_memory
        )
        
    async def aprompt(self, prompt_text: str) -> WorkflowHandler:
        return await self.agent.run(prompt_text, memory=self.memory)
    
    def prompt(self, prompt_text: str) -> str:
        return str(asyncio.run(self.aprompt(prompt_text)))