from __future__ import annotations
from threading import Lock
import os, subprocess, asyncio, gc
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Generator, Any, AsyncGenerator
 
import nest_asyncio
from pydantic import BaseModel

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import MessageRole
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, ToolCall
from llama_index.core.memory import Memory, VectorMemory, SimpleComposableMemory

from ..paths import (
    PORTABLE_OLLAMA,  OLLAMA_HOME_FOLDER, 
    MODELS_FOLDER, PORTABLE_OLLAMA_EXE
)

from ..variables import ConfigLoader

if TYPE_CHECKING:
    from subprocess import Popen, STARTUPINFO
    
    from llama_index.core.schema import BaseNode
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.workflow.handler import WorkflowHandler
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    
    from ..extractors.extraction_router import ExtractionRouter


__all__ = ["ModelParams", "SYSREM_PROMPT", "ChatModel", "DirectToolDesc"]


if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ModelParams(BaseModel):
    temperature: float
    context_window: int
    rag_top_k: int
    history_tokens: int
    long_term_memory: bool
    long_term_tokens: int
    top_k_memory: int


EMBEDDING_DIMENSIONS: int = 384


class DirectToolDesc(BaseModel):
    name: str
    direction: str
    priotity: bool
    usage_examples: list[str]
    non_usage: list[str]
    
    def _make_list_sec(self, strings: list[str]) -> str:
        return "\n".join(f"- {s}" for s in strings)
     
    def make_md(self) -> str:
        example_section: str = self._make_list_sec(self.usage_examples)
        md_text: str = f"## {self.name}\n{self.direction}\n{example_section}"
        return md_text
    
    def __str__(self) -> str:
        return self.make_md()
        

config_loader: ConfigLoader = ConfigLoader()
config_loader.load()


class ChatModel:
    
    __slots__ = (
        "extraction_router", 
        "character_prompt", 
        "thinking_on", 
        "faiss_index", 
        "agent", 
        "system_prompt", 
        "llm_name", 
        "llm_params", 
        "ollama_server", 
        "error_flag", 
        "memory", 
        "model", 
        "embedding", 
        "vector_store", 
        "tools",
        "_stop_lock",
        "_stop_flag"
    )
    
    def __init__(self, extractor: ExtractionRouter, system_prompt: str | None = None, character_prompt: str | None = None) -> None:
        self.system_prompt: str | None = system_prompt
        self.character_prompt: str | None = character_prompt
        self.extraction_router: ExtractionRouter = extractor
        self.thinking_on: bool = False
        self.tools: list[BaseTool] = []
        self.model: FunctionCallingLLM | None = None
        self.llm_name: str | None = None
        self.ollama_server: Popen | None = None
        self.error_flag: Exception | None = None
        self.agent: FunctionAgent  | None = None
        self.llm_params: ModelParams | None = None
        self.embedding: BaseEmbedding | None = None
        self.memory: SimpleComposableMemory | None = None
        self.vector_store: VectorStoreIndex | None = None
        self._stop_lock: Lock = Lock()
        self._stop_flag: bool = False
        
        os.environ.setdefault('OLLAMA_HOST', config_loader.get("URLs", "SERVER_URL"))
        os.environ["OLLAMA_PATH"] = str(PORTABLE_OLLAMA_EXE / "ollama")
        os.environ.setdefault('OLLAMA_MODELS', str(MODELS_FOLDER))
        os.environ.setdefault('OLLAMA_HOME', str(OLLAMA_HOME_FOLDER))
        os.environ.setdefault('OLLAMA_NO_AUTOSTART', str(1))
        
    def add_documents(self, paths: list[str]) -> list[str] | None:
        errors: list[str] = []
        results: list[list[BaseNode] | str]
        deletion_indexes: list[int] = []
        path_length: int = len(paths)
        
        if path_length == 0: 
            return
        
        if path_length > 1:
            with ThreadPoolExecutor(4) as pool:
                results = pool.map(self.extraction_router.extract, paths)

            for i, _ in filter(lambda x: isinstance(x[1], str), enumerate(results)):
                errors.append(paths[i])
                deletion_indexes.append(i)
            
            if deletion_indexes:
                for index in deletion_indexes[::-1]: 
                    del results[index]               
            
        elif path_length == 1:
            results = [self.extraction_router.extract(paths[0])]
            if isinstance(results[0], str): 
                errors = [paths[0]]
        
        
        if self.vector_store is None: 
            self.vector_store = VectorStoreIndex(
                [], vector_store=None, 
                embed_model=self.embedding, show_progress=True
            )
        
        for res in results: 
            self.vector_store.insert_nodes(res)
            
        query_tool: QueryEngineTool = self._create_rag_tool(
            config_loader.get("Prompts", "RAG_PROMPT"), self.llm_params.rag_top_k
        )
        
        self.delete_tools(["document_data_tool"], False)
        self.add_tools([query_tool])
                
        if len(errors) > 0:
            return errors
        
    def delete_tools(self, names: list[str], update_tools: bool = True) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        self.tools = [t for t in self.tools if t.metadata.name not in names]
        
        if update_tools:
            self.agent.tools = self.tools
    
    def set_temperature(self, value: float) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        self.agent.llm.temperature = value
    
    def set_thinking(self, value: bool) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        self.thinking_on = value
    
    def set_context_window(self, value: bool) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        self.agent.llm.context_window = value
        
    def set_system_prompt(self, prompt: str) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        self.system_prompt = prompt
        self.agent.llm.system_prompt = self.system_prompt
        
    def load_parameters(self, params: ModelParams) -> None:
        self.llm_params = params
        
    def add_tools(self, functions: list[BaseTool]) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        if not isinstance(functions, list):
            raise TypeError("parameter 'functions' must be of type List[BaseTool]")
        
        valids: list[BaseTool] = []
        for function in functions:
            if not isinstance(function, BaseTool):
                raise TypeError("element in functions is not of tyoe BaseTool")
            valids.append(function)
        
        self.tools.extend(valids)
        self.agent.tools = self.tools
        
    def _create_rag_tool(self, description: str, top_k: int = 4) -> QueryEngineTool:
        query_engine: BaseQueryEngine = self.vector_store.as_query_engine(
            llm=self.model, similarity_top_k=top_k
        )
        
        query_tool: QueryEngineTool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="document_data_tool",
            description=description
        )

        return query_tool
            
    def run_ollama_server(self, timeout: float = 20) -> None:
        
        try:
            args: list[str] = [str(PORTABLE_OLLAMA), "serve"]
            
            startup_info: STARTUPINFO = subprocess.STARTUPINFO()
            startup_info.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE
            
            self.ollama_server = subprocess.Popen(
                args, shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=startup_info
            )
            
            self.ollama_server.wait(timeout)
            
        except subprocess.TimeoutExpired as e:
            self.error_flag = e
            self.ollama_server.terminate()
            self.ollama_server.wait()
            
        except Exception as e:
            self.error_flag = e
            
    def kill(self) -> None:
        
        def run_cmd(model_name: str) -> None:
            args: list[str] = [str(PORTABLE_OLLAMA), "stop", model_name]
            
            startup_info: STARTUPINFO = subprocess.STARTUPINFO()
            startup_info.dwFlags = subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE
            
            server: subprocess.Popen = subprocess.Popen(
                args, shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=startup_info
            )
            
            server.wait(10)
            
        models: list[str] = [
            config_loader.get("Models", key) 
            for key in ("EMBEDDING_MODEL_NAME", "BASE_MODEL")
        ]
        
        for model_name in models:
            try:
                run_cmd(model_name)
                gc.collect()
            except Exception as e:
                continue
    
    @contextmanager
    def _temporary_context(self) -> Generator[None, Any, Any]:
        if self.memory is None:
            yield
            return

        memory: Memory = self.memory.primary_memory
        injected: list[ChatMessage] = []

        def inject(prompt: str | None) -> None:
            if not prompt:
                return
            
            msg: ChatMessage = ChatMessage(
                role=MessageRole.SYSTEM, content=prompt,
                additional_kwargs={"ephemeral": True},
            )
            
            memory.put(msg)
            injected.append(msg)
            
        inject(self.system_prompt)
        inject(self.character_prompt)

        try:
            yield
        finally:
            messages: list[ChatMessage] = memory.get_all()
            for msg in reversed(injected):
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role != MessageRole.SYSTEM:
                        continue
                    if messages[i].content != msg.content:
                        continue
                    messages.pop(i)
                    break

            
    def load_model(self, tools: list[BaseTool] | None = None, new_tools: list[DirectToolDesc] | None = None) -> None:
        from ..tools import BetterWikipediaToolSpec, WebSearchToolSpec, UnitConvertionToolSpec, AutoChainedSympyMathToolSpec
        
        self.llm_name = config_loader.get("Models", "BASE_MODEL")
        self.run_ollama_server(10)
        
        server_url: str = config_loader.get("URLs", "SERVER_URL")
        system_prompt: str = config_loader.get("Prompts", "SYSREM_PROMPT")
        
        self.embedding = OllamaEmbedding(
            config_loader.get("Models", "EMBEDDING_MODEL_NAME"), 
            base_url=server_url, embed_batch_size=128
        )
        
        tool_map: map[str] = map(str, [] if not new_tools else new_tools)
        self.system_prompt = system_prompt.format(
            new_tools="" if new_tools is None else "\n\n".join(tool_map)
        )
        
        self.model = Ollama(
            model=self.llm_name, temperature=self.llm_params.temperature, system_prompt=self.system_prompt,
            context_window=self.llm_params.context_window, base_url=server_url, thinking=False
        )
        
        Settings.llm = self.model
        Settings.embed_model = self.embedding
        
        self._initialize_memory()
        self.agent = FunctionAgent(
            llm=self.model, request_timeout=360.0, 
            tools=self.tools, system_prompt=self.system_prompt
        )
        
        self.add_tools(BetterWikipediaToolSpec().to_tool_list())
        self.add_tools(WebSearchToolSpec().to_tool_list())
        self.add_tools(UnitConvertionToolSpec().to_tool_list())
        self.add_tools(AutoChainedSympyMathToolSpec().to_tool_list())
        
        if tools is not None:
            self.add_tools(tools)
    
    def _initialize_memory(self) -> None:
        memory: Memory = Memory.from_defaults(token_limit=self.llm_params.history_tokens)
        vector_memory: VectorMemory | None = None
        
        if not self.llm_params.long_term_memory:
            self.memory = SimpleComposableMemory(
                primary_memory=memory, secondary_memory_sources=None
            )
            return
            
        vector_memory: VectorMemory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=self.embedding,
            index_kwargs={
                "similarity_top_k": self.llm_params.top_k_memory,
                "max_tokens": self.llm_params.long_term_tokens,  
            }
        )
        
        self.memory = SimpleComposableMemory(
            primary_memory=memory, secondary_memory_sources=[vector_memory]
        )
        
    def clear_memory(self) -> None:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        self.memory.reset()
        
    def stop(self) -> None:
        with self._stop_lock:
            self._stop_flag = True

    def _check_stop(self) -> bool:
        with self._stop_lock:
            return self._stop_flag
        
    async def astream_prompt(self, prompt) -> AsyncGenerator[str, None]:
        self._stop_flag = False
        handler = self.agent.run(user_msg=prompt, memory=self.memory)

        async for event in handler.stream_events():
            if self._check_stop(): break
            
            if isinstance(event, AgentStream):
                yield event.delta
            elif isinstance(event, ToolCall):
                yield f"\n🔧 Tool called: {event.tool_name}\n"
        
    async def aprompt(self, prompt_text: str) -> WorkflowHandler:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        with self._temporary_context():
            if self.thinking_on:
                return await self.agent.run(f"/think {prompt_text}", memory=self.memory)
            return await self.agent.run(f"/no_think {prompt_text}", memory=self.memory)
    
    def prompt(self, prompt_text: str) -> WorkflowHandler:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        nest_asyncio.apply()
        async def func() -> WorkflowHandler: 
            with self._temporary_context():
                if self.thinking_on:
                    return await self.agent.run(f"/think {prompt_text}", memory=self.memory)
                return await self.agent.run(f"/no_think {prompt_text}", memory=self.memory)
        
        return asyncio.run(func())
    
    def add_memory(self, messages: list[ChatMessage]) -> None:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        self.memory.put_messages(messages)
    
    
    

        
