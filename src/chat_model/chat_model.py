from __future__ import annotations
from threading import Lock
import os, subprocess, asyncio, gc, datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Generator, Any, AsyncGenerator, Callable
 
import nest_asyncio
from pydantic import BaseModel, Field
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.prompts import MessageRole
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, ToolCall
from llama_index.core.memory import Memory
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from ..extractors.extraction_router import make_default_router

from ..paths import (
    PORTABLE_OLLAMA,  OLLAMA_HOME_FOLDER, 
    MODELS_FOLDER, PORTABLE_OLLAMA_EXE
)

from ..variables import ConfigLoader
from ..knowledge import KnowledgeManager

if TYPE_CHECKING:
    from subprocess import Popen, STARTUPINFO
    
    from llama_index.core.schema import BaseNode
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.workflow.handler import WorkflowHandler
    from llama_index.core.workflow.events import Event
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    
    from ..extractors.extraction_router import ExtractionRouter


__all__ = ["ModelParams", "SYSREM_PROMPT", "ChatModel", "DirectToolDesc"]


if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ModelParams(BaseModel):
    temperature: float
    context_window: int
    rag_top_k: int
    history_tokens: int
    long_term_memory: bool = Field(False)
    long_term_tokens: int = Field(2048)
    top_k_memory: int = Field(4)





config_loader: ConfigLoader = ConfigLoader()
config_loader.load()





class ChatModel:
    
    __slots__ = (
        "extraction_router", "character_prompt", "thinking_on", 
        "faiss_index", "agent", "system_prompt", "llm_name", 
        "llm_params", "ollama_server", "error_flag", "memory", 
        "model", "embedding", "vector_store", "tools", "_stop_lock",
        "_stop_flag", "user_info", "tool_vector_store", "tool_text_splitter",
        "added_nodes"
    )
    
    def __init__(self, extractor: ExtractionRouter) -> None:
        self.extraction_router: ExtractionRouter = extractor
        self.thinking_on: bool = False
        self.tools: dict[str, BaseTool] = {}
        
        self.ollama_server: Popen | None = None
        self.error_flag: Exception | None = None
        
        self.llm_name: str | None = None
        self.llm_params: ModelParams | None = None
        
        self.agent: FunctionAgent  | None = None
        self.embedding: BaseEmbedding | None = None
        self.model: FunctionCallingLLM | None = None
        self.memory: Memory | None = None
        self.vector_store: VectorStoreIndex | None = None
        self.added_nodes: set[str] = set()
        
        self._stop_lock: Lock = Lock()
        self._stop_flag: bool = False
        
        self.user_info: str | None = None
        self.system_prompt: str | None = None
        self.character_prompt: str | None = None
        
        self.tool_vector_store: VectorStoreIndex | None = None
        self.tool_text_splitter: LangchainNodeParser = LangchainNodeParser(
            RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=256)
        )
        
        os.environ.setdefault('OLLAMA_HOST', config_loader.get("URLs", "SERVER_URL"))
        os.environ["OLLAMA_PATH"] = str(PORTABLE_OLLAMA_EXE / "ollama")
        os.environ.setdefault('OLLAMA_MODELS', str(MODELS_FOLDER))
        os.environ.setdefault('OLLAMA_HOME', str(OLLAMA_HOME_FOLDER))
        os.environ.setdefault('OLLAMA_NO_AUTOSTART', str(1))
        
    def add_documents(self, paths: list[str]) -> dict[str, str]:
        errors: dict[str, str] = {k:"ok" for k in paths}
        results: list[list[BaseNode] | str]
        deletion_indexes: list[int] = []
        path_length: int = len(paths)
        
        if path_length == 0: 
            return
        
        if path_length > 1:
            with ThreadPoolExecutor(4) as pool:
                results = pool.map(self.extraction_router.extract, paths)

            for i, _ in filter(lambda x: isinstance(x[1], str), enumerate(results)):
                errors[paths[i]] = "failed"
                deletion_indexes.append(i)
            
            if deletion_indexes:
                for index in deletion_indexes[::-1]: 
                    del results[index]               
            
        elif path_length == 1:
            results = [self.extraction_router.extract(paths[0])]
            if isinstance(results[0], str): 
                errors[paths[0]] = "failed"
                del results[0]
        
        
        if self.vector_store is None: 
            self.vector_store = VectorStoreIndex(
                [], vector_store=None, 
                embed_model=self.embedding, show_progress=True
            )
        
        for res in results:
            self._clear_nodes(res)
            self.vector_store.insert_nodes(res)
            
        query_tool: QueryEngineTool = self._create_rag_tool(
            config_loader.get("Prompts", "RAG_PROMPT"), self.llm_params.rag_top_k
        )
        
        self.delete_tools(["document_data_tool"], False)
        self.add_tools([query_tool])
        
        return {Path(k).name:v for k, v in errors.items()}
        
    def _clear_nodes(self, nodes: list[BaseNode]) -> None:
        clear_list: list[int] = []
        for i, node in enumerate(nodes):
            if node.hash in self.added_nodes:
                clear_list.append(i)
                continue
            self.added_nodes.add(node.hash)
            
        for index in clear_list[::-1]:
            del nodes[index]
        
    def delete_tools(self, names: list[str], update_tools: bool = True) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        for name in filter(lambda x: x in self.tools, names):
            del self.tools[name]
        
        if update_tools:
            self.agent.tools = self.get_tools()
    
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
        self.system_prompt = prompt
        
    def set_character_prompt(self, prompt: str) -> None:
        self.character_prompt = prompt
        
    def set_user_info(self, info: str) -> None:
        self.user_info = info
        
    def load_parameters(self, params: ModelParams) -> None:
        self.llm_params = params
        
    def get_tools(self) -> list[BaseTool]:
        return list(self.tools.values())

        
    def add_tools(self, functions: list[BaseTool]) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        if not isinstance(functions, list):
            raise TypeError("parameter 'functions' must be of type List[BaseTool]")
        
        valids: dict[str, BaseTool] = {}
        for function in functions:
            if not isinstance(function, BaseTool):
                raise TypeError("element in functions is not of tyoe BaseTool")
            valids[function.metadata.name] = function
        
        self.tools.update(valids)
        self.agent.tools = self.get_tools()
        
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
        
    def add_doc_store_query_engine(self, doc_store_paths: list[str]) -> None:
        new_tools: list[BaseTool] = []
        manager: KnowledgeManager = KnowledgeManager()
        for doc_store_path in doc_store_paths:
            index, info = manager.load_index(doc_store_path)
            new_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=index.as_query_engine(similarity_top_k=3),
                    name=info.query_engine_name,
                    description=info.query_engine_desc
                )
            )
            
        self.add_tools(new_tools)
        
    
    @contextmanager
    def _temporary_context(self) -> Generator[None, Any, Any]:
        if self.memory is None:
            yield
            return
        
        injected: list[ChatMessage] = []

        def inject(prompt: str | None) -> None:
            if not prompt:
                return
            
            msg: ChatMessage = ChatMessage(
                role=MessageRole.SYSTEM, content=prompt,
                additional_kwargs={"ephemeral": True},
            )
            
            self.memory.put(msg)
            injected.append(msg)
        
        new_system_prompt: str = "\n\n".join((
            f"Current date and time: {datetime.datetime.now()}",
            f"## User Information:\n{self.user_info if self.user_info else "None"}",
            f"## Additional System Prompt:\n{self.system_prompt}"
        ))
        
        if self.character_prompt:
            new_system_prompt += f"\n\n## Use this persona while adhearing to all other rules:\n{self.character_prompt}"
            
        inject(new_system_prompt)

        try:
            yield
        finally:
            messages: list[ChatMessage] = self.memory.get_all()
            for msg in reversed(injected):
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].role != MessageRole.SYSTEM:
                        continue
                    if messages[i].content != msg.content:
                        continue
                    messages.pop(i)
                    break

            
    def load_model(self, tools: list[BaseTool] | None = None, doc_store_paths: list[str] | None = None) -> None:
        from ..tools import (
            BetterWikipediaToolSpec, WebSearchToolSpec, UnitConvertionToolSpec, 
            AutoChainedSympyMathToolSpec, SimpleMathToolSpec
        )
        
        self.llm_name = config_loader.get("Models", "BASE_MODEL")
        self.run_ollama_server(10)
        
        server_url: str = config_loader.get("URLs", "SERVER_URL")
        self.system_prompt = config_loader.get("Prompts", "SYSREM_PROMPT")
        
        self.embedding = OllamaEmbedding(
            config_loader.get("Models", "EMBEDDING_MODEL_NAME"), 
            base_url=server_url, embed_batch_size=128
        )
        
        self.model = Ollama(
            model=self.llm_name, temperature=self.llm_params.temperature,
            context_window=self.llm_params.context_window, base_url=server_url, thinking=False
        )
        
        Settings.llm = self.model
        Settings.embed_model = self.embedding
        
        self._initialize_memory()
        self.agent = FunctionAgent(
            request_timeout=360.0, tools=self.get_tools(), system_prompt=self.system_prompt
        )
        
        self.tool_vector_store = VectorStoreIndex(
            [], vector_store=None, embed_model=self.embedding
        )
        
        self.add_tools(BetterWikipediaToolSpec(self.tool_vector_store, self.tool_text_splitter).to_tool_list())
        self.add_tools(WebSearchToolSpec(self.tool_vector_store, self.tool_text_splitter).to_tool_list())
        self.add_tools(UnitConvertionToolSpec().to_tool_list())
        self.add_tools(AutoChainedSympyMathToolSpec().to_tool_list())
        self.add_tools(SimpleMathToolSpec().to_tool_list())
        
        if tools is not None:
            self.add_tools(tools)
            
        if doc_store_paths is not None:
            self.add_doc_store_query_engine(doc_store_paths)
    
    def _initialize_memory(self) -> None:
        self.memory: Memory = Memory.from_defaults(token_limit=self.llm_params.history_tokens)
        
    def clear_memory(self) -> None:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        self.memory.reset()
        
    def clear_vector_store(self) -> None:
        if self.vector_store is None:
            return
        self.vector_store = None
        self.added_nodes.clear()
        
    def stop(self) -> None:
        with self._stop_lock:
            self._stop_flag = True

    def _check_stop(self) -> bool:
        with self._stop_lock:
            return self._stop_flag
        
    async def agenerate_title(self, initial_prompt: str) -> str:
        assert self.model is not None, "self.model of ChatModel must be set before generating a title."
        
        system_instruction: str = (
            "Your only task is to generate a concise title summarizing the user's initial prompt. "
            "The title MUST be strictly between 3 and 6 words long. "
            "Output ONLY the raw title text. Do not include quotes, prefixes, punctuation, or conversational text."
        )
        
        try:
            messages: list[ChatMessage] = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_instruction),
                ChatMessage(role=MessageRole.USER, content=initial_prompt)
            ]
            
            # Query the LLM directly to avoid agent overhead and tool usage
            response: ChatResponse = await self.model.achat(messages)
            title: str = response.message.content.strip(' "\'\n\r\t')
            
            # Safety cleanup: ensure it doesn't exceed word counts if the LLM misbehaves
            words: list[str] = title.split()
            if len(words) > 6:
                return " ".join(words[:6])
            elif not words:
                return "New Conversation"
                
            return title
            
        except Exception as e:
            self.error_flag = e
            return "New Conversation"

    def generate_title(self, initial_prompt: str) -> str:
        nest_asyncio.apply()
        
        async def func() -> str:
            return await self.agenerate_title(initial_prompt)
            
        return asyncio.run(func())
        
    async def astream_prompt(self, prompt: str, tool_wrap: Callable[[str], str] | None = None) -> AsyncGenerator[str, None]:
        self._stop_flag = False
        with self._temporary_context():
            handler: WorkflowHandler = self.agent.run(user_msg=prompt, memory=self.memory)
            event_stream: AsyncGenerator[Event, None] = handler.stream_events()
            try:
                async for event in event_stream:
                    if self._check_stop():
                        await handler.cancel_run()
                        break
                    
                    if isinstance(event, AgentStream):
                        yield event.delta
                    elif isinstance(event, ToolCall):
                        if tool_wrap:
                            yield tool_wrap(event.tool_name)
                        yield f"\nTool Called: {event.tool_name}\n"
            finally:
                await event_stream.aclose()
                try: 
                    await handler
                except:
                    pass
    
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
    
    
    

        
