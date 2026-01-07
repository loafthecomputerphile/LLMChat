from __future__ import annotations
import subprocess, gc, datetime
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from pydantic import BaseModel

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import MessageRole
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import Memory

from ..paths import PORTABLE_OLLAMA
from ..variables import ConfigLoader

if TYPE_CHECKING:
    from subprocess import STARTUPINFO
    
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent   
    from llama_index.core.query_engine import BaseQueryEngine


class ModelParams(BaseModel):
    temperature: float
    context_window: int
    rag_top_k: int
    history_tokens: int
    long_term_memory: bool
    long_term_tokens: int
    top_k_memory: int
    

class MemoryMod:
    memory: Memory
    user_info: str
    system_prompt: str
    character_prompt: str
    
    def add_memory(self, messages: list[ChatMessage]) -> None:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        self.memory.put_messages(messages)
        
    def clear_memory(self) -> None:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        self.memory.reset()
        
    def get_full_memory(self) -> list[ChatMessage]:
        assert self.memory is not None, "self.memory of ChatModel must be set"
        return self.memory.get_all()
    
    def _initialize_memory(self, llm_params: ModelParams) -> None:
        memory: Memory = Memory.from_defaults(token_limit=llm_params.history_tokens)
        
        self.memory = memory
        
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


class ToolMod:
    agent: FunctionAgent
    vector_store: VectorStoreIndex
        
    def delete_tools(self, names: list[str], update_tools: bool = True) -> None:
        assert self.agent is not None, "self.agent of ChatModel must be set"
        
        self.tools = [t for t in self.tools if t.metadata.name not in names]
        
        if update_tools:
            self.agent.tools = self.tools 
    
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
            llm=Settings.llm, similarity_top_k=top_k
        )
        
        query_tool: QueryEngineTool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="document_data_tool",
            description=description
        )

        return query_tool
    

class CMDMod:
    config_loader: ConfigLoader
    
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
            self.config_loader.get("Models", key) 
            for key in ("EMBEDDING_MODEL_NAME", "BASE_MODEL")
        ]
        
        for model_name in models:
            try:
                run_cmd(model_name)
                gc.collect()
            except Exception as e:
                continue
