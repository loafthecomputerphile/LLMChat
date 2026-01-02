from __future__ import annotations
from typing import TYPE_CHECKING, AsyncGenerator
import gzip, gc

from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field, ConfigDict
import dill

from ..chat_model import ChatModel
from .profile import BaseProfile, ChatHistory

if TYPE_CHECKING:
    from llama_index.core.tools import BaseTool
    from ..chat_model import ModelParams
    
    
    
"""
using the code below make me a flask api to searve a chat UI based on the class.

there will be a sidebar that has past histories that when clicked would load load the messages into chat. uses:
    load_history: loads new history
    get_histories: for all histories
    get_history_messages: past history messages to replace main window content
    
the main section is the chat app itself that allows adding documents. this would use:
    add_documents: adds documents
    
there will be an input box whuch the user will input their prompt and return a streamed response:
    send_message
"""


class BaseChatSession(BaseModel):
    model: ChatModel
    user: BaseProfile
    in_session: bool = Field(False, init=False)
    history_loaded: bool = Field(False, init=False)
    history_path: str | None = Field(None, init=False)
    history: ChatHistory = Field(default_factory=ChatHistory, init=False)
    
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    
    def get_histories(self) -> list[str]:
        return self.user.history_names
    
    def send_message(self) -> AsyncGenerator[str, None]:
        assert self.in_session, "session has not been started"
        return self.model.astream_prompt
    
    def start_session(self, params: ModelParams, tools: list[BaseTool] | None = None) -> None:
        assert not self.in_session, "session has already started"
        
        self.model.set_user_info(self.user.info)
        self.model.load_parameters(params)
        self.model.load_model(tools)
        self.in_session = True
        
    def end_session(self) -> None:
        assert self.in_session, "session has not been started"
        
        if self.history_loaded:
            self.save_history()
            self.history_loaded = False
            
        self.user.save_profile()
        
        self.model.clear_memory()
        self.model.kill()
        self.in_session = False
        
    def new_history(self, name: str) -> str | None:
        assert self.in_session, "session has not been started"
        self.user.add_new_history(name) 
        
    def set_character_instruction(self, instruction: str) -> None:
        assert self.in_session, "session has not been started"
        if not instruction:
            return
        self.history.character_instruction = instruction
        self.model.set_character_prompt(instruction)
        
    def set_user_information(self, info: str) -> None:
        assert self.in_session, "session has not been started"
        if not info:
            return
        self.user.info = info
        self.model.set_user_info(info)
    
    def save_history(self) -> None:
        assert self.in_session, "session has not been started"
        self.history.messages = self.model.memory.get_all()
        
        with gzip.open(self.history_path, "wb") as file:
            dill.dump(self.history.model_dump(), file)
            
    def load_history(self, name: str) -> None:
        assert self.in_session, "session has not been started"
        assert name in self.user.history_files, "name bust be an saved chat history"
        
        if self.history_loaded:
            self.save_history()
        
        self.history_path = self.user.history_files[name]
        with gzip.open(self.history_path, "rb") as file:
            self.history = ChatHistory.model_validate(dill.load(file))
        
        if self.history.character_instruction:
            self.model.set_character_prompt(self.history.character_instruction)
            
        self.model.clear_memory()  
        self.model.add_memory(self.get_history_messages())
        self.history_loaded = True
            
    def get_history_messages(self) -> list[ChatMessage]:
        return self.history.messages
    
    def stop_prompt(self) -> None:
        assert self.in_session, "session has not been started"
        self.model.stop()
        
    def add_documents(self, docs_paths: list[str]) -> None:
        assert self.in_session, "session has not been started"
        self.model.add_documents(docs_paths)