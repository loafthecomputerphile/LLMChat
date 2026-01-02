from __future__ import annotations
from typing import Callable
from base64 import b64encode
import gzip

import dill
from pathlib import Path
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field, ConfigDict

CHAT_DATA: Path = Path.home() / "Documents" / "LLMChat"
PROFILE_PATH: Callable[[str], Path] = lambda name: CHAT_DATA / name
HISTORY_PATH: Callable[[str], Path] = lambda name: PROFILE_PATH(name) / "histories"


class ChatHistory(BaseModel):
    character_instruction: str = Field("")
    messages: list[ChatMessage] = Field(default_factory=list)
    
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    

class BaseProfile:

    def __init__(self, username: str) -> None:
        self.username: str = username
        self.info: str = ""
        self.history_files: dict[str, str] = {}
        
    def get_history_path(self, name: str) -> str | None:
        return self.history_files.get(name, None)
        
    def add_new_history(self, name: str) -> None:
        path: str = str(
            HISTORY_PATH(self.username) / 
            b64encode(name.encode('utf-8')).decode('utf-8')
        )
        
        with gzip.open(path, "wb") as file: 
            dill.dump(ChatHistory().model_dump(), file)
        
        self.history_files[name] = path
        
    def delete_history(self, name: str) -> None:
        if name not in self.history_files:
            return
        
        Path(self.history_files[name]).unlink(True)
        del self.history_files[name]
    
    @property
    def history_names(self) -> list[str]:
        return list(self.history_files.keys())
    
    def save_profile(self) -> None:
        with gzip.open(str(PROFILE_PATH(self.username) / "profile"), "wb") as file:
            dill.dump(self, file)
    
    @classmethod
    def load_profile(cls, name: str) -> BaseProfile:
        with gzip.open(str(PROFILE_PATH(name) / "profile"), "rb") as file:
            return dill.load(file)