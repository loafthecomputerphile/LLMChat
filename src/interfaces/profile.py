from __future__ import annotations
from typing import Callable
from base64 import b64encode
import gzip

import orjson
from pathlib import Path
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field, ConfigDict

from src.utils.persistent_model import TarPersistentModel

CHAT_DATA: Path = Path.home() / "Documents" / "LLMChat"
PROFILE_PATH: Callable[[str], Path] = lambda name: CHAT_DATA / name
HISTORY_PATH: Callable[[str], Path] = lambda name: PROFILE_PATH(name) / "histories"


class ChatHistory(TarPersistentModel):
    character_instruction: str = Field("")
    messages: list[ChatMessage] = Field(default_factory=list)
    
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)


class BaseProfile(BaseModel):
    username: str
    info: str = Field(default="", init=False)
    history_files: dict[str, str] = Field(default_factory=dict, init=False)

    def get_history_path(self, name: str) -> str | None:
        return self.history_files.get(name, None)
        
    def add_new_history(self, name: str) -> None:
        path: str = str(
            HISTORY_PATH(self.username) / 
            b64encode(name.encode('utf-8')).decode('utf-8')
        )
        
        history: ChatHistory = ChatHistory()
        history.attach_tar(path)
        history.close()
        
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
        (PROFILE_PATH(self.username) / "profile").write_bytes(
            orjson.dumps(
                self.model_dump()
            )
        )
    
    @classmethod
    def load_profile(cls, name: str) -> BaseProfile:
        return cls.model_validate(
            orjson.loads(
                str(PROFILE_PATH(name) / "profile")
            )
        )
        
