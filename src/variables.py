from typing import Any
import os.path

from pathlib import Path

from .paths import CONFIG


class ConfigLoader:
    
    def __init__(self) -> None:
        self.path: Path = CONFIG
        self.data: dict[str, Any] = {}
        
    def load(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError()
        
        import tomli
        with open(self.path, "rb") as file:
            self.data = tomli.load(file)
        
    def get(self, *args: str) -> Any:
        result: Any =  self.data
        for arg in args:
            result = result[arg]
        return result