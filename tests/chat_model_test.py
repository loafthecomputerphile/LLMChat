from __future__ import annotations
from typing import TYPE_CHECKING
import re

import pytest

from src.chat_model import ChatModel, ModelParams
from src.extractors import *

if TYPE_CHECKING:
    from llama_index.core import Document
    

def strip_all_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)



