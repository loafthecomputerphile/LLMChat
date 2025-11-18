from __future__ import annotations
from typing import TYPE_CHECKING
import re

import pytest

from src.extractors import *

if TYPE_CHECKING:
    from llama_index.core import Document