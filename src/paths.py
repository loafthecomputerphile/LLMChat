import platform
from pathlib import Path


ROOT: Path = Path(__file__).parent.parent

DATA_FOLDER: Path = ROOT / "data" 

BINARIES_FOLDER: Path = DATA_FOLDER / "bin"
PANDOC_EXE: Path = BINARIES_FOLDER / "pandoc" / "pandoc.exe"
PORTABLE_OLLAMA: Path = BINARIES_FOLDER / "ollama" / "ollama_portable.bat"
PORTABLE_OLLAMA_EXE: Path = BINARIES_FOLDER / "ollama"


if platform.system().lower() in ("linux", "darwin"):
    PORTABLE_OLLAMA = BINARIES_FOLDER / "ollama" / "ollama_portable.sh"
    PANDOC_EXE = BINARIES_FOLDER / "pandoc" / "bin" / "pandoc"
    
if platform.system().lower() == "linux":
    PORTABLE_OLLAMA_EXE = PORTABLE_OLLAMA_EXE / "bin"


MODELS_FOLDER: Path = DATA_FOLDER / "models"
EMBEDDINGS_FOLDER: Path = MODELS_FOLDER / "embeddings"
LLM_FOLDER: Path = MODELS_FOLDER / "llm"



