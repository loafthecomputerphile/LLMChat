from __future__ import annotations
from typing import Any, TYPE_CHECKING


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from llama_index.core.node_parser import LangchainNodeParser

from .extractors import *
from .types import ExtractorBundle

if TYPE_CHECKING:
    from llama_index.core import Document
    from llama_index.core.node_parser import TextSplitter

BASE_CHUNK_SIZE: int = 512
BASE_OVERLAP: int = 128



EXT_TO_LANG = {
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".php": Language.PHP,
    ".proto": Language.PROTO,
    ".py": Language.PYTHON,
    ".rst": Language.RST,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".tex": Language.LATEX,
    ".html": Language.HTML,
    ".sol": Language.SOL,
    ".cs": Language.CSHARP,
    ".cbl": Language.COBOL,
    ".c": Language.C,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
    ".ex": Language.ELIXIR,
    ".ps1": Language.POWERSHELL,
    ".vbs": Language.VISUALBASIC6
}


class CodeTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, extension: str, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        if extension in EXT_TO_LANG:
            separators: list[str] = self.get_separators_for_language(EXT_TO_LANG[extension])
        else:
            separators: list[str] = ["\n\n", "\n", " ", ""]
        super().__init__(separators=separators, **kwargs)
        
          
python_splitter     : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".py", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
cpp_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".cpp", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
go_splitter         : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".go", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
java_splitter       : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".java", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
kotlin_splitter     : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".kt", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
js_splitter         : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".js", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
ts_splitter         : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".ts", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
php_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".php", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
proto_splitter      : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".proto", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
rst_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".rst", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
ruby_splitter       : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".rb", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
rust_splitter       : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".rs", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
scala_splitter      : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".scala", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
swift_splitter      : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".swift", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
markdown_splitter   : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".md", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
latex_splitter      : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".tex", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
html_splitter       : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".html", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
sol_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".sol", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
csharp_splitter     : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".cs", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
cobol_splitter      : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".cbl", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
c_splitter          : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".c", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
lua_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".lua", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
#perl_splitter       : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".pl", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
haskell_splitter    : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".hs", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
elixir_splitter     : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".ex", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
powershell_splitter : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".ps1", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))
vb6_splitter        : LangchainNodeParser = LangchainNodeParser(CodeTextSplitter(".vbs", chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))

recursive_splitter  : LangchainNodeParser = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=BASE_CHUNK_SIZE, chunk_overlap=BASE_OVERLAP))



bundles: list[ExtractorBundle] = [
    ExtractorBundle(
        name="excel",
        extractor=excel_extractor,
        splitter=recursive_splitter,
        types=["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"],
    ),
    ExtractorBundle(
        name="word",
        extractor=word_extractor,
        splitter=recursive_splitter,
        types=["docx", "doc", "odf"],
    ),
    ExtractorBundle(
        name="pdf",
        extractor=pdf_extractor,
        splitter=recursive_splitter,
        types=["pdf"],
    ),
    ExtractorBundle(
        name="presentation",
        extractor=presentation_extractor,
        splitter=recursive_splitter,
        types=["ppt", "pptx", "odp"],
    ),
    ExtractorBundle(
        name="epub",
        extractor=epub_extractor,
        splitter=recursive_splitter,
        types=["epub"],
    ),
    ExtractorBundle(
        name="python",
        extractor=plain_extractor,
        splitter=python_splitter,
        types=["py"],
    ),
    ExtractorBundle(
        name="cpp",
        extractor=plain_extractor,
        splitter=cpp_splitter,
        types=["cpp"],
    ),
    ExtractorBundle(
        name="go",
        extractor=plain_extractor,
        splitter=go_splitter,
        types=["go"],
    ),
    ExtractorBundle(
        name="java",
        extractor=plain_extractor,
        splitter=java_splitter,
        types=["java"],
    ),
    ExtractorBundle(
        name="kotlin",
        extractor=plain_extractor,
        splitter=kotlin_splitter,
        types=["kt"],
    ),
    ExtractorBundle(
        name="javascript",
        extractor=plain_extractor,
        splitter=js_splitter,
        types=["js"],
    ),
    ExtractorBundle(
        name="typescript",
        extractor=plain_extractor,
        splitter=ts_splitter,
        types=["ts"],
    ),
    ExtractorBundle(
        name="php",
        extractor=plain_extractor,
        splitter=php_splitter,
        types=["php"],
    ),
    ExtractorBundle(
        name="proto",
        extractor=plain_extractor,
        splitter=proto_splitter,
        types=["proto"],
    ),
    ExtractorBundle(
        name="rst",
        extractor=plain_extractor,
        splitter=rst_splitter,
        types=["rst"],
    ),
    ExtractorBundle(
        name="ruby",
        extractor=plain_extractor,
        splitter=ruby_splitter,
        types=["rb"],
    ),
    ExtractorBundle(
        name="rust",
        extractor=plain_extractor,
        splitter=rust_splitter,
        types=["rs"],
    ),
    ExtractorBundle(
        name="scala",
        extractor=plain_extractor,
        splitter=scala_splitter,
        types=["scala"],
    ),
    ExtractorBundle(
        name="swift",
        extractor=plain_extractor,
        splitter=swift_splitter,
        types=["swift"],
    ),
    ExtractorBundle(
        name="markdown",
        extractor=plain_extractor,
        splitter=markdown_splitter,
        types=["md"],
    ),
    ExtractorBundle(
        name="latex",
        extractor=plain_extractor,
        splitter=latex_splitter,
        types=["tex"],
    ),
    ExtractorBundle(
        name="html",
        extractor=plain_extractor,
        splitter=html_splitter,
        types=["html", "htm"],
    ),
    ExtractorBundle(
        name="sol",
        extractor=plain_extractor,
        splitter=sol_splitter,
        types=["sol"],
    ),
    ExtractorBundle(
        name="csharp",
        extractor=plain_extractor,
        splitter=csharp_splitter,
        types=["cs"],
    ),
    ExtractorBundle(
        name="cobol",
        extractor=plain_extractor,
        splitter=cobol_splitter,
        types=["cbl", "cob", "cobol"],
    ),
    ExtractorBundle(
        name="c",
        extractor=plain_extractor,
        splitter=c_splitter,
        types=["c"],
    ),
    ExtractorBundle(
        name="lua",
        extractor=plain_extractor,
        splitter=lua_splitter,
        types=["lua"],
    ),
    ExtractorBundle(
        name="haskell",
        extractor=plain_extractor,
        splitter=haskell_splitter,
        types=["hs"],
    ),
    ExtractorBundle(
        name="elixir",
        extractor=plain_extractor,
        splitter=elixir_splitter,
        types=["ex", "exs"],
    ),
    ExtractorBundle(
        name="powershell",
        extractor=plain_extractor,
        splitter=powershell_splitter,
        types=["ps1"],
    ),
    ExtractorBundle(
        name="vb6",
        extractor=plain_extractor,
        splitter=vb6_splitter,
        types=["vbs", "bas", "cls", "frm"],
    ),

    ExtractorBundle(
        name="plain",
        extractor=plain_extractor,
        splitter=recursive_splitter,
        types=["txt", "csv", "env", "ini", "cfg", "log", "yml", "yaml", "toml"],
    ),
]
