import os

import filetype

from ..paths import PANDOC_EXE



def get_mimetype(file_path: str) -> str | None:
    return filetype.guess_mime(file_path)


def bytes_to_megabytes(bytes: int) -> int:
    return round(bytes/(1024*1024))


def set_pandoc_env() -> None:
    os.environ.setdefault('PYPANDOC_PANDOC', PANDOC_EXE)
    