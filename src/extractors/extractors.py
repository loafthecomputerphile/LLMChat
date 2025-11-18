import os

from .extraction_utils import bytes_to_megabytes, get_mimetype, set_pandoc_env
from ..flags import ExtractionErrors, EXTRACTION_ERROR_FLAG

from llama_index.core import Document


__all__ = [
    "excel_extractor", "plain_extractor", "word_extractor", "pdf_extractor", "presentation_extractor",
    "epub_extractor", "ExtractionErrors", "EXTRACTION_ERROR_FLAG", "FILE_SIZE_LIMIT"
]

set_pandoc_env()
 

ocr_model = None
FILE_SIZE_LIMIT: int = 25


def excel_extractor(file_path: str) -> list[Document] | str:
    """
    supports xls, xlsx, xlsm, xlsb, odf, ods, odt
    """
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    
    import pandas as pd
    
    result: list[Document] = []
    
    spread_sheet: pd.DataFrame | dict[str, pd.DataFrame] = pd.read_excel(file_path, sheet_name=None)
    metadata: dict[str, str | int] = {
        "file_name":file_path.rsplit(".", 1)[0],
        "sheet_index":0
    }
    
    if not isinstance(spread_sheet, dict):
        return [Document(text=spread_sheet.to_string(), metadata=metadata)]
    
    for i, (name, data) in enumerate(spread_sheet.items()):
        meta = dict(metadata)
        meta["sheet_index"] = i
        meta["sheet_name"] = name
        result.append(Document(text=data.to_string(), metadata=meta))
    
    return result


def plain_extractor(file_path: str) -> list[Document] | str:
    """
    supports all other text based files
    """
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    text: str = ""
    with open(file_path, "r") as file:
        text: str = file.read()
    
    return [
        Document(text=text, metadata={
            "file_name":file_path.rsplit(".", 1)[0],
            "file_path": file_path, "file_type":file_path.split(".")[-1]
        })
    ]


def word_extractor(file_path: str) -> list[Document] | str:
    """
    supports docx, doc and odf
    """
    
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    import pypandoc as pypd
    text: str = pypd.convert_file(file_path, 'plain', sandbox=True)
    
    return [
        Document(text=text, metadata={
            "file_name":file_path.rsplit(".", 1)[0],
            "file_path": file_path, "file_type":get_mimetype(file_path)
        })
    ]


def pdf_extractor(file_path: str) -> list[Document] | str:
    
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    import fitz
    
    doc: fitz.Document = fitz.open(file_path)
    page: fitz.Page = None
    
    text: str = ""
        
    for page in map(lambda i: doc.load_page(i), range(len(doc))):
        text += page.get_text() + "\n"
        
    return [
        Document(
            text=text, metadata={
                "file_name":file_path.rsplit(".", 1)[0], "file_path":file_path,
                "file_type":get_mimetype(file_path)
            }
        )
    ]
        
    
def presentation_extractor(file_path: str) -> list[Document] | str:
    
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) == FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    from officeparserpy import parse_office
    
    try:
        return [
            Document(text=parse_office(file_path), metadata={
                "file_path":file_path, "file_type": get_mimetype(file_path)
            })
        ]
    except Exception as e:
        pass
    
    return ExtractionErrors.UNKNOWN_ERROR
    
        
def epub_extractor(file_path: str) -> list[Document] | str:
    return word_extractor(file_path)
    
    
    

