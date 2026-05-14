from __future__ import annotations
import os, uuid
from typing import Any, TYPE_CHECKING

from .extraction_utils import bytes_to_megabytes, get_mimetype, set_pandoc_env
from ..flags import ExtractionErrors, EXTRACTION_ERROR_FLAG
from ..paths import OCR_MODELS_FOLDER

from llama_index.core import Document

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from doctr.io import Document as DoctrDocument

__all__ = [
    "excel_extractor", "plain_extractor", "word_extractor", "pdf_extractor", "presentation_extractor",
    "epub_extractor", "ExtractionErrors", "EXTRACTION_ERROR_FLAG", "FILE_SIZE_LIMIT"
]

set_pandoc_env()

os.environ["DOCTR_CACHE_DIR"] = str(OCR_MODELS_FOLDER)

MODEL_KWARGS: dict[str, Any] = {
    "reco_arch":"crnn_vgg16_bn",
    "det_arch":"fast_base",
    "pretrained":True,
    "reco_bs":256, 
    "det_bs":2
}


def pdf_to_text(file_path: str, start_page: int = 0, end_page: int = -1) -> str:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    
    ocr_model = ocr_predictor(**MODEL_KWARGS)
    
    doc: list[NDArray] = DocumentFile.from_pdf(file_path)

    doc = doc[
        start_page: len(doc) 
        if end_page == -1 or end_page is None else 
        end_page
    ]
    
    result: DoctrDocument = ocr_model(doc)
    return result.render()


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
        return [Document(text=spread_sheet.to_string(), id_=str(uuid.uuid4()), metadata=metadata)]
    
    for i, (name, data) in enumerate(spread_sheet.items()):
        meta = dict(metadata)
        meta["sheet_index"] = i
        meta["sheet_name"] = name
        result.append(Document(text=data.to_string(), id_=str(uuid.uuid4()), metadata=meta))
    
    return result


def plain_extractor(file_path: str) -> list[Document] | str:
    """
    supports all other text based files
    """
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    text: str = ""
    with open(file_path, "r", encoding="utf8") as file:
        text: str = file.read()
    
    return [
        Document(text=text, id_=str(uuid.uuid4()), metadata={
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
        Document(text=text, id_=str(uuid.uuid4()), metadata={
            "file_name":file_path.rsplit(".", 1)[0],
            "file_path": file_path, "file_type":get_mimetype(file_path)
        })
    ]


def pdf_extractor(file_path: str) -> list[Document] | str:
    
    global FILE_SIZE_LIMIT
    
    if bytes_to_megabytes(os.path.getsize(file_path)) > FILE_SIZE_LIMIT:
        return ExtractionErrors.FILE_SIZE_LIMIT
    
    text: str = pdf_to_text(file_path)
    
    #import fitz
    
    
    # doc: fitz.Document = fitz.open(file_path)
    # page: fitz.Page = None
    # 
    # text: str = ""
    #     
    # for page in map(lambda i: doc.load_page(i), range(len(doc))):
    #     text += page.get_text() + "\n"
    
    return [
        Document(
            text=text, id_=str(uuid.uuid4()), metadata={
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
            Document(text=parse_office(file_path), id_=str(uuid.uuid4()), metadata={
                "file_path":file_path, "file_type": get_mimetype(file_path)
            })
        ]
    except Exception as e:
        pass
    
    return ExtractionErrors.UNKNOWN_ERROR
    
        
def epub_extractor(file_path: str) -> list[Document] | str:
    return word_extractor(file_path)
    
    
    

