from __future__ import annotations
from typing import TYPE_CHECKING
import re

import pytest

from src.extractors import *

if TYPE_CHECKING:
    from langchain_core.documents import Document


target_test: str = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non sem dapibus, porttitor nisi eget, varius justo. Vestibulum volutpat tristique suscipit. Nunc orci quam, lobortis ac facilisis sit amet, condimentum eget ante. Sed dignissim eleifend elementum. In egestas purus dolor, quis elementum risus egestas vel. Donec feugiat nulla erat, vel hendrerit turpis ullamcorper in. Sed at iaculis enim. Pellentesque vestibulum pharetra urna ut faucibus. Sed porttitor nibh id risus mattis molestie. Suspendisse fermentum urna eget mollis pulvinar. Ut in consectetur nisi. Donec libero quam, vestibulum ut turpis et, tincidunt blandit urna. Aliquam vitae semper lacus. Morbi viverra lobortis felis ut vulputate. Suspendisse porta vel odio quis sodales. Vivamus luctus massa quis neque posuere iaculis. Nunc dictum mattis justo. Vestibulum vitae ex ante. Nunc varius ac velit tincidunt tristique. Nullam odio elit, sagittis quis leo a, porta lacinia arcu. Vestibulum erat velit, pretium vel iaculis a, vestibulum eu turpis. Aenean semper sapien id tristique fringilla. Morbi quis sapien arcu. Proin elementum massa eu velit commodo commodo. Suspendisse porttitor ante diam, at semper turpis egestas ut. Vestibulum imperdiet enim fermentum mollis sagittis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur lacus augue, scelerisque ut tortor a, tincidunt porta mauris. Nullam dignissim, neque dignissim euismod euismod, elit nibh fringilla sem, ac laoreet felis nulla ut ante. Vivamus efficitur lectus massa, vitae tristique velit venenatis eget. Praesent augue diam, consectetur volutpat cursus id, pharetra ut ante. Donec pretium erat in tellus viverra lobortis sed ac mauris. Integer efficitur nisi id venenatis finibus. Integer consectetur lacus sed ultricies interdum. Suspendisse faucibus, dui semper efficitur consectetur, est nisi aliquam dui, eget varius neque risus id arcu. Sed varius massa ut mauris vehicula, nec lacinia lacus bibendum. Aliquam a ipsum euismod, varius ipsum tristique, vulputate diam. Ut consectetur purus non dui ultrices elementum. Morbi ut auctor nisi. Nulla velit libero, elementum id consequat eget, consequat in neque. Aliquam metus felis, commodo sit amet dignissim in, lacinia vehicula lacus.
"""


def strip_all_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_excel_extractor() -> None:
    
    target: str = """                    Age
0              Under 18
1                 18-24
2                 25-34
3                 35-44
4                 45-54
5                 55-64
6           65 or Above
7  Prefer Not to Answer

                   Age 2
0              Under 18
1                 18-24
2                 25-34
3                 35-44
4                 45-54
5                 55-64
6           65 or Above
7  Prefer Not to Answer
    
    """
    data: list[Document] = excel_extractor("tests\\docs\\age.xlsx")
    
    test_data: str = "\n\n".join(doc.page_content for doc in data)

    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)



def test_csv_extractor() -> None:
    
    target: str = """Age,Age 2
Under 18,Under 18
18-24,18-24
25-34,25-34
35-44,35-44
45-54,45-54
55-64,55-64
65 or Above,65 or Above
Prefer Not to Answer,Prefer Not to Answer
    """
    
    data: list[Document] = plain_extractor("tests\\docs\\age.csv")
    
    test_data = "\n\n".join(doc.page_content for doc in data)
    
    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)
    
    
    
def test_pdf_extractor() -> None:
    
    global target_test
    
    target: str = target_test
    
    data: list[Document] = pdf_extractor("tests\\docs\\test.pdf")
    
    test_data = "\n\n".join(doc.page_content for doc in data)
    
    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)
    
    
    
def test_epub_extractor() -> None:
    
    global target_test
    
    target: str = target_test
    
    data: list[Document] = epub_extractor("tests\\docs\\test.epub")
    
    test_data = "\n\n".join(doc.page_content for doc in data)
    
    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)
    
    
def test_word_extractor() -> None:
    
    global target_test
    
    target: str = target_test
    
    data: list[Document] = word_extractor("tests\\docs\\test.docx")
    
    test_data = "\n\n".join(doc.page_content for doc in data)
    
    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)
    
    
    
def test_presentation_extractor() -> None:
    
    global target_test
    
    target: str = target_test
    
    data: list[Document] = presentation_extractor("tests\\docs\\test.docx")
    
    test_data = "\n\n".join(doc.page_content for doc in data)
    
    assert EXTRACTION_ERROR_FLAG == ExtractionErrors.SUCCESS
    assert strip_all_ws(target) == strip_all_ws(test_data)