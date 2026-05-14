from __future__ import annotations
import os.path, json, argparse, shutil
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from pydantic import BaseModel
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from ..paths import KNOWLEDGE_FOLDER
from ..extractors.extraction_router import make_default_router


if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    
    from ..extractors.extraction_router import ExtractionRouter
    
    
EMBEDDING_DIMENSIONS: int = 768


class IndexInfo(BaseModel):
    query_engine_name: str
    query_engine_desc: str
    documents: list[str]
    

class KnowledgeManager:
        
    __slots__ = ("faiss_index")    
    
    def __init__(self) -> None:
        self.faiss_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(EMBEDDING_DIMENSIONS)
        
    def get_knowledge_bases(self) -> list[str]:
        results: list[str] = []
        for item in filter(lambda x: not x.is_file(), KNOWLEDGE_FOLDER.iterdir()):
            results.append(item.name)
        return results
    
    def delete_knowledge_base(self, name: str) -> None:
        dir_path: str = f"{str(KNOWLEDGE_FOLDER)}/{name}"
        assert os.path.exists(dir_path), "the knowledge base '{name}' does not exist"
        
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error deleting knowledge base: {e}")
        
    def make_knowledge(self, documents_path: str) -> list[str] | None:
        assert os.path.exists(documents_path), "folder path must exist"
        
        info_path: str = f"{documents_path}/info.json"
        assert os.path.exists(info_path), "file `info.json` must be present"
        
        with open(info_path, 'r') as file:
            info_data: dict[str, str] = json.load(file)
            
            assert "name" in info_data, "info.json is missing key `name`"
            assert "description" in info_data, "info.json is missing key `description`"
            
            info: IndexInfo = IndexInfo(
                info_data["name"], info_data["description"], []
            )
        
        extractor: ExtractionRouter = make_default_router()
        valid_keys: set[str] = set(extractor.file_map.keys())
        
        index: VectorStoreIndex = self.make_new_index()
        raw_docs: list[str] = list(
            filter(lambda x: x.is_file() and x.suffix in valid_keys, Path(documents_path).iterdir())
        )
        doc_names: list[str] = [Path(doc).name for doc in raw_docs]
        
        
        errors: list[str] = self.add_knowledge(index, raw_docs, extractor)
        if len(errors) > 0:
            error_names: list[str] = [Path(error).name for error in errors]
            for name in error_names:
                doc_names.remove(name)
                
            info.documents = doc_names
            self.save_knowledge(index, info, Path(documents_path).name)
            return errors
        
        info.documents = doc_names
        self.save_knowledge(index, info, Path(documents_path).name)
        
    def display_knowledge_base_info(self, name: str) -> str:
        with open(f"{str(KNOWLEDGE_FOLDER)}/{name}/index_info", "r") as file:
            info: IndexInfo = IndexInfo.model_validate_json(file.read())
            return f"name: {info.query_engine_name}\ndescription: {info.query_engine_desc}\ndocuments: {info.documents}"
        return f"name: None\ndescription: None\ndocuments: None"
        
    def load_index(self, name: str) -> tuple[VectorStoreIndex, IndexInfo]:
        vector_store: BasePydanticVectorStore = FaissVectorStore.from_persist_dir(f"{str(KNOWLEDGE_FOLDER)}/{name}")
        storage_context: StorageContext = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=f"{str(KNOWLEDGE_FOLDER)}/{name}"
        )
        
        with open(f"{str(KNOWLEDGE_FOLDER)}/{name}/index_info", "r") as file:
            info: IndexInfo = IndexInfo.model_validate_json(file.read())
            
        return load_index_from_storage(storage_context=storage_context), info
        
    def save_knowledge(self, name: str, index: VectorStoreIndex, info: IndexInfo) -> None:
        save_path: str = f"{str(KNOWLEDGE_FOLDER)}/{name}"
        index.storage_context.persist(save_path)
        with open(f"{save_path}/index_info", "w") as file:
            file.write(info.model_dump_json(indent=2))
        
    def make_new_index(self) -> VectorStoreIndex:
        vector_store: BasePydanticVectorStore = FaissVectorStore(faiss_index=self.faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            [], storage_context=storage_context
        )
        
    def add_knowledge(index: VectorStoreIndex, doc_paths: list[str], extraction_router: ExtractionRouter | None = None) -> list[str]:
        if extraction_router is None:
            extraction_router = make_default_router()
        
        errors: list[str] = []
        results: list[list[BaseNode] | str]
        deletion_indexes: list[int] = []
        path_length: int = len(doc_paths)
        
        if path_length == 0: 
            return
        
        if path_length > 1:
            with ThreadPoolExecutor(4) as pool:
                results = pool.map(extraction_router.extract, doc_paths)

            for i, _ in filter(lambda x: isinstance(x[1], str), enumerate(results)):
                errors.append(doc_paths[i])
                deletion_indexes.append(i)
            
            if deletion_indexes:
                for index in deletion_indexes[::-1]: 
                    del results[index]               
            
        elif path_length == 1:
            results = [extraction_router.extract(doc_paths[0])]
            if isinstance(results[0], str): 
                errors = [doc_paths[0]]
                
        for res in results:
            index.insert_nodes(res)
            
        return errors


def console_knowledge_manager() -> None:
    knowledge_manager: KnowledgeManager = KnowledgeManager()
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="create a knowledge base from a folder of documents"
    )
    
    parser.add_argument("action", choices=["make", "del", "info", "list"])
    
    parser.add_argument("path-or-name", required=True)
    
    action: str = parser.action
    name: str = parser.path_or_name
    
    
    if action == "del":
        return knowledge_manager.delete_knowledge_base(name)
        
    if action == "make":
        results: list[str] | None = knowledge_manager.make_knowledge(name)
        if results is None: return
        return print(f"Could not process files: {', '.join(results)}")
        
    if action == "info":
        return print(knowledge_manager.display_knowledge_base_info(name))
    
    if action == "list":
        names: list[str] = knowledge_manager.get_knowledge_bases()
        display: str = f"Knowledge Bases:\n{'\n'.join(names)}"
        return print(display)
           

    