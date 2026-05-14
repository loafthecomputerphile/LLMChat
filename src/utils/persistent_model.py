from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr
)

from pathlib import Path
from typing import Iterable, Any, Generic, TypeVar, Sequence

import tarfile
import tempfile
import shutil
import orjson
import orjsonl

T = TypeVar("T")


class Syncable:
    def _sync_field(self, field_name: str) -> None:
        ...


class PersistentList(list[T], Generic[T]):

    def __init__(self, iterable: Iterable[T] = (), *, owner: Syncable | None = None, field_name: str | None = None) -> None:
        super().__init__(iterable)

        self._owner: Syncable | None = owner
        self._field_name: str | None = field_name

    def _sync(self) -> None:

        if self._owner:
            self._owner._sync_field(
                self._field_name
            )

    def append(self, item: T) -> None:
        super().append(item)
        self._sync()

    def extend(self, items: Iterable[T]) -> None:
        super().extend(items)
        self._sync()

    def insert(self, index: int, item: T) -> None:
        super().insert(index, item)
        self._sync()

    def remove(self, item: T):
        super().remove(item)
        self._sync()

    def clear(self):
        super().clear()
        self._sync()

    def pop(self, index=-1):
        value = super().pop(index)
        self._sync()
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._sync()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._sync()


class TarPersistentModel(BaseModel):

    model_config: ConfigDict = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    jsonl_fields: set[str] = set()
    
    _tar_path: Path | None = PrivateAttr(default=None)
    _workspace: Path | None = PrivateAttr(default=None)
    _field_types: dict[str, type] = PrivateAttr(default_factory=dict)
    
    def model_post_init(self, context):
        super().model_post_init(context)
        self._get_annotation_types()
    
    def _get_annotation_types(self) -> None:
        
        def get_to_base(dtype: Any) -> Any:
            if isinstance(dtype, Sequence):
                return get_to_base(dtype.__args__[0])
            return dtype
        
        for field, dtype in self.model_fields.items():
            root_dtype: type = get_to_base(dtype)
            self._field_types[field] = root_dtype
        
    def attach_tar(self, tar_path: str | Path) -> None:
        self._tar_path = Path(tar_path)
        self._workspace = Path(
            tempfile.mkdtemp(prefix="persistent_model_")
        )

        if not self._tar_path.exists():
            self._full_sync()
            
        with tarfile.open(self._tar_path, "r") as tar:
            tar.extractall(self._workspace)
            
    def save(self) -> None:
        self._rebuild_tar()
        
    def close(self) -> None:
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name.startswith("_"):
            return

        self._sync_field(name)

    def _full_sync(self) -> None:
        self._write_data_json()

        for field in self.jsonl_fields:
            self._write_jsonl_field(field)

        self._rebuild_tar()

    def _sync_field(self, field_name: str) -> None:
        if not self._workspace:
            return

        if field_name in self.jsonl_fields:
            self._write_jsonl_field(field_name)
            return self._rebuild_tar()
        
        self._write_data_json()
        self._rebuild_tar()

    def _write_data_json(self) -> None:

        data: dict[str, Any] = {}

        for field in self.model_fields:
            if field in self.jsonl_fields:
                continue

            value = getattr(self, field)

            if isinstance(value, BaseModel):
                data[field] = value.model_dump(mode="json")
                continue
            
            data[field] = value

        path: Path = self._workspace / "data.json"

        path.write_bytes(orjson.dumps(data))

    def _write_jsonl_field(self, field_name: str) -> None :

        path: Path = self._workspace / f"{field_name}.jsonl"
        value: Any = getattr(self, field_name)

        serialized: list[Any] = []

        for item in value:
            if isinstance(item, BaseModel):
                serialized.append(item.model_dump(mode="json"))
            else:
                serialized.append(item)

        orjsonl.save(path, serialized)

    def _rebuild_tar(self) -> None:
        temp_tar: Path = self._tar_path.with_suffix(".tmp")

        with tarfile.open(temp_tar, "w") as tar:
            for file in self._workspace.iterdir():
                tar.add(file, arcname=file.name)

        temp_tar.replace(self._tar_path)
        
    def append(self, field_name: str, value: Any) -> None:
        if field_name not in self.model_fields_set:
            raise AttributeError(f"{self.__class__.__name__} does not have a field {field_name}")
        
        field: Any = getattr(self, field_name)
        
        if not isinstance(field, Sequence):
            raise Exception("field {field_name} must be  a Sequence type")
        
        path: Path = self._workspace / f"{field_name}.jsonl"
        getattr(self, field_name).append(value)
        
        if isinstance(value, BaseModel):
            return orjsonl.append(path, value.model_dump(mode="json"))
        
        orjsonl.append(path, value)
            
    @classmethod
    def load(cls, tar_path: str | Path) -> TarPersistentModel:
        cls._get_annotation_types()
        tar_path: Path = Path(tar_path)

        workspace: Path = Path(
            tempfile.mkdtemp(
                prefix="persistent_model_load_"
            )
        )

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(workspace)

        data_path: Path = workspace / "data.json"

        if data_path.exists():
            data: dict[str, Any] = orjson.loads(data_path.read_bytes())
            for name, value in data.items():
                if type(cls._field_types[name]) is not type(BaseModel):
                    continue
                data[name] = cls._field_types[name].model_validate(value)       
        else:
            data = {}

        for field in cls.jsonl_fields:
            jsonl_path: Path = workspace / f"{field}.jsonl"
            if not jsonl_path.exists():
                data[field] = []
                continue
                
            with open(jsonl_path, "rb") as f:
                if type(cls._field_types[field]) is not type(BaseModel):
                    data[field] = list(orjsonl.load(f))
                    continue
                
                data[field] = list(
                    cls._field_types[field].model_validate(data) 
                    for data in orjsonl.load(f)
                )
                    
        obj: TarPersistentModel = cls(**data)

        obj._workspace = workspace
        obj._tar_path = tar_path

        return obj
