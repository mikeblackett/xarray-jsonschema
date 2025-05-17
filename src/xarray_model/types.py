from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any, Literal
from numpy.typing import DTypeLike

__all__ = [
    'AttrType',
    'AttrsType',
    'DimsType',
    'DTypeLike',
    'SizesType',
    'ChunkType',
    'ChunksType',
    'NameType',
]

type JSONDataType = Literal[
    'array',
    'boolean',
    'const',
    'integer',
    'null',
    'number',
    'object',
    'pattern',
    'string',
]
type NameType = str | Sequence[str] | re.Pattern[str]
type DimsType = Iterable[str | None]
type SizesType = Mapping[str, int | None]
type ChunkType = bool | int | Sequence[int]
type ChunksType = bool | Mapping[str, ChunkType]
type AttrType = Mapping[str, Any] | None
type AttrsType = Mapping[str, AttrType]
