import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy.typing as npt

NameLike: TypeAlias = str | re.Pattern
DTypeLike = npt.DTypeLike
DimsLike: TypeAlias = Sequence[str | re.Pattern]
ShapeLike: TypeAlias = Sequence[int]
AttrsLike: TypeAlias = Mapping[str, Any]
