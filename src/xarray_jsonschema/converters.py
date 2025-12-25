"""This module provides custom `attrs` field converters"""

from typing import Callable

import numpy as np
import numpy.typing as npt


def optional_type(converter: type) -> Callable:
    """Optionally apply a type converter to the value.

    If the `converter` can accept instances of `type`, this method is not needed. You can just use `attrs.converter.optional()`.
    """

    def optional_converter(obj):
        if obj is None or isinstance(obj, converter):
            return obj
        return converter(obj)

    return optional_converter


def dtype(obj: npt.DTypeLike) -> str:
    """Return the canonical string representation of a numpy dtype.

    The returned string matches how xarray serializes dtypes with `to_dict(data=False)`
    """
    return str(np.dtype(obj))
