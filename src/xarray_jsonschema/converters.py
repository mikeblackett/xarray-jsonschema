from typing import Callable

import numpy as np
import numpy.typing as npt


def optional_type(converter: type) -> Callable:
    """Optionally apply a converter to the value.

    If the object is None or already an instance of the converter, it is returned as-is.
    Otherwise, the converter is applied to the object and the result is returned.

    This is only needed for optional Model instance
    """

    def optional_converter(obj):
        if obj is None or isinstance(obj, converter):
            return obj
        return converter(obj)

    return optional_converter


def dtype(obj: npt.DTypeLike) -> str:
    """Return the canonical string representation of a numpy dtype.

    This representation matches how xarray serializes dtypes in `to_dict(data=False)`
    """
    return str(np.dtype(obj))
