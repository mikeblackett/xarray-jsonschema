"""This module provides custom `attrs` field converters"""

from typing import Callable


def optional_type(converter: type) -> Callable:
    """Optionally apply a type converter to the value.

    If the `converter` can accept instances of `type`, this method is not needed. You can just use `attrs.converter.optional()`.
    """

    def optional_converter(obj):
        if obj is None or isinstance(obj, converter):
            return obj
        return converter(obj)

    return optional_converter
