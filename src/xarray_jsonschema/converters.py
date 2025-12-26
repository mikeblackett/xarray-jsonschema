"""This module provides custom `attrs` field converters"""

from typing import Callable, TypeVar

TOptionalType = TypeVar('TOptionalType')


def optional_type(
    converter: type[TOptionalType],
) -> Callable[[object], TOptionalType | None]:
    """Optionally apply a type converter to the value.

    If the `converter` can accept instances of its own `type`, then you can just use `attrs.converter.optional()`. This method is primarily used for converting sub-model instances.
    """

    def optional_converter(obj: object) -> TOptionalType | None:
        if obj is None or isinstance(obj, converter):
            return obj
        return converter(obj)  # type: ignore [reportCallIssue]

    return optional_converter
