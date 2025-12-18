import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, Self

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from xarray.core.types import T_Xarray

from xarray_jsonschema._normalizers import Normalizer, ObjectNormalizer
from xarray_jsonschema.validator import XarrayValidator

__all__ = ['XarraySchema', 'ValidationError', 'SchemaError']

# TODO: (mike) extend jsonschema exceptions.


class XarraySchema(Generic[T_Xarray], ABC):
    """Abstract base class for xarray schema objects.

    An ``XarraySchema`` normalizes its attributes to a JSON Schema compatible
    schema object which can be used to validate xarray objects.

    Parameters
    ----------
    key : str | None, default None
        A key to identify the schema programmatically.
    title : str | None, default None
        A human-readable label for the schema.
    description : str | None, default None
        An arbitrary textual description of the schema.
    """

    _validator: ClassVar = XarrayValidator

    key: str | None
    """A key to identify the schema programmatically."""
    title: str | None
    """A human-readable label for the schema."""
    description: str | None
    """An arbitrary textual description of the schema."""

    def __init__(
        self,
        key: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        self._set(key=key, title=title, description=description)

    def __call__(self, obj: T_Xarray) -> None:
        """Validate an xarray object against this schema."""
        return self.validate(obj)

    def __setattr__(self, name, value):
        raise AttributeError(f'{self.__class__.__name__} is immutable')

    def _set(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    @abstractmethod
    def validate(self, obj: T_Xarray) -> None:
        """Validate an xarray object against this schema.

        Parameters
        ------
        obj : T_Xarray
            The xarray object to validate.

        Raises
        ------
        ValidationError
            If the object does not validate against this schema.
        """
        raise NotImplementedError  # pragma: no cover

    @cached_property
    @abstractmethod
    def _normalizer(self) -> Normalizer:
        """The normalizer for this schema."""
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def json(self) -> dict[str, Any]:
        """The JSON Schema for this schema."""
        return self._normalizer.schema

    @cached_property
    def validator(self) -> Validator:
        """The JSON Schema validator class for this schema"""
        return self._validator(schema=self.json)  # type: ignore

    def check_schema(self) -> None:
        """Validate this schema against the validator's meta-schema.

        Raises
        ------
        :py:class:`~.SchemaError`
            If the schema is invalid.
        """
        self._validator.check_schema(schema=self.json)

    def dumps(self, **kwargs) -> str:
        """Serialize this schema to a JSON formatted string.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments to pass to ``json.dumps``.

        Returns
        -------
        str
            The schema serialized as a JSON formatted string.
        """
        return json.dumps(self.json, **kwargs)

    def _validate(self, instance: Any) -> None:
        """A simple wrapper around ``jsonschema.Validator.validate``.

        Subclasses should call this method in their ``validate`` method.
        """
        return self.validator.validate(instance=instance)

    @classmethod
    def from_python(cls, value: Any) -> Self:
        """Create an instance of this schema from a Python value.

        Parameters
        ----------
        value : Any
            A Python value to convert.
            If ``value`` is already an instance of this schema, it will be returned as-is.

        Raises
        ------
        ValueError
            If the conversion fails.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except Exception as error:
            raise ValueError(
                f'failed to convert {value!r} to {cls.__name__}'
            ) from error

    def __repr__(self):
        # Show only non-default arguments...
        args = [
            (f.name, f.value)
            for f in fields(self)
            if f.value is not None and f.value is not f.default
        ]
        args_string = ', '.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'


@dataclass
class Field:
    name: str
    default: Any
    value: Any


def fields(obj: XarraySchema) -> tuple['Field', ...]:
    """A tuple of ``Field`` instances for this schema.

    Originally I (mike) wanted to use dataclasses for XarraySchema. But
    performing parameter type conversions in `__post_init__` caused too many
    type issues, so I refactored to use normal Python classes. This `fields`
    and the associated `Fields` dataclass are very rudimentary
    implementations of the similarly named features from the standard
    library dataclasses. Their purpose is to gather parameter names, values
     and default values, nothing more.
    """
    signature = inspect.signature(obj.__init__)
    return tuple(
        Field(name=k, default=v.default, value=getattr(obj, v.name))
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    )


def mapping_to_object_normalizer(
    data: Mapping[str, XarraySchema], *, strict: bool = False
) -> ObjectNormalizer:
    """Convert a mapping of schema components to an ``ObjectNormalizer``
    instance.

    Parameters
    ----------
    data : Mapping[str, XarraySchema]
        A mapping of schema components. If the schema component has a regex attribute,
        the key will be treated as a regex pattern.
    strict : bool, default False
        A flag indicating if additional properties should be allowed.
    """
    properties = {}
    pattern_properties = {}
    required = set()
    required_pattern_properties = set()
    for key, schema in data.items():
        if getattr(schema, 'regex', False):
            pattern_properties[key] = schema._normalizer
        else:
            properties[key] = schema._normalizer
            if getattr(schema, 'required', False):
                required.add(key)
    return ObjectNormalizer(
        properties=properties or None,
        pattern_properties=pattern_properties or None,
        required=required or None,
        required_pattern_properties=required_pattern_properties or None,
        additional_properties=not strict,
    )
