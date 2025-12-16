import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Generic, Self

from jsonschema.protocols import Validator
from xarray.core.types import T_Xarray

from xarray_jsonschema._normalizers import Normalizer, ObjectNormalizer
from xarray_jsonschema.validator import XarrayValidator

__all__ = ['XarraySchema']


class XarraySchema(ABC, Generic[T_Xarray]):
    """Abstract base class for xarray schema components."""

    _validator: ClassVar = XarrayValidator

    def __init__(
        self,
        key: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        """Construct a new schema instance.

        Parameters
        ----------
        key : str | None, default None
            A key to identify the schema programmatically.
        title : str | None, default None
            A human-readable label for the schema.
        description : str | None, default None
            An arbitrary textual description of the schema.
        """
        self.key = key
        self.title = title
        self.description = description

    def __call__(self, obj: T_Xarray) -> None:
        """Validate an instance against this schema."""
        return self.validate(obj)

    @abstractmethod
    def validate(self, obj: T_Xarray) -> None:
        """Validate an instance against this schema."""
        raise NotImplementedError  # pragma: no cover

    @cached_property
    @abstractmethod
    def normalizer(self) -> Normalizer:
        """The ``Normalizer`` instance for this schema."""
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def json(self) -> dict[str, Any]:
        """The JSON schema for this object."""
        return self.normalizer.normalize()

    @cached_property
    def validator(self) -> Validator:
        """The JSON Schema ``Validator`` for this schema"""
        return self._validator(schema=self.json)  # type: ignore

    def check_schema(self) -> None:
        """Validate this schema against the validator's meta-schema.

        Raises
        ------
        :py:class:`jsonschema.exceptions.SchemaError`
            If the schema is invalid.
        """
        self._validator.check_schema(schema=self.json)

    def dumps(self, **kwargs) -> str:
        """Serialize this schema to a JSON formatted ``str``.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments to pass to ``json.dumps``.

        Returns
        -------
        str
            The schema serialized as a JSON formatted ``str``.
        """
        return json.dumps(self.json, **kwargs)

    def _validate(self, instance: Any) -> None:
        """A simple wrapper around ``jsonschema.Validator.validate``

        Subclass should call this method in their ``validate`` method.
        """
        return self.validator.validate(instance=instance)

    @classmethod
    def from_python(cls, value: Any) -> Self:
        """Attempt to convert a value to this schema."""
        if isinstance(value, cls):
            return value
        return cls(value)

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
    type issues, so I refactored to use normal Python classes. This `_fields`
    property and the associated `Fields` dataclass are very rudimentary
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
            pattern_properties[key] = schema.normalizer
        else:
            properties[key] = schema.normalizer
            if getattr(schema, 'required', False):
                required.add(key)
    return ObjectNormalizer(
        properties=properties or None,
        pattern_properties=pattern_properties or None,
        required=required or None,
        required_pattern_properties=required_pattern_properties or None,
        additional_properties=not strict,
    )
