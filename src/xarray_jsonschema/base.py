import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from typing import Any, ClassVar, Self

from jsonschema.protocols import Validator


from xarray_jsonschema.encoders import encode_value
from xarray_jsonschema.serializers import Serializer
from xarray_jsonschema.validators import XarrayModelValidator

__all__ = ['XarraySchema']


@dataclass(frozen=True, kw_only=True, repr=False)
class XarraySchema(ABC):
    """Abstract base class for xarray schema components.

    Parameters
    ----------
    key : str | None, default None
        A key to identify the schema programmatically.
    title : str | None, default None
        A human-readable label for the schema.
    description : str | None, default None
        An arbitrary textual description of the schema.
    """

    _validator: ClassVar[type[Validator]] = XarrayModelValidator

    key: str | None = None
    title: str | None = None
    description: str | None = None

    @cached_property
    @abstractmethod
    def serializer(self) -> Serializer:
        """The ``Serializer`` for this schema."""
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def json(self) -> dict[str, Any]:
        """The JSON schema for this object."""
        return self.serializer.serialize()

    @cached_property
    def validator(self) -> Validator:
        """The JSON Schema validator for this schema"""
        return self._validator(schema=self.json)

    def check_schema(self) -> None:
        """Validate this schema against the validator's meta-schema.

        Raises
        ------
        jsonschema.exceptions.SchemaError
            If the schema is invalid.
        """
        self._validator.check_schema(schema=self.json)

    def _validate(self, instance: Any) -> None:
        """A simple wrapper around ``jsonschema.Validator.validate``"""
        return self.validator.validate(instance=instance)

    @abstractmethod
    def validate(self, *args, **kwargs) -> None:
        """Validate an instance against this schema."""
        raise NotImplementedError  # pragma: no cover

    def __call__(self, *args, **kwargs) -> None:
        """Validate an instance against this schema."""
        return self.validate(*args, **kwargs)

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

    def to_dict(self):
        """Convert this schema to a Python dictionary."""
        return asdict(self, dict_factory=_custom_dict_factory)

    @classmethod
    def convert(cls, value: Any) -> Self:
        """Attempt to convert a value to this schema."""
        if isinstance(value, cls):
            return value
        return cls(value)  # type: ignore

    def __repr__(self):
        # Show only non-default arguments...
        args = [
            (f.name, getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) != f.default
        ]
        args_string = ''.join(f'{name}={value}' for name, value in args)
        return f'{self.__class__.__name__}({args_string})'


def _custom_dict_factory(data: list[tuple[str, Any]]):
    """Custom dict_factory for ``dataclasses.asdict()`` method which omits fields with values == None."""
    return {k: encode_value(v) for k, v in data if v is not None}
