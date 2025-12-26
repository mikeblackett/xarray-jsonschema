"""This module defines a base class for defining models of Python objects that can be used to validate object instances using JSON Schema.

Concrete implementations are provided for Xarray data structures and their key features.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Generic, Self, TypeAlias, TypeVar

import attrs as at
import genson as gs
import jsonschema.protocols as jsp
import numpy as np
import numpy.typing as npt
import xarray as xr

from xarray_jsonschema import converters
from xarray_jsonschema.schema import (
    SchemaBuilder,
)
from xarray_jsonschema.validator import XarrayValidator

__all__ = [
    'AttrsModel',
    'CoordsModel',
    'DataArrayModel',
    'DatasetModel',
    'DataVarsModel',
    'DimsModel',
    'DTypeModel',
    'NameModel',
    'ShapeModel',
    'Model',
]


NameLike: TypeAlias = str | type[str] | re.Pattern
DTypeLike = npt.DTypeLike
DimsLike: TypeAlias = Sequence[str | type[str] | re.Pattern]
ShapeLike: TypeAlias = Sequence[int]
AttrsLike: TypeAlias = Mapping[str, Any]

TObj = TypeVar('TObj')


def value_serializer(
    instance: type, field: at.Attribute, value: object
) -> object:
    """Serialize a model field.

    Unwraps `value` if it is a model instance containing an attribute with
    the same name as `field`.
    """
    if isinstance(value, Model):
        return getattr(value, field.name, value)
    return value


def filter(attr: at.Attribute, value: object) -> bool:
    """Return `False` if the attribute is private or optional.

    Parameters
    ----------
    attr : at.Attribute
        The attribute being validated.
    value : object
        The value of the attribute being validated.
    """
    return not (attr.name.startswith('_') or value is None)


@at.define(kw_only=True, frozen=True)
class Model(ABC, Generic[TObj]):
    """A base class for validation models."""

    _validator: ClassVar = XarrayValidator
    """The JSON Schema validator class used for validation."""
    _builder: gs.SchemaBuilder = at.field(init=False, factory=SchemaBuilder)
    """The JSON Schema builder instance used for schema generation."""

    def __attrs_post_init__(self) -> None:
        # Models are immutable so we build once only.
        self.build()

    @property
    def validator(self) -> jsp.Validator:
        """The validator instance for this model"""
        return self._validator(schema=self.to_schema())  # type: ignore

    @abstractmethod
    def build(self) -> None:
        """Build the schema object for this model."""
        return self._builder.add_object(self.to_dict())

    @abstractmethod
    def validate(self, obj: TObj) -> None:
        """Validate an object against this model's schema.

        Parameters
        ----------
        obj : TObj
           The object to validate.

        Raises
        ------
        ValidationError
           If the object does not match the schema.
        """
        return self._validate(obj)

    def _validate(self, instance: Any) -> None:
        return self.validator.validate(instance=instance)

    def to_schema(self) -> Mapping[str, object]:
        """Return the JSON schema for this model.

        Returns
        -------
        Mapping[str, object]
            The JSON schema representation of this model.
        """
        return self._builder.to_schema()

    def to_dict(self) -> dict[str, object]:
        """Return this model as a dictionary.

        Returns
        -------
        dict[str, object]
           The dictionary representation of this model.
        """
        return at.asdict(
            self, filter=filter, value_serializer=value_serializer
        )

    def to_json(self, *args, **kwargs) -> str:
        """Return the JSON schema for this model as a string.

        Parameters
        ----------
        *args : Any
            Additional arguments to pass to ``json.dumps``.
        **kwargs : Any
            Additional keyword arguments to pass to ``json.dumps``.

        Returns
        -------
        str
            The JSON schema representation of this model as a
        """
        return json.dumps(self.to_schema(), *args, **kwargs)

    def __call__(self, obj: TObj) -> None:
        """Validate an object against this model's schema.

        Parameters
        ----------
        obj : TObj
          The object to validate.

        Raises
        ------
        ValidationError
           If the object does not match the schema.
        """
        return self.validate(obj)

    @classmethod
    def from_dict(cls, data: Mapping) -> Self:
        """Create a new instance from a dictionary.

        Parameters
        ----------
        data : Mapping
          The dictionary to create the instance from.

        Returns
        -------
        Self
          A new instance of this class.
        """
        keys = [attribute.name for attribute in at.fields(cls)]
        return cls(
            **{key: value for key, value in data.items() if key in keys}
        )

    @classmethod
    def check_schema(cls, schema: Mapping) -> None:
        """Validate the given schema against this model's meta-schema.

        Parameters
        ----------
        schema : Mapping
            The schema to validate.

        Raises
        ------
        SchemaError
           If the schema does not match the meta schema.
        """
        cls._validator.check_schema(schema)  # type: ignore[reportArgumentType]


@at.define(kw_only=True, frozen=True)
class DTypeModel(Model[np.dtype]):
    """A model for validating xarray object ``dtype`` attributes.

    Parameters
    ----------
    dtype : DTypeLike | None
        The expected dtype value. Must be coercible by `numpy.dtype()`.
        Passing `None` will produce the default numpy dtype (float64).

    Attributes
    ----------
    dtype : np.dtype
        The expected numpy dtype.
    """

    dtype: DTypeLike | None = at.field(
        default=None, converter=np.dtype, kw_only=False
    )

    def build(self) -> None:
        return self._builder.add_object(self.dtype)

    def validate(self, obj: np.dtype) -> None:
        return super()._validate(str(obj))


@at.define(kw_only=True, frozen=True)
class NameModel(Model[str]):
    """A model for validating xarray object ``name`` attributes.

    Parameters
    ----------
    name : NameLike | None
        The expected name.

    Attributes
    ----------
    name
    """

    name: NameLike | None = at.field(default=str, kw_only=False)

    @classmethod
    def regex(cls, pattern: str) -> Self:
        """Create a new instance with a regex pattern to validate the name.

        Parameters
        ----------
        pattern : str
            The regex pattern to validate the name.

        Returns
        -------
        NameModel[str]
            A new instance parameterized with a regex pattern.
        """
        return cls(re.compile(pattern))  # type: ignore[call-arg]

    def build(self) -> None:
        return self._builder.add_object(self.name)

    def validate(self, obj: str) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DimsModel(Model[Sequence]):
    """A model for validating xarray object ``dims`` attributes.

    Parameters
    ----------
    dims : DimsLike | None
        The expected sequence of dimension names. Order matters.

    Attributes
    ----------
    dims
    """

    dims: DimsLike | None = at.field(factory=tuple, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.dims)

    def validate(self, obj: Sequence) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class ShapeModel(Model[Sequence]):
    """A model for validating xarray object ``shape`` attributes.

    Parameters
    ----------
    shape : ShapeLike | None
       The expected sequence of dimension sizes. Order matters.

    Attributes
    ----------
    shape
    """

    shape: ShapeLike | None = at.field(factory=tuple, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.shape)

    def validate(self, obj: Sequence) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class AttrsModel(Model[Mapping]):
    """A model for validating xarray object ``attrs`` attributes.

    Parameters
    ----------
    attrs : AttrsLike | None
        The expected attributes.

    Attributes
    ----------
    attrs
    """

    attrs: AttrsLike | None = at.field(factory=dict, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.attrs)

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class CoordsModel(Model[Mapping]):
    """A model for validating xarray object ``coords`` attributes.

    Parameters
    ----------
    coords : Mapping[str, DataArrayModel] | None
        The expected coordinates.

    Attributes
    ----------
    coords
    """

    coords: Mapping[str, DataArrayModel] | None = at.field(
        factory=dict, kw_only=False
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DataVarsModel(Model[Mapping]):
    """A model for validating xarray object ``data_vars`` attributes.

    Parameters
    ----------
    data_vars : Mapping[str, DataArrayModel]
        The expected data variables.

    Attributes
    ----------
    data_vars
    """

    data_vars: Mapping[str, DataArrayModel] | None = at.field(
        factory=dict, kw_only=False
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DataArrayModel(Model[xr.DataArray]):
    """A model for validating xarray ``DataArray`` objects.

    Parameters
    ----------
    dtype : DTypeLike | DTypeModel | None
        The expected data type.
    coords : Mapping[str, DataArrayModel]
        The expected coordinates.
    dims : DimsLike | DimsModel | None
        The expected dimensions.
    name : NameLike | NameModel | None
       The expected name.
    shape : ShapeLike | ShapeModel | None
        The expected shape.
    attrs : AttrsLike | AttrsModel | None
       The expected attributes.

    Attributes
    ----------
    dtype : DTypeModel | None
        Sub-model describing the array's data type.
    coords : CoordsModel | None
        Sub-model describing the array's coordinates.
    dims : DimsModel | None
        Sub-model describing the array's dimensions.
    name : NameModel | None
        Sub-model describing the array's name.
    shape : ShapeModel | None
        Sub-model describing the array's shape.
    attrs : AttrsModel | None
        Sub-model describing the array's attributes.
    """

    dtype: DTypeLike | DTypeModel | None = at.field(
        default=None, converter=converters.optional_type(DTypeModel)
    )
    coords: Mapping[str, DataArrayModel] | CoordsModel | None = at.field(
        default=None,
        converter=converters.optional_type(CoordsModel),
    )
    dims: DimsLike | DimsModel | None = at.field(
        default=None,
        converter=converters.optional_type(DimsModel),
    )
    name: NameLike | NameModel | None = at.field(
        default=None,
        converter=converters.optional_type(NameModel),
    )
    shape: ShapeLike | ShapeModel | None = at.field(
        default=None,
        converter=converters.optional_type(ShapeModel),
    )
    attrs: AttrsLike | AttrsModel | None = at.field(
        default=None,
        converter=converters.optional_type(AttrsModel),
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: xr.DataArray) -> None:
        return self._validate(obj.to_dict(data=False))


@at.define(kw_only=True, frozen=True)
class DatasetModel(Model[xr.Dataset]):
    """A model for validating xarray ``Dataset`` objects.

    Parameters
    ----------
    data_vars : Mapping[str, DataArrayModel]
        The expected data variables.
    coords : Mapping[str, DataArrayModel]
        The expected coordinates.
    attrs : AttrsLike | AttrsModel | None
       The expected attributes.

    Attributes
    ----------
    coords : CoordsModel | None
        Sub-model describing the array's coordinates.
    data_vars : DataVarsModel | None
       Sub-model describing the array's data variables.
    attrs : AttrsModel | None
        Sub-model describing the array's attributes.
    """

    data_vars: Mapping[str, DataArrayModel] | DataVarsModel | None = at.field(
        default=None,
        converter=converters.optional_type(DataVarsModel),
    )
    coords: Mapping[str, DataArrayModel] | CoordsModel | None = at.field(
        default=None,
        converter=converters.optional_type(CoordsModel),
    )
    attrs: AttrsLike | AttrsModel | None = at.field(
        default=None,
        converter=converters.optional_type(AttrsModel),
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: xr.Dataset) -> None:
        return self._validate(obj.to_dict(data=False))
