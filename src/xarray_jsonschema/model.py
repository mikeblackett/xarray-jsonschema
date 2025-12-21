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
    if isinstance(value, Model):
        return getattr(value, field.name, value)
    return value


def filter(attr: at.Attribute, value: object) -> bool:
    return not (attr.name.startswith('_') or value is None)


@at.define(kw_only=True, frozen=True)
class Model(ABC, Generic[TObj]):
    """A base class for validation models."""

    _validator: ClassVar = XarrayValidator
    """The JSON Schema validator class used for validation."""
    _builder: gs.SchemaBuilder = at.field(init=False, factory=SchemaBuilder)
    """The JSON Schema builder instance used for schema generation."""

    def __attrs_post_init__(self) -> None:
        self.build()

    @property
    def validator(self) -> jsp.Validator:
        """The validator instance for this model"""
        return self._validator(schema=self.to_schema())  # type: ignore

    @abstractmethod
    def build(self) -> None:
        """Build the schema object from this model's attributes."""
        return self._builder.add_object(self.to_dict())

    @abstractmethod
    def validate(self, obj: TObj) -> None:
        """Validate an object against this model's schema."""
        return self._validate(obj)

    def _validate(self, instance: Any) -> None:
        return self.validator.validate(instance=instance)

    def to_dict(self) -> dict[str, object]:
        """Return this model as a dictionary."""
        return at.asdict(
            self, filter=filter, value_serializer=value_serializer
        )

    def to_schema(self) -> Mapping[str, object]:
        """Return the JSON schema for this model."""
        return self._builder.to_schema()

    def to_json(self, *args, **kwargs) -> str:
        """Return the JSON schema for this model as a string."""
        return json.dumps(self.to_schema(), *args, **kwargs)

    def __call__(self, obj: TObj) -> None:
        """Validate an object against this model's schema."""
        return self.validate(obj)

    @classmethod
    def from_dict(cls, data: Mapping) -> Self:
        """Create a new instance from a dictionary."""
        keys = [attribute.name for attribute in at.fields(cls)]
        return cls(
            **{key: value for key, value in data.items() if key in keys}
        )

    @classmethod
    def check_schema(cls, schema: Mapping | bool) -> None:
        """Validate the given schema against this model's meta-schema."""
        cls._validator.check_schema(schema)  # type: ignore[reportArgumentType]


@at.define(kw_only=True, frozen=True)
class DTypeModel(Model[np.dtype]):
    """A model for validating xarray object ``dtype`` attributes."""

    dtype: DTypeLike | None = at.field(
        default=None, converter=converters.dtype, kw_only=False
    )

    def build(self) -> None:
        return self._builder.add_object(self.dtype)

    def validate(self, obj: np.dtype) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class NameModel(Model[str]):
    """A model for validating xarray object ``name`` attributes."""

    name: NameLike | None = at.field(default=str, kw_only=False)

    @classmethod
    def regex(cls, pattern: str) -> Self:
        return cls(re.compile(pattern))  # type: ignore[call-arg]

    def build(self) -> None:
        return self._builder.add_object(self.name)

    def validate(self, obj: str) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DimsModel(Model[Sequence]):
    """A model for validating xarray object ``dims`` attributes."""

    dims: DimsLike | None = at.field(factory=tuple, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.dims)

    def validate(self, obj: Sequence) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class ShapeModel(Model[Sequence]):
    """A model for validating xarray object ``shape`` attributes."""

    shape: ShapeLike | None = at.field(factory=tuple, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.shape)

    def validate(self, obj: Sequence) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class AttrsModel(Model[Mapping]):
    """A model for validating xarray object ``attrs`` attributes."""

    attrs: AttrsLike | None = at.field(factory=dict, kw_only=False)

    def build(self) -> None:
        return self._builder.add_object(self.attrs)

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class CoordsModel(Model):
    """A model for validating xarray object ``coords`` attributes."""

    coords: Mapping[str, DataArrayModel] | None = at.field(
        factory=dict, kw_only=False
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DataVarsModel(Model[Mapping]):
    """A model for validating xarray object ``data_vars`` attributes."""

    data_vars: Mapping[str, DataArrayModel] | None = at.field(
        factory=dict, kw_only=False
    )

    def build(self) -> None:
        return super().build()

    def validate(self, obj: Mapping) -> None:
        return super().validate(obj)


@at.define(kw_only=True, frozen=True)
class DataArrayModel(Model[xr.DataArray]):
    """A model for validating xarray ``DataArray`` objects."""

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
    """A model for validating xarray ``Dataset`` objects."""

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
