from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import ClassVar

import xarray as xr
from numpy.typing import DTypeLike

from xarray_jsonschema import (
    AttrSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
    SizeSchema,
)
from xarray_jsonschema._normalizers import Normalizer, ObjectNormalizer
from xarray_jsonschema.base import XarraySchema, mapping_to_object_normalizer
from xarray_jsonschema.components import AttrsSchema


class CoordsSchema(XarraySchema):
    """Validate the coordinates of a ``xarray.DataArray``."""

    def __init__(
        self,
        coords: Mapping[str, 'DataArraySchema'] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.coords = {} if coords is None else coords
        self.strict = strict

    @cached_property
    def normalizer(self) -> Normalizer:
        return mapping_to_object_normalizer(self.coords, strict=self.strict)

    def validate(self, obj: xr.DataArray | xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['coords']
        return super()._validate(instance=instance)


class DataArraySchema(XarraySchema[xr.DataArray]):
    """Validate a ``xarray.DataArray``.

    Parameters
    ----------
    dims : Sequence[str | NameSchema] | DimsSchema | None, default None
    attrs : Mapping[str, AttrSchema | object] | AttrsSchema | None, default None
    dtype : DTypeLike | DTypeSchema | None, default None
    shape : Sequence[int | SizeSchema] | ShapeSchema | None, default None
    coords : Mapping[str, DataArraySchema] | CoordsSchema | None, default None
    name : str | Iterable[str] | NameSchema | None, default None
    regex : bool, default False
    required : bool, default True
    title : str | None, default None
    description : str | None, default None
    """

    _components: ClassVar[tuple[str, ...]] = (
        'dims',
        'attrs',
        'dtype',
        'shape',
        'coords',
        'name',
    )

    def __init__(
        self,
        *,
        dims: Sequence[str | NameSchema] | DimsSchema | None = None,
        attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None,
        dtype: DTypeLike | DTypeSchema | None = None,
        shape: Sequence[int | SizeSchema] | ShapeSchema | None = None,
        coords: Mapping[str, 'DataArraySchema'] | CoordsSchema | None = None,
        name: str | Iterable[str] | NameSchema | None = None,
        regex: bool = False,
        required: bool = True,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        self.dims = DimsSchema.from_python(dims) if dims else None
        self.attrs = AttrsSchema.from_python(attrs) if attrs else None
        self.dtype = DTypeSchema.from_python(dtype) if dtype else None
        self.shape = ShapeSchema.from_python(shape) if shape else None
        self.coords = CoordsSchema.from_python(coords) if coords else None
        self.name = NameSchema.from_python(name) if name else None
        self.regex = regex
        self.required = required

        super().__init__(title=title, description=description)

    @cached_property
    def normalizer(self) -> Normalizer:
        return ObjectNormalizer(
            title=self.title,
            description=self.description,
            properties={
                key: getattr(self, key).normalizer
                for key in self._components
                if getattr(self, key) is not None
            },
        )

    def validate(self, obj: xr.DataArray) -> None:
        """Validate a data-array against this schema.

        Parameters
        ----------
        obj : xr.DataArray
            The data-array instance to validate.

        Raises
        ------
        jsonschema.exceptions.ValidationError
            If the data-array instance is invalid.
        """
        instance = obj.to_dict(data=False)
        return super()._validate(instance=instance)
