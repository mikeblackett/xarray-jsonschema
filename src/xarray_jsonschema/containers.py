from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import ClassVar

import xarray as xr
from numpy.typing import DTypeLike

from xarray_jsonschema._normalizers import Normalizer, ObjectNormalizer
from xarray_jsonschema.base import XarraySchema, mapping_to_object_normalizer
from xarray_jsonschema.components import (
    AttrSchema,
    AttrsSchema,
    DimsSchema,
    DTypeSchema,
    NameSchema,
    ShapeSchema,
    SizeSchema,
)


class CoordsSchema(XarraySchema[xr.DataArray]):
    """Validate the coordinates of a ``xarray.DataArray``."""

    coords: Mapping[str, 'DataArraySchema']
    strict: bool

    def __init__(
        self,
        coords: Mapping[str, 'DataArraySchema'] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._set(coords={} if coords is None else coords, strict=strict)

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

    dims: DimsSchema | None
    attrs: AttrsSchema | None
    dtype: DTypeSchema | None
    shape: ShapeSchema | None
    coords: CoordsSchema | None
    name: NameSchema | None
    regex: bool
    required: bool

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
        super().__init__(title=title, description=description)
        self._set(
            dims=DimsSchema.from_python(dims) if dims else None,
            attrs=AttrsSchema.from_python(attrs) if attrs else None,
            dtype=DTypeSchema.from_python(dtype) if dtype else None,
            shape=ShapeSchema.from_python(shape) if shape else None,
            coords=CoordsSchema.from_python(coords) if coords else None,
            name=NameSchema.from_python(name) if name else None,
            regex=regex,
            required=required,
        )

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


class DataVarsSchema(XarraySchema[xr.Dataset]):
    """Validate the data variables of a ``xarray.Dataset``."""

    data_vars: Mapping[str, DataArraySchema]
    strict: bool

    def __init__(
        self,
        data_vars: Mapping[str, DataArraySchema] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._set(
            data_vars={} if data_vars is None else data_vars,
            strict=strict,
        )

    @cached_property
    def normalizer(self) -> Normalizer:
        return mapping_to_object_normalizer(self.data_vars, strict=self.strict)

    def validate(self, obj: xr.Dataset) -> None:
        instance = obj.to_dict(data=False)['data_vars']
        return super()._validate(instance=instance)


class DatasetSchema(XarraySchema[xr.Dataset]):
    """Validate a ``xarray.Dataset``.

    Parameters
    ----------
    data_vars : Mapping[str, DataArraySchema] | DataVarsSchema | None, default None
    dims : DimsSchema | None = None
    coords : Mapping[str, DataArraySchema] | CoordsSchema | None, default None
    attrs : Mapping[str, AttrSchema | object] | AttrsSchema | None, default None
    title : str | None, default None
    description : str | None, default None

    Attributes
    ----------
    coords : CoordsSchema | None, default None
    attrs : AttrsSchema | None, default None
    data_vars : DataVarsSchema | None, default None
    title : str | None, default None
    description : str | None, default None
    """

    _components: ClassVar[tuple[str, ...]] = (
        'coords',
        'attrs',
        'data_vars',
    )
    data_vars: DataVarsSchema | None
    dims: DimsSchema | None
    attrs: AttrsSchema | None
    coords: CoordsSchema | None

    def __init__(
        self,
        *,
        coords: Mapping[str, DataArraySchema] | CoordsSchema | None = None,
        data_vars: Mapping[str, DataArraySchema]
        | DataVarsSchema
        | None = None,
        dims: DimsSchema | None = None,
        attrs: Mapping[str, AttrSchema | object] | AttrsSchema | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(title=title, description=description)
        self._set(
            coords=CoordsSchema.from_python(coords) if coords else None,
            attrs=AttrsSchema.from_python(attrs) if attrs else None,
            data_vars=(
                DataVarsSchema.from_python(data_vars) if data_vars else None
            ),
        )

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

    def validate(self, obj: xr.Dataset) -> None:
        """Validate a ``Dataset`` against this schema.

        Parameters
        ----------
        obj : xr.Dataset
            The dataset instance to validate.

        Raises
        ------
        jsonschema.exceptions.ValidationError
            If the dataset instance is invalid.
        """
        instance = obj.to_dict(data=False)
        return super()._validate(instance=instance)
