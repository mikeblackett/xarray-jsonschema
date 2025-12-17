import hypothesis as hp
import xarray as xr

from tests._strategies import coords_schemas, data_array_schemas
from xarray_jsonschema import CoordsSchema, DataArraySchema
from xarray_jsonschema.testing import data_arrays


class TestDataArraySchema:
    @hp.given(model=data_array_schemas())
    def test_schema_is_valid(self, model: DataArraySchema):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        assert model.check_schema() is None

    @hp.given(da=data_arrays())
    def test_validation_with_default_parameters_passes(self, da: xr.DataArray):
        """Should always pass with default values."""
        DataArraySchema().validate(da)


class TestCoordsModel:
    @hp.given(model=coords_schemas())
    def test_schema_is_valid(self, model: CoordsSchema):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        assert model.check_schema() is None

    @hp.given(da=data_arrays())
    def test_validation_with_default_parameters_passes(self, da: xr.DataArray):
        """Should always pass with default values."""
        CoordsSchema().validate(da)
