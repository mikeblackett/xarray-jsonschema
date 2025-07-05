import hypothesis as hp
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_jsonschema.dataset import DatasetSchema, DataVarsSchema


class TestDatasetModel:
    @hp.given(model=dataset_models())
    def test_arguments(self, model: DatasetSchema):
        """Should always produce a valid JSON Schema"""
        schema = model.schema

    @hp.given(variable=xrst.variables())
    def test_validates_with_defaults(self, variable: xr.Variable):
        """Should pass with default values."""
        DatasetSchema().validate(variable)


class TestDataVarsModel:
    @hp.given(model=data_vars_models())
    def test_arguments(self, model: DataVarsSchema):
        """Should always produce a valid JSON Schema"""
        schema = model.schema
