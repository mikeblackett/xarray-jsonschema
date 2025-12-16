import hypothesis as hp
import xarray as xr
import xarray.testing.strategies as xrst

from tests._strategies import data_vars_schemas, dataset_schemas
from xarray_jsonschema.dataset import DatasetSchema, DataVarsSchema


class TestDatasetModel:
    @hp.given(model=dataset_schemas())
    def test_schema_is_valid(self, model: DatasetSchema):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        assert model.check_schema() is None

    @hp.given(instance=xrst.variables())
    def test_validates_with_defaults(self, instance: xr.Variable):
        """Should pass with default values."""
        DatasetSchema().validate(instance)


class TestDataVarsModel:
    @hp.given(model=data_vars_schemas())
    def test_schema_is_valid(self, model: DataVarsSchema):
        """Should always produce a valid JSON Schema with any combination of parameters."""
        assert model.check_schema() is None
