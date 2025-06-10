import hypothesis as hp
import xarray as xr
import xarray.testing.strategies as xrst

from tests.test_containers._strategies import data_vars_models, dataset_models
from xarray_model.containers import DatasetModel, DataVars


class TestDatasetModel:
    @hp.given(model=dataset_models())
    def test_arguments(self, model: DatasetModel):
        """Should always produce a valid JSON Schema"""
        schema = model.schema

    @hp.given(variable=xrst.variables())
    def test_validates_with_defaults(self, variable: xr.Variable):
        """Should pass with default values."""
        DatasetModel().validate(variable)


class TestDataVarsModel:
    @hp.given(model=data_vars_models())
    def test_arguments(self, model: DataVars):
        """Should always produce a valid JSON Schema"""
        schema = model.schema
