import hypothesis as hp
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import strategies as st

from tests.test_containers.test_data_array import (
    attrs_models,
    coords_models,
    data_array_models,
)
from xarray_model.containers import CoordsModel, DataVarsModel, DatasetModel


@st.composite
def dataset_models(draw: st.DrawFn):
    attrs = draw(attrs_models())
    coords = draw(coords_models())
    data_vars = draw(data_vars_models())

    return DatasetModel(
        attrs=attrs,
        coords=coords,
        data_vars=data_vars,
    )


@st.composite
def data_vars_models(draw: st.DrawFn) -> DataVarsModel:
    data_vars = draw(
        st.dictionaries(keys=xrst.names(), values=data_array_models())
    )
    require_all_keys = draw(st.booleans())
    allow_extra_keys = draw(st.booleans())
    return DataVarsModel(
        data_vars,
        require_all_keys=require_all_keys,
        allow_extra_keys=allow_extra_keys,
    )


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
    def test_arguments(self, model: DataVarsModel):
        """Should always produce a valid JSON Schema"""
        schema = model.schema

    @hp.given(data=st.data())
    def test_validates_with_defaults(self, data: st.DataObject):
        """Should pass with default values."""
        vars = data.draw(
            st.dictionaries(
                keys=xrst.names(), values=xrst.variables(), max_size=1
            )
        )
        coords = xr.Coordinates(
            {k: xr.DataArray(data=v, name=k) for k, v in vars.items()}
        )
        CoordsModel().validate(coords)
