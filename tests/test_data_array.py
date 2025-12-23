import hypothesis as hp
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import xarray as xr
import xarray.testing.strategies as xrst

from xarray_jsonschema import DataArrayModel

from .strategies import attrs, readable_text, supported_dtype_likes


class TestDataArrayModel:
    @hp.given(
        attrs=st.one_of(attrs(), st.none()),
        dims=st.one_of(xrst.dimension_names(), st.none()),
        shape=st.one_of(npst.array_shapes(), st.none()),
        name=st.one_of(readable_text(), st.none()),
        dtype=st.one_of(supported_dtype_likes(), st.none()),
    )
    def test_generates_valid_schema(
        self, dtype, name, shape, dims, attrs
    ) -> None:
        """Should produce valid JSON Model."""
        model = DataArrayModel(
            dtype=dtype, name=name, shape=shape, dims=dims, attrs=attrs
        )
        assert DataArrayModel.check_schema(model.to_schema()) is None

    def test_validation_with_default_parameters(self):
        """Should validate any data array when instantiated with default values."""
        da = xr.tutorial.open_dataset('air_temperature').air
        DataArrayModel().validate(da)
