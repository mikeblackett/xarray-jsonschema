import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator
from xarray_model import Dims

import xarray.testing.strategies as xrst


@hp.given(data=st.data())
def test_dims_schema_is_valid(
    data: st.DataObject, validator: Validator
) -> None:
    """Should always produce a valid JSON Schema"""
    dims = data.draw(st.one_of(st.none(), xrst.dimension_names()))
    contains = data.draw(st.one_of(st.none(), st.text()))
    min_dims = data.draw(st.integers(min_value=0))
    max_dims = data.draw(st.integers(min_value=min_dims))
    schema = Dims(
        dims, contains=contains, max_dims=max_dims, min_dims=min_dims
    ).schema
    validator.check_schema(schema)
