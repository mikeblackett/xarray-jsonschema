import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator

from xarray_model import Name

from xarray_model.testing import patterns


@hp.given(data=st.data())
def test_name_schema_is_valid(
    data: st.DataObject, validator: Validator
) -> None:
    """Should always produce a valid JSON Schema"""
    regex = data.draw(st.one_of(st.none(), st.booleans()))
    if regex:
        name = data.draw(patterns())
    else:
        name = data.draw(st.one_of(st.none(), st.text(), st.lists(st.text())))
    min_length = data.draw(st.integers(min_value=0))
    max_length = data.draw(st.integers(min_value=min_length))
    hp.assume(min_length < max_length)
    schema = Name(
        name=name,
        regex=regex,
        min_length=min_length,
        max_length=max_length,
    ).schema
    validator.check_schema(schema)
