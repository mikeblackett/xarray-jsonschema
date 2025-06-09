from typing import Any
import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator

from xarray_model import Attr, Attrs
from xarray_model.testing import attrs, readable_text


@hp.given(
    key=readable_text(),
    regex=st.booleans(),
    value=st.one_of(
        st.none(), st.booleans(), st.integers(), st.floats(), st.text()
    ),
    required=st.booleans(),
    data=st.data(),
)
def test_attr_schema_is_valid(
    validator: Validator,
    key: str,
    regex: bool,
    value: Any,
    required: bool,
    data: st.DataObject,
):
    """Should always produce a valid JSON Schema"""
    if data.draw(st.booleans()) and value is not None:
        value = type(value)
    schema = Attr(key, regex=regex, value=value, required=required).schema
    validator.check_schema(schema)


@hp.given(attrs_=attrs(), allow_extra_items=st.booleans(), data=st.data())
def test_attrs_schema_is_valid(
    validator: Validator,
    attrs_: dict,
    allow_extra_items: bool,
    data: st.DataObject,
):
    """Should always produce a valid JSON Schema"""
    expected = [
        Attr(
            key,
            value=value if data.draw(st.booleans()) else type(value),
            required=data.draw(st.booleans()),
            regex=data.draw(st.booleans()),
        )
        for key, value in attrs_.items()
    ]
    schema = Attrs(expected, allow_extra_items=allow_extra_items).schema
    validator.check_schema(schema)
