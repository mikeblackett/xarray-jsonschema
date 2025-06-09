import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator

from xarray_model import Size


@hp.given(
    size=st.integers(min_value=0),
    multiple_of=st.integers(min_value=0),
    maximum=st.integers(min_value=0),
    minimum=st.integers(min_value=0),
)
def test_size_schema_is_valid(
    size: int,
    multiple_of: int,
    maximum: int,
    minimum: int,
    validator: Validator,
) -> None:
    """Should always produce a valid JSON Schema"""
    schema = Size(
        size, multiple_of=multiple_of, maximum=maximum, minimum=minimum
    ).schema
    validator.check_schema(schema)
