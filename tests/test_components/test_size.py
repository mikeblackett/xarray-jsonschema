import hypothesis as hp
from hypothesis import strategies as st

from xarray_jsonschema import SizeSchema


@hp.given(
    size=st.integers(min_value=0),
    maximum=st.integers(min_value=0),
    minimum=st.integers(min_value=0),
)
def test_size_schema_is_valid(
    size: int,
    maximum: int,
    minimum: int,
) -> None:
    """Should always produce a valid JSON Schema"""
    schema = SizeSchema(size, maximum=maximum, minimum=minimum)
    assert schema.check_schema() is None
