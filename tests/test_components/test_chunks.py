from collections.abc import Sequence

import hypothesis as hp
from hypothesis import strategies as st
from jsonschema import Validator

from xarray_model import Chunks
from xarray_model.components import _Chunk


@hp.given(shape=st.one_of(st.integers(), st.lists(st.integers())))
def test_chunk_schema_is_valid(
    shape: int | Sequence[int], validator: Validator
):
    """Should always produce a valid JSON Schema"""
    schema = _Chunk(shape).schema
    validator.check_schema(schema)


@hp.given(
    expected=st.one_of(
        st.booleans(),
        st.integers(min_value=0),
        st.lists(st.integers(min_value=0)),
        st.lists(st.lists(st.integers(min_value=0))),
    )
)
def test_chunks_schema_is_valid(
    expected: bool | int | Sequence[int | Sequence[int]], validator: Validator
):
    """Should always produce a valid JSON Schema"""
    schema = Chunks(expected).schema
    validator.check_schema(schema)
