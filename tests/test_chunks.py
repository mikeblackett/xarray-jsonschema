# from collections.abc import Sequence

# import hypothesis as hp
# import numpy as np
# import pytest as pt
# import xarray.testing.strategies as xrst
# from hypothesis import strategies as st
# from jsonschema import ValidationError

# from xarray_jsonschema import ChunksSchema
# from xarray_jsonschema.testing import data_arrays


# class TestChunks:
#     @hp.given(
#         expected=st.one_of(
#             st.booleans(),
#             st.integers(min_value=0),
#             st.lists(st.integers(min_value=0)),
#             st.lists(st.lists(st.integers(min_value=0))),
#         )
#     )
#     def test_chunks_schema_is_valid(
#         self,
#         expected: bool | int | Sequence[int | Sequence[int]],
#     ):
#         """Should always produce a valid JSON Schema"""
#         schema = ChunksSchema(expected)
#         assert schema.check_schema() is None

#     @hp.given(data=st.data())
#     def test_validation_with_boolean_passes(self, data: st.DataObject):
#         """Should pass when the chunked/unchunked state matches a boolean."""
#         expected = data.draw(st.booleans())
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         if expected:
#             da = da.chunk('auto')
#         ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_boolean_fails(self, data: st.DataObject):
#         """Should fail when the chunked state does not match a boolean."""
#         expected = data.draw(st.booleans())
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         if not expected:
#             da = da.chunk('auto')
#         with pt.raises(ValidationError):
#             ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_integer_passes(self, data: st.DataObject):
#         """Should pass if the block sizes of all dimensions match an integer."""
#         expected = data.draw(st.integers(min_value=1, max_value=10))
#         da = data.draw(
#             data_arrays(
#                 dims=xrst.dimension_sizes(min_dims=1, min_side=expected)
#             )
#         )
#         da = da.chunk(expected)
#         ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_integer_fails(self, data: st.DataObject):
#         """Should fail if the block sizes of all dimensions do not match an integer."""
#         expected = data.draw(st.integers(min_value=1, max_value=10))
#         da = data.draw(
#             data_arrays(
#                 dims=xrst.dimension_sizes(min_dims=1, min_side=expected)
#             )
#         )
#         da = da.chunk(1)
#         with pt.raises(ValidationError):
#             ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_sequence_of_integers_passes(
#         self, data: st.DataObject
#     ):
#         """Should pass if the block sizes of each dimension match a sequence of integers."""
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         da = da.chunk('auto')
#         assert da.chunks is not None
#         expected = [c[0] for c in da.chunks]
#         ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_sequence_of_integers_fails(
#         self, data: st.DataObject
#     ):
#         """Should fail if the block sizes of each dimension do not match a sequence of integers."""
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         da = da.chunk('auto')
#         assert da.chunks is not None
#         expected = expected = [c[0] + 1 for c in da.chunks]
#         with pt.raises(ValidationError):
#             ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_sequence_of_sequence_of_integers_passes(
#         self, data: st.DataObject
#     ):
#         """
#         Test exact block sizes per dimension.

#         It is equivalent to testing the output of:
#         `DataArray(dims=('x', 'y'), ...).chunk(x=(1, 2, 3), y=(1, 2, 3)).chunks`
#         """
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         da = da.chunk('auto')
#         assert da.chunks is not None
#         expected = da.chunks
#         ChunksSchema(expected).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_sequence_of_sequence_of_integers_fails(
#         self, data: st.DataObject
#     ):
#         """Should fail if the block sizes of each dimension do not match a sequence of integers."""
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         da = da.chunk('auto')
#         assert da.chunks is not None
#         expected = np.array(da.chunks) + 1
#         with pt.raises(ValidationError):
#             ChunksSchema(expected.tolist()).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_wildcard_passes(self, data: st.DataObject):
#         da = data.draw(data_arrays(dims=xrst.dimension_sizes(min_dims=1)))
#         da = da.chunk(-1)
#         ChunksSchema(-1).validate(da.chunks)

#     @hp.given(data=st.data())
#     def test_validation_with_wildcard_fails(self, data: st.DataObject):
#         da = data.draw(
#             data_arrays(dims=xrst.dimension_sizes(min_dims=1, min_side=2))
#         )
#         da = da.chunk(1)
#         with pt.raises(ValidationError):
#             ChunksSchema(-1).validate(da.chunks)
