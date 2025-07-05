import hypothesis as hp
import pytest as pt
import xarray as xr
from hypothesis import strategies as st
from jsonschema import ValidationError
from xarray.testing.strategies import dimension_sizes

from xarray_jsonschema import ShapeSchema, SizeSchema
from xarray_jsonschema.testing import data_arrays, dimension_shapes


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


class TestShape:
    @hp.given(
        shape=dimension_shapes(min_dims=1),
        min_dims=st.integers(min_value=0),
        max_dims=st.integers(min_value=0),
    )
    def test_schema_is_valid(
        self,
        shape: tuple[int, ...],
        min_dims: int,
        max_dims: int,
    ) -> None:
        """Should always produce a valid JSON Schema"""
        schema = ShapeSchema(shape, min_dims=min_dims, max_dims=max_dims)
        assert schema.check_schema() is None

    @hp.given(instance=data_arrays())
    def test_validation_with_sequence_passes(self, instance: xr.DataArray):
        """Should pass when the shape matches a sequence of integers."""
        ShapeSchema(instance.shape).validate(instance)

    @hp.given(expected=dimension_shapes(), instance=data_arrays())
    def test_validation_with_sequence_fails(
        self, expected: tuple[int, ...], instance: xr.DataArray
    ):
        """Should fail when the shape does not match a sequence of integers."""
        hp.assume(expected != instance.shape)
        with pt.raises(ValidationError):
            ShapeSchema(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_dims_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the instance matches the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        instance = data.draw(
            data_arrays(
                dims=dimension_sizes(min_dims=min_dims, max_dims=max_dims)
            )
        )
        ShapeSchema(max_dims=max_dims, min_dims=min_dims).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_dims_constraints_fails(self, data: st.DataObject):
        """Should fail if the instance does not match the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        instance = data.draw(
            data_arrays(
                dims=dimension_sizes(min_dims=min_dims, max_dims=max_dims)
            )
        )
        with pt.raises(ValidationError):
            ShapeSchema(max_dims=min_dims - 1).validate(instance)
            ShapeSchema(min_dims=max_dims + 1).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_size_match_passes(self, data: st.DataObject):
        """Should pass if the instance dims match the expected sizes"""
        instance = data.draw(data_arrays())
        ShapeSchema(instance.shape).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_size_match_fails(self, data: st.DataObject):
        """Should fail if the instance dims do not match the expected sizes"""
        instance = data.draw(data_arrays(dims=dimension_sizes(min_dims=1)))
        expected = [dim + 1 for dim in instance.shape]
        with pt.raises(ValidationError):
            ShapeSchema(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_size_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the instance dims match the expected sizes"""
        min_side = data.draw(st.integers(min_value=1, max_value=3))
        max_side = data.draw(st.integers(min_value=min_side, max_value=5))
        instance = data.draw(
            data_arrays(
                dims=dimension_sizes(min_side=min_side, max_side=max_side)
            )
        )
        expected = [
            SizeSchema(minimum=min_side, maximum=max_side)
            for _ in instance.shape
        ]
        ShapeSchema(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_size_match_fails(self, data: st.DataObject):
        """Should fail if the instance dims do not match the expected sizes"""
        min_side = data.draw(st.integers(min_value=1, max_value=3))
        max_side = data.draw(st.integers(min_value=min_side, max_value=5))
        instance = data.draw(
            data_arrays(
                dims=dimension_sizes(min_side=min_side, max_side=max_side)
            )
        )
        expected = [
            SizeSchema(minimum=min_side + 1, maximum=max_side - 1)
            for _ in instance.shape
        ]
        with pt.raises(ValidationError):
            ShapeSchema(expected).validate(instance)
