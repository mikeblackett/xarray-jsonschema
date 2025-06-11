import hypothesis as hp
import pytest as pt
from hypothesis import strategies as st
from jsonschema import ValidationError, Validator

from xarray_jsonschema import ShapeSchema
from xarray_jsonschema.testing import dimension_shapes


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
        validator: Validator,
    ) -> None:
        """Should always produce a valid JSON Schema"""
        schema = ShapeSchema(shape, min_dims=min_dims, max_dims=max_dims)
        assert schema.check_schema() is None

    @hp.given(expected=dimension_shapes())
    def test_validation_with_sequence_passes(self, expected: tuple[int, ...]):
        """Should pass when the shape matches a sequence of integers."""
        instance = expected
        ShapeSchema(expected).validate(instance)

    @hp.given(expected=dimension_shapes(), instance=dimension_shapes())
    def test_validation_with_sequence_fails(
        self, expected: tuple[int, ...], instance: tuple[int, ...]
    ):
        """Should fail when the shape does not match a sequence of integers."""
        hp.assume(expected != instance)
        with pt.raises(ValidationError):
            ShapeSchema(expected).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_item_constraints_passes(
        self, data: st.DataObject
    ):
        """Should pass if the instance matches the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        instance = data.draw(
            dimension_shapes(min_dims=min_dims, max_dims=max_dims)
        )
        ShapeSchema(max_dims=max_dims, min_dims=min_dims).validate(instance)

    @hp.given(data=st.data())
    def test_validation_with_item_constraints_fails(self, data: st.DataObject):
        """Should fail if the instance does not match the item constraints."""
        min_dims = data.draw(st.integers(min_value=1, max_value=3))
        max_dims = data.draw(st.integers(min_value=min_dims, max_value=5))
        instance = data.draw(
            dimension_shapes(min_dims=min_dims, max_dims=max_dims)
        )
        with pt.raises(ValidationError):
            ShapeSchema(max_dims=min_dims - 1).validate(instance)
            ShapeSchema(min_dims=max_dims + 1).validate(instance)
